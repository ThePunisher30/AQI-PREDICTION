"""WA-STGN — Wind-Aware Spatio-Temporal Graph Network.

Unlike WM-STGN where wind is just a node feature, WA-STGN uses atmospheric
physics to compute DYNAMIC per-timestep edge weights:

1. Bidirectional advection: wind at source pushes pollution toward destination,
   wind at destination receives from source direction (inspired by MSDGNN).
2. Pressure gradient: high→low pressure strengthens transport (inspired by TransNet).
3. Diffusion: symmetric spatial smoothing via graph Laplacian with learned
   per-feature coefficients (inspired by TransNet's K=[κ₁,κ₂,...]).
4. Distance decay: transport weakens with distance.

A_final(t) = A_fixed + A_adaptive + α*A_transport(t) + γ*A_diffusion

Learnable parameters: α (advection weight), β (pressure sensitivity),
γ (diffusion weight), κ (per-feature diffusivity).
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Load city coordinates for direction vectors
_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
_META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "meta.json")


def _build_direction_and_distance():
    """Precompute normalized direction vectors and distances between all city pairs."""
    with open(_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    with open(_META_PATH) as f:
        meta = json.load(f)
    cities = meta["city_names"]
    locs = cfg["locations"]
    n = len(cities)
    # direction[i,j] = (lon_j - lon_i, lat_j - lat_i) — unnormalized
    dir_lon = np.zeros((n, n), dtype=np.float32)
    dir_lat = np.zeros((n, n), dtype=np.float32)
    dist = np.zeros((n, n), dtype=np.float32)
    for i, ci in enumerate(cities):
        for j, cj in enumerate(cities):
            if i != j:
                dlat = locs[cj][0] - locs[ci][0]
                dlon = locs[cj][1] - locs[ci][1]
                d = np.sqrt(dlat**2 + dlon**2) + 1e-6
                dir_lon[i, j] = dlon / d  # normalized
                dir_lat[i, j] = dlat / d
                dist[i, j] = d
    return dir_lon, dir_lat, dist


class WindAwareAdjacency(nn.Module):
    """Dynamic adjacency based on atmospheric transport physics.

    A_final(t) = A_fixed + A_adaptive + α * A_transport(t)

    Transport model (inspired by TransNet, MSDGNN):
    1. Advection: wind at source pushes pollution toward destination,
       wind at destination receives from source direction (bidirectional).
    2. Pressure gradient: high→low pressure drives flow (barometric transport).
    3. Diffusion: symmetric spatial smoothing via graph Laplacian with learned
       per-feature coefficients (inspired by TransNet, Nature 2026).
    4. Combined: A_final = A_static + α*A_transport + γ*A_diffusion
    """
    def __init__(self, num_nodes, embed_dim, fixed_adj, num_features=18):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.register_buffer("fixed_adj", torch.FloatTensor(fixed_adj))

        dir_lon, dir_lat, dist = _build_direction_and_distance()
        self.register_buffer("dir_lon", torch.FloatTensor(dir_lon))
        self.register_buffer("dir_lat", torch.FloatTensor(dir_lat))
        self.register_buffer("inv_dist", torch.FloatTensor(1.0 / (dist + 1e-6)))

        # Compute normalized graph Laplacian for diffusion (from fixed adj)
        D = np.diag(fixed_adj.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-8))
        L_norm = np.eye(num_nodes) - D_inv_sqrt @ fixed_adj @ D_inv_sqrt
        self.register_buffer("L_norm", torch.FloatTensor(L_norm))

        # Learnable scaling for advection contribution
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # Learnable pressure sensitivity
        self.beta = nn.Parameter(torch.tensor(0.3))
        # Learnable per-feature diffusion coefficients (TransNet-inspired)
        self.gamma = nn.Parameter(torch.tensor(0.3))
        self.kappa = nn.Parameter(torch.ones(num_features) * 0.1)  # per-feature diffusivity

    def forward(self, wind_u=None, wind_v=None, pressure=None):
        """
        Args:
            wind_u: (batch, time, nodes) — east-west wind
            wind_v: (batch, time, nodes) — north-south wind
            pressure: (batch, time, nodes) — surface pressure
        Returns:
            (batch, time, nodes, nodes) if wind provided, else (nodes, nodes)
        """
        adp = F.softmax(F.relu(self.E1 @ self.E2.T), dim=1)
        A_static = self.fixed_adj + adp

        if wind_u is None or wind_v is None:
            return A_static

        # 1. Bidirectional advection: source pushes + destination receives
        # Source push: wind_i dot direction(i→j)
        push = (wind_u.unsqueeze(-1) * self.dir_lon.unsqueeze(0).unsqueeze(0) +
                wind_v.unsqueeze(-1) * self.dir_lat.unsqueeze(0).unsqueeze(0))
        # Destination receive: wind_j dot direction(j←i) = wind_j dot (-direction(i→j))
        recv = -(wind_u.unsqueeze(-2) * self.dir_lon.unsqueeze(0).unsqueeze(0) +
                 wind_v.unsqueeze(-2) * self.dir_lat.unsqueeze(0).unsqueeze(0))
        advection = F.relu(push + recv) * 0.5

        # 2. Pressure gradient modulation: high_i → low_j strengthens transport
        if pressure is not None:
            # pressure_diff[i,j] = pressure[i] - pressure[j]
            p_diff = pressure.unsqueeze(-1) - pressure.unsqueeze(-2)  # (B, T, N, N)
            pressure_mod = torch.sigmoid(torch.sigmoid(self.beta) * p_diff)
        else:
            pressure_mod = 1.0

        # 3. Combine: transport = advection * pressure_mod / distance
        A_transport = advection * pressure_mod * self.inv_dist.unsqueeze(0).unsqueeze(0)
        A_transport = A_transport / (A_transport.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. Diffusion: symmetric smoothing via graph Laplacian
        # kappa controls per-feature diffusivity; mean across features for edge-level
        kappa_mean = torch.sigmoid(self.kappa).mean()
        A_diffusion = kappa_mean * (torch.eye(self.L_norm.shape[0], device=self.L_norm.device) - self.L_norm)

        return (A_static.unsqueeze(0).unsqueeze(0)
                + torch.sigmoid(self.alpha) * A_transport
                + torch.sigmoid(self.gamma) * A_diffusion.unsqueeze(0).unsqueeze(0))


class GLUTemporalConv(nn.Module):
    """Gated Linear Unit temporal convolution."""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv_gate = nn.Conv2d(in_ch, out_ch, (1, kernel_size), padding=(0, kernel_size // 2))
        self.conv_filter = nn.Conv2d(in_ch, out_ch, (1, kernel_size), padding=(0, kernel_size // 2))

    def forward(self, x):
        return self.conv_filter(x) * torch.sigmoid(self.conv_gate(x))


class DynGraphConv(nn.Module):
    """Graph convolution that accepts per-timestep adjacency."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        """
        x: (batch, nodes, features)
        adj: (batch, nodes, nodes) — can be dynamic per sample
        """
        return torch.bmm(adj, x) @ self.W + self.bias


class WASTConvBlock(nn.Module):
    """ST-Conv block with dynamic per-timestep graph convolution."""
    def __init__(self, in_ch, spatial_ch, temporal_ch):
        super().__init__()
        self.temp1 = GLUTemporalConv(in_ch, temporal_ch)
        self.spatial = DynGraphConv(temporal_ch, spatial_ch)
        self.temp2 = GLUTemporalConv(spatial_ch, temporal_ch)
        self.norm = nn.LayerNorm(temporal_ch)
        self.dropout = nn.Dropout(0.2)
        self.residual = (nn.Conv2d(in_ch, temporal_ch, (1, 1))
                         if in_ch != temporal_ch else nn.Identity())

    def forward(self, x, adj_seq):
        """
        x: (batch, time, nodes, features)
        adj_seq: (batch, time, nodes, nodes) — dynamic adjacency per timestep
        """
        b, t, n, f = x.shape
        residual = self.residual(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        h = self.temp1(x.permute(0, 3, 2, 1))  # (b, tc, n, t)
        h = h.permute(0, 3, 2, 1)  # (b, t, n, tc)

        # Per-timestep graph conv
        h_spatial = []
        for i in range(h.shape[1]):
            h_spatial.append(self.spatial(h[:, i], adj_seq[:, i]))
        h = torch.stack(h_spatial, dim=1)
        h = F.relu(h)

        h = self.temp2(h.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return self.dropout(self.norm(h + residual))


class WASTGN(nn.Module):
    """Wind-Aware Spatio-Temporal Graph Network.

    Key difference from WM-STGN: the graph adjacency changes every timestep
    based on actual wind conditions, physically modeling pollution transport.
    """
    def __init__(self, num_nodes, in_channels, spatial_ch, temporal_ch,
                 time_steps, num_horizons, num_targets, fixed_adj,
                 embed_dim=8, mask_prob=0.15,
                 wind_u_idx=12, wind_v_idx=13, pressure_idx=10):
        super().__init__()
        self.wind_adj = WindAwareAdjacency(num_nodes, embed_dim, fixed_adj, in_channels)
        self.block1 = WASTConvBlock(in_channels, spatial_ch, temporal_ch)
        self.block2 = WASTConvBlock(temporal_ch, spatial_ch, temporal_ch)
        self.output_layer = nn.Linear(temporal_ch * time_steps, num_horizons * num_targets)
        self.num_horizons = num_horizons
        self.num_targets = num_targets
        self.mask_prob = mask_prob
        self.wind_u_idx = wind_u_idx
        self.wind_v_idx = wind_v_idx
        self.pressure_idx = pressure_idx

    def forward(self, x, mask_cities=False):
        """x: (batch, time, nodes, features) → (batch, horizons, nodes, targets)"""
        if mask_cities and self.training:
            b, t, n, f = x.shape
            mask = torch.ones(b, 1, n, 1, device=x.device)
            for i in range(b):
                mask[i, :, torch.randint(0, n, (1,)).item(), :] = 0.0
            x = x * mask

        # Extract wind vectors and pressure for dynamic adjacency
        wind_u = x[:, :, :, self.wind_u_idx]  # (B, T, N)
        wind_v = x[:, :, :, self.wind_v_idx]
        pressure = x[:, :, :, self.pressure_idx]

        adj_seq = self.wind_adj(wind_u, wind_v, pressure)  # (B, T, N, N)

        h = self.block1(x, adj_seq)
        h = self.block2(h, adj_seq)

        b, t, n, c = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b, n, t * c)
        out = self.output_layer(h)
        return out.reshape(b, n, self.num_horizons, self.num_targets).permute(0, 2, 1, 3)

    def predict_with_uncertainty(self, x, n_samples=20):
        """MC Dropout inference for uncertainty estimation."""
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self(x, mask_cities=False).cpu().numpy())
        self.eval()
        preds = np.stack(preds)
        return preds.mean(axis=0), preds.std(axis=0)
