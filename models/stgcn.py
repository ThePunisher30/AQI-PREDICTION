"""WM-STGN — Masked Spatio-Temporal Graph Network.

- Adaptive adjacency matrix via learnable node embeddings [24] Graph WaveNet
- GLU activation in temporal convolutions [7] STGCN
- Residual connections in ST-Conv blocks [4] ST-ResNet
- Random city masking during training (novel)
- MC Dropout at inference for uncertainty estimation (novel)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAdjacency(nn.Module):
    """Graph WaveNet-style adaptive adjacency: A_adp = softmax(ReLU(E1 @ E2^T)).
    Combined with fixed distance-based adjacency: A_final = A_fixed + A_adp.
    Reference: [24] Wu et al., Graph WaveNet, IJCAI 2019.
    """
    def __init__(self, num_nodes, embed_dim, fixed_adj):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.register_buffer("fixed_adj", torch.FloatTensor(fixed_adj))

    def forward(self):
        adp = F.softmax(F.relu(self.E1 @ self.E2.T), dim=1)
        return self.fixed_adj + adp


class GLUTemporalConv(nn.Module):
    """Gated Linear Unit temporal convolution.
    Reference: [7] Yu et al., STGCN, IJCAI 2018.
    Γ = (W1*X + b1) ⊙ σ(W2*X + b2)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_gate = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size // 2))
        self.conv_filter = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size // 2))

    def forward(self, x):
        """x: (batch, channels, nodes, time)"""
        return self.conv_filter(x) * torch.sigmoid(self.conv_gate(x))


class GraphConv(nn.Module):
    """Simple 1-hop graph convolution: H = A @ X @ W."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        """x: (batch, nodes, features), adj: (nodes, nodes)"""
        return torch.einsum("nm,bmc->bnc", adj, x) @ self.W + self.bias


class STConvBlock(nn.Module):
    """Spatio-Temporal Conv Block with GLU + graph conv + residual connection.
    References: [7] STGCN, [4] ST-ResNet, [24] Graph WaveNet.
    """
    def __init__(self, in_channels, spatial_channels, temporal_channels):
        super().__init__()
        self.temp1 = GLUTemporalConv(in_channels, temporal_channels)
        self.spatial = GraphConv(temporal_channels, spatial_channels)
        self.temp2 = GLUTemporalConv(spatial_channels, temporal_channels)
        self.norm = nn.LayerNorm(temporal_channels)
        self.dropout = nn.Dropout(0.2)
        # Residual projection if dimensions differ [4]
        self.residual = nn.Conv2d(in_channels, temporal_channels, (1, 1)) if in_channels != temporal_channels else nn.Identity()

    def forward(self, x, adj):
        """x: (batch, time, nodes, features)"""
        b, t, n, f = x.shape
        residual = x.permute(0, 3, 2, 1)  # (b, f, n, t)
        residual = self.residual(residual).permute(0, 3, 2, 1)  # (b, t, n, temp_ch)

        # Temporal conv 1
        h = x.permute(0, 3, 2, 1)  # (b, f, n, t)
        h = self.temp1(h)  # (b, temp_ch, n, t)

        # Spatial conv per timestep
        temp_ch = h.shape[1]
        h = h.permute(0, 3, 2, 1)  # (b, t, n, temp_ch)
        h_spatial = []
        for i in range(h.shape[1]):
            h_spatial.append(self.spatial(h[:, i], adj))
        h = torch.stack(h_spatial, dim=1)  # (b, t, n, spatial_ch)
        h = F.relu(h)

        # Temporal conv 2
        h = h.permute(0, 3, 2, 1)  # (b, sp_ch, n, t)
        h = self.temp2(h)  # (b, temp_ch, n, t)
        h = h.permute(0, 3, 2, 1)  # (b, t, n, temp_ch)

        # Residual connection + norm + dropout
        return self.dropout(self.norm(h + residual))


class WMSTGN(nn.Module):
    """Wind-aware Masked Spatio-Temporal Graph Network.

    Novel contributions:
    1. Adaptive adjacency (Graph WaveNet style) + fixed distance adjacency
    2. GLU temporal convolutions (from original STGCN paper)
    3. Residual connections (from ST-ResNet)
    4. Random city masking during training (novel - forces spatial generalization)
    5. MC Dropout at inference (novel - uncertainty estimation)
    """
    def __init__(self, num_nodes, in_channels, spatial_ch, temporal_ch,
                 time_steps, num_horizons, num_targets, fixed_adj,
                 embed_dim=8, mask_prob=0.15):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, embed_dim, fixed_adj)
        self.block1 = STConvBlock(in_channels, spatial_ch, temporal_ch)
        self.block2 = STConvBlock(temporal_ch, spatial_ch, temporal_ch)
        self.output_layer = nn.Linear(temporal_ch * time_steps, num_horizons * num_targets)
        self.num_horizons = num_horizons
        self.num_targets = num_targets
        self.mask_prob = mask_prob

    def forward(self, x, mask_cities=False):
        """x: (batch, time, nodes, features) → (batch, horizons, nodes, targets)"""
        # Random city masking during training (novel contribution)
        if mask_cities and self.training:
            b, t, n, f = x.shape
            mask = torch.ones(b, 1, n, 1, device=x.device)
            for i in range(b):
                # Mask 1 random city per sample
                city_idx = torch.randint(0, n, (1,)).item()
                mask[i, :, city_idx, :] = 0.0
            x = x * mask

        adj = self.adaptive_adj()
        h = self.block1(x, adj)
        h = self.block2(h, adj)

        b, t, n, c = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b, n, t * c)
        out = self.output_layer(h)
        return out.reshape(b, n, self.num_horizons, self.num_targets).permute(0, 2, 1, 3)

    def predict_with_uncertainty(self, x, n_samples=20):
        """MC Dropout inference for uncertainty estimation (novel contribution).
        Run model n_samples times with dropout enabled → mean ± std.
        """
        self.train()  # Keep dropout active
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self(x, mask_cities=False).cpu().numpy())
        self.eval()
        preds = np.stack(preds)
        return preds.mean(axis=0), preds.std(axis=0)

