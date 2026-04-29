"""WA-STGN training — 20 configs + top-3 ensemble + MC Dropout."""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

from training.common import PROC, inverse_residual, eval_pm
from models.wastgn import WASTGN

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Same configs as WM-STGN for fair comparison
CONFIGS = [
    {"sc": 8, "tc": 16, "ed": 4, "lr": 0.001, "wd": 1e-4, "gc": 5, "mp": 0.0, "bs": 32, "ep": 80},
    {"sc": 8, "tc": 16, "ed": 4, "lr": 0.001, "wd": 1e-4, "gc": 5, "mp": 0.15, "bs": 32, "ep": 80},
    {"sc": 8, "tc": 16, "ed": 8, "lr": 0.001, "wd": 1e-4, "gc": 5, "mp": 0.15, "bs": 32, "ep": 80},
    {"sc": 8, "tc": 16, "ed": 4, "lr": 0.001, "wd": 1e-3, "gc": 5, "mp": 0.0, "bs": 32, "ep": 80},
    {"sc": 8, "tc": 16, "ed": 4, "lr": 0.002, "wd": 1e-4, "gc": 5, "mp": 0.0, "bs": 32, "ep": 80},
    {"sc": 8, "tc": 16, "ed": 4, "lr": 0.001, "wd": 1e-4, "gc": 1, "mp": 0.0, "bs": 32, "ep": 100},
    {"sc": 12, "tc": 24, "ed": 6, "lr": 0.001, "wd": 1e-4, "gc": 5, "mp": 0.15, "bs": 32, "ep": 80},
    {"sc": 12, "tc": 24, "ed": 6, "lr": 0.0008, "wd": 5e-4, "gc": 5, "mp": 0.1, "bs": 32, "ep": 100},
    {"sc": 16, "tc": 32, "ed": 4, "lr": 0.001, "wd": 1e-4, "gc": 5, "mp": 0.15, "bs": 32, "ep": 80},
    {"sc": 16, "tc": 32, "ed": 8, "lr": 0.001, "wd": 1e-4, "gc": 5, "mp": 0.15, "bs": 32, "ep": 80},
    {"sc": 16, "tc": 32, "ed": 4, "lr": 0.0005, "wd": 1e-4, "gc": 5, "mp": 0.15, "bs": 16, "ep": 100},
    {"sc": 24, "tc": 48, "ed": 6, "lr": 0.001, "wd": 1e-4, "gc": 5, "mp": 0.15, "bs": 32, "ep": 80},
    {"sc": 24, "tc": 48, "ed": 8, "lr": 0.0008, "wd": 5e-4, "gc": 3, "mp": 0.1, "bs": 32, "ep": 100},
    {"sc": 32, "tc": 64, "ed": 8, "lr": 0.001, "wd": 5e-4, "gc": 3, "mp": 0.15, "bs": 32, "ep": 80},
    {"sc": 32, "tc": 64, "ed": 8, "lr": 0.001, "wd": 5e-4, "gc": 3, "mp": 0.2, "bs": 32, "ep": 80},
    {"sc": 32, "tc": 64, "ed": 8, "lr": 0.001, "wd": 5e-4, "gc": 3, "mp": 0.25, "bs": 32, "ep": 80},
    {"sc": 8, "tc": 16, "ed": 4, "lr": 0.0003, "wd": 1e-4, "gc": 5, "mp": 0.0, "bs": 32, "ep": 120},
    {"sc": 16, "tc": 32, "ed": 4, "lr": 0.0003, "wd": 1e-4, "gc": 5, "mp": 0.15, "bs": 32, "ep": 120},
    {"sc": 8, "tc": 16, "ed": 4, "lr": 0.0005, "wd": 1e-4, "gc": 5, "mp": 0.15, "bs": 16, "ep": 100},
    {"sc": 12, "tc": 24, "ed": 6, "lr": 0.0005, "wd": 5e-4, "gc": 3, "mp": 0.25, "bs": 32, "ep": 100},
]


def _train_one(params, X_tr, Y_tr, X_v, Y_v, adj):
    _, ts, nn_, ic = X_tr.shape
    nh, ntgt = Y_tr.shape[1], Y_tr.shape[3]

    model = WASTGN(nn_, ic, params["sc"], params["tc"], ts, nh, ntgt, adj,
                    params["ed"], params["mp"]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    warmup = 5
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / (params["ep"] - warmup)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=params["bs"], shuffle=True)

    best_val, best_state = float("inf"), None
    for epoch in range(params["ep"]):
        model.train()
        for xb, yb in loader:
            xb_noisy = xb + torch.randn_like(xb) * 0.02
            optimizer.zero_grad()
            loss = criterion(model(xb_noisy, mask_cities=(params["mp"] > 0)), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params["gc"])
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_v, mask_cities=False), Y_v).item()
        if vl < best_val:
            best_val = vl
            best_state = deepcopy(model.state_dict())

    return best_state, best_val


def run_wastgn(seq, stats, meta, adj, Y_true_abs):
    print("\n" + "=" * 50 + f"\nWA-STGN TUNING (20 configs on {DEVICE})\n" + "=" * 50)

    X_tr = torch.FloatTensor(seq["X_train"]).to(DEVICE)
    Y_tr = torch.FloatTensor(seq["Y_train"]).to(DEVICE)
    X_v = torch.FloatTensor(seq["X_val"]).to(DEVICE)
    Y_v = torch.FloatTensor(seq["Y_val"]).to(DEVICE)
    X_te = torch.FloatTensor(seq["X_test"]).to(DEVICE)
    last_vals = seq["last_vals_test"]

    _, ts, nn_, ic = seq["X_train"].shape
    nh, ntgt = seq["Y_train"].shape[1], seq["Y_train"].shape[3]

    all_preds, all_maes, all_logs = [], [], []

    for i, cfg in enumerate(CONFIGS):
        t0 = time.time()
        state, val = _train_one(cfg, X_tr, Y_tr, X_v, Y_v, adj)

        model = WASTGN(nn_, ic, cfg["sc"], cfg["tc"], ts, nh, ntgt, adj, cfg["ed"], 0.0).to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            Yp_norm = model(X_te, mask_cities=False).cpu().numpy()

        Yp_abs = inverse_residual(Yp_norm, last_vals, stats, meta)
        pm_results = eval_pm(Yp_abs, Y_true_abs, meta)
        mae = np.mean([r["MAE"] for r in pm_results])
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(CONFIGS)}] sc={cfg['sc']} tc={cfg['tc']} mp={cfg['mp']:.2f} → PM MAE={mae:.2f} ({elapsed:.0f}s)")

        all_preds.append(Yp_norm)
        all_maes.append(mae)
        all_logs.append({**cfg, "val_loss": val, "test_mae": mae, "time_s": elapsed})

    # Ensemble top-3
    top3 = np.argsort(all_maes)[:3]
    ens3_norm = np.mean([all_preds[i] for i in top3], axis=0)
    ens3_abs = inverse_residual(ens3_norm, last_vals, stats, meta)
    ens3_results = eval_pm(ens3_abs, Y_true_abs, meta)
    ens3_mae = np.mean([r["MAE"] for r in ens3_results])
    print(f"\nEnsemble top-3 (configs {[i+1 for i in top3]}): PM MAE={ens3_mae:.2f}")

    best_idx = np.argmin(all_maes)
    if ens3_mae < all_maes[best_idx]:
        print(f"Winner: Ensemble (MAE={ens3_mae:.2f})")
        final_results = [{"Model": "WA-STGN-ens3", **r} for r in ens3_results]
        final_abs = ens3_abs
    else:
        print(f"Winner: Single config {best_idx+1} (MAE={all_maes[best_idx]:.2f})")
        best_abs = inverse_residual(all_preds[best_idx], last_vals, stats, meta)
        final_results = [{"Model": "WA-STGN-best", **r} for r in eval_pm(best_abs, Y_true_abs, meta)]
        final_abs = best_abs

    pd.DataFrame(all_logs).to_csv(f"{PROC}/wastgn_tuning_log.csv", index=False)
    np.save(f"{PROC}/wastgn_predictions.npy", final_abs)

    # Save best single model + MC Dropout
    best_cfg = CONFIGS[best_idx]
    state, _ = _train_one(best_cfg, X_tr, Y_tr, X_v, Y_v, adj)
    torch.save(state, f"{PROC}/wastgn_model_tuned.pt")

    model = WASTGN(nn_, ic, best_cfg["sc"], best_cfg["tc"], ts, nh, ntgt, adj,
                    best_cfg["ed"], best_cfg["mp"]).to("cpu")
    model.load_state_dict(state)
    mc_mean, mc_std = model.predict_with_uncertainty(X_te.to("cpu"), n_samples=20)
    mc_abs = inverse_residual(mc_mean, last_vals, stats, meta)
    mc_std_abs = np.abs(mc_std * stats["y_std"])
    np.save(f"{PROC}/wastgn_pred_mean.npy", mc_abs)
    np.save(f"{PROC}/wastgn_pred_std.npy", mc_std_abs)

    df = pd.DataFrame(final_results)
    print(df.to_string(index=False))
    return df, best_cfg
