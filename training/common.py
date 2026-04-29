"""Shared utilities for all models — data loading, inverse transform, evaluation."""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import evaluate_all

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
GRAPH = os.path.join(os.path.dirname(__file__), "..", "data", "graphs")


def load():
    seq = dict(np.load(f"{PROC}/sequences.npz"))
    stats = dict(np.load(f"{PROC}/norm_stats.npz"))
    with open(f"{PROC}/meta.json") as f:
        meta = json.load(f)
    adj = np.load(f"{GRAPH}/adj_matrix.npy")
    return seq, stats, meta, adj


def inverse_residual(Y_delta_norm, last_vals, stats, meta):
    """Convert normalized deltas back to absolute values in original scale."""
    delta_real = Y_delta_norm * stats["y_std"] + stats["y_mean"]
    pred_log = last_vals[:, np.newaxis, :, :] + delta_real
    pred_original = np.expm1(np.clip(pred_log, -10, 20))
    return np.clip(pred_original, 0, None)


def eval_pm(Y_pred_abs, Y_true_abs, meta):
    """Evaluate only PM2.5 and PM10 (for AQI comparison)."""
    pm_idx = meta["pm_eval_idx"]
    results = []
    for h, hz in enumerate(meta["horizons"]):
        for pi, tgt in zip(pm_idx, ["pm2_5", "pm10"]):
            yt = Y_true_abs[:, h, :, pi].flatten()
            yp = Y_pred_abs[:, h, :, pi].flatten()
            m = evaluate_all(yt, yp)
            results.append({"Horizon": f"{hz}h", "Target": tgt, **m})
    return results
