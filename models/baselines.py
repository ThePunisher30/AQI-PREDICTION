"""Baselines — Persistence and Historical Average (residual prediction)."""

import numpy as np
import pandas as pd
from training.common import PROC, inverse_residual, eval_pm


def run_baselines(seq, stats, meta):
    print("\n" + "=" * 50 + "\nBASELINES\n" + "=" * 50)
    last_vals = seq["last_vals_test"]
    Y_test_delta = seq["Y_test"]

    Y_true_abs = inverse_residual(Y_test_delta, last_vals, stats, meta)

    # Persistence: delta = 0
    persist_abs = inverse_residual(np.zeros_like(Y_test_delta), last_vals, stats, meta)

    # Historical average delta
    Y_train_delta_real = seq["Y_train"] * stats["y_std"] + stats["y_mean"]
    avg_delta = Y_train_delta_real.mean(axis=0, keepdims=True)
    avg_delta_norm = (avg_delta - stats["y_mean"]) / stats["y_std"]
    hist_delta = np.broadcast_to(avg_delta_norm, Y_test_delta.shape).copy()
    hist_abs = inverse_residual(hist_delta, last_vals, stats, meta)

    r1 = [{"Model": "Persistence", **r} for r in eval_pm(persist_abs, Y_true_abs, meta)]
    r2 = [{"Model": "HistoricalAvg", **r} for r in eval_pm(hist_abs, Y_true_abs, meta)]

    df = pd.DataFrame(r1 + r2)
    df.to_csv(f"{PROC}/baseline_results.csv", index=False)
    print(df.to_string(index=False))
    return df, Y_true_abs
