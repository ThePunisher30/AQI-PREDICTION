"""Full retraining — orchestrates all models, optimized for M4 Pro MPS."""

import os, sys, time
import pandas as pd
import torch

torch.set_num_threads(12)
torch.set_num_interop_threads(14)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.common import PROC, load
from models.baselines import run_baselines
from training.training_lstm import run_lstm
from training.training_cnn_lstm import run_cnn_lstm
from training.training_wmstgn import run_wmstgn
from training.training_wastgn import run_wastgn

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    t_start = time.time()

    seq, stats, meta, adj = load()
    print(f"Data: {seq['X_train'].shape[0]} train, {seq['X_test'].shape[0]} test, "
          f"{seq['X_train'].shape[2]} cities, {seq['X_train'].shape[3]} features, "
          f"{seq['Y_train'].shape[3]} targets")
    print(f"Mode: RESIDUAL prediction, log-transformed")

    baseline_df, Y_true_abs = run_baselines(seq, stats, meta)
    lstm_df = run_lstm(seq, stats, meta, Y_true_abs)
    cnnlstm_df = run_cnn_lstm(seq, stats, meta, Y_true_abs)
    wmstgn_df, best_cfg = run_wmstgn(seq, stats, meta, adj, Y_true_abs)
    wastgn_df, best_wa_cfg = run_wastgn(seq, stats, meta, adj, Y_true_abs)

    all_df = pd.concat([baseline_df, lstm_df, cnnlstm_df, wmstgn_df, wastgn_df], ignore_index=True).round(3)
    all_df.to_csv(f"{PROC}/all_results.csv", index=False)

    print("\n" + "=" * 50 + "\nFINAL RESULTS\n" + "=" * 50)
    print(all_df.to_string(index=False))

    elapsed = time.time() - t_start
    print(f"\n✅ Complete in {elapsed/60:.1f} min. Best WM-STGN: {best_cfg} | Best WA-STGN: {best_wa_cfg}")
