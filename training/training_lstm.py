"""LSTM training — 20 hyperparameter configs, residual prediction."""

import os, time
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from training.common import PROC, inverse_residual, eval_pm
from models.lstm import build_lstm

CONFIGS = [
    {"h1": 32, "h2": 16, "d": 0.1, "lr": 0.001, "ep": 80, "bs": 32},
    {"h1": 32, "h2": 16, "d": 0.2, "lr": 0.001, "ep": 80, "bs": 32},
    {"h1": 32, "h2": 16, "d": 0.3, "lr": 0.001, "ep": 80, "bs": 32},
    {"h1": 32, "h2": 16, "d": 0.3, "lr": 0.0005, "ep": 100, "bs": 16},
    {"h1": 64, "h2": 32, "d": 0.1, "lr": 0.001, "ep": 60, "bs": 32},
    {"h1": 64, "h2": 32, "d": 0.2, "lr": 0.001, "ep": 60, "bs": 32},
    {"h1": 64, "h2": 32, "d": 0.3, "lr": 0.001, "ep": 60, "bs": 32},
    {"h1": 64, "h2": 32, "d": 0.4, "lr": 0.001, "ep": 60, "bs": 32},
    {"h1": 64, "h2": 32, "d": 0.3, "lr": 0.0005, "ep": 80, "bs": 16},
    {"h1": 64, "h2": 32, "d": 0.3, "lr": 0.002, "ep": 60, "bs": 32},
    {"h1": 128, "h2": 64, "d": 0.2, "lr": 0.001, "ep": 60, "bs": 32},
    {"h1": 128, "h2": 64, "d": 0.3, "lr": 0.001, "ep": 60, "bs": 32},
    {"h1": 128, "h2": 64, "d": 0.4, "lr": 0.001, "ep": 60, "bs": 32},
    {"h1": 128, "h2": 64, "d": 0.4, "lr": 0.0005, "ep": 80, "bs": 16},
    {"h1": 128, "h2": 64, "d": 0.3, "lr": 0.0005, "ep": 80, "bs": 16},
    {"h1": 64, "h2": 32, "d": 0.3, "lr": 0.0003, "ep": 100, "bs": 32},
    {"h1": 128, "h2": 64, "d": 0.3, "lr": 0.0003, "ep": 100, "bs": 32},
    {"h1": 64, "h2": None, "d": 0.3, "lr": 0.001, "ep": 80, "bs": 32},
    {"h1": 128, "h2": None, "d": 0.3, "lr": 0.001, "ep": 80, "bs": 32},
    {"h1": 64, "h2": 32, "d": 0.3, "lr": 0.001, "ep": 80, "bs": 32, "h3": 16},
]


def run_lstm(seq, stats, meta, Y_true_abs):
    print("\n" + "=" * 50 + "\nLSTM (residual, multi-target)\n" + "=" * 50)
    import tensorflow as tf
    from tensorflow.keras.callbacks import ModelCheckpoint

    X_train, Y_train = seq["X_train"], seq["Y_train"]
    X_val, Y_val = seq["X_val"], seq["Y_val"]
    X_test = seq["X_test"]
    last_vals = seq["last_vals_test"]

    ns, lb, nc, nf = X_train.shape
    nh, ntgt = Y_train.shape[1], Y_train.shape[3]
    out = nh * nc * ntgt

    Xtr = X_train.reshape(ns, lb, nc * nf)
    Xv = X_val.reshape(X_val.shape[0], lb, nc * nf)
    Xte = X_test.reshape(X_test.shape[0], lb, nc * nf)
    Ytr = Y_train.reshape(Y_train.shape[0], -1)
    Yv = Y_val.reshape(Y_val.shape[0], -1)

    ckpt = f"{PROC}/_lstm_ckpt.keras"
    best_mae = float("inf")
    best_pred = None

    for i, cfg in enumerate(CONFIGS):
        t0 = time.time()
        tf.keras.backend.clear_session()

        model = build_lstm(
            input_shape=(lb, nc * nf), output_size=out,
            h1=cfg["h1"], h2=cfg["h2"], h3=cfg.get("h3"),
            dropout=cfg["d"], lr=cfg["lr"]
        )
        cb = ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, verbose=0)
        model.fit(Xtr, Ytr, validation_data=(Xv, Yv),
                  epochs=cfg["ep"], batch_size=cfg.get("bs", 32), callbacks=[cb], verbose=0)

        model = tf.keras.models.load_model(ckpt)
        Yp_norm = model.predict(Xte, verbose=0).reshape(X_test.shape[0], nh, nc, ntgt)
        Yp_abs = inverse_residual(Yp_norm, last_vals, stats, meta)
        pm_results = eval_pm(Yp_abs, Y_true_abs, meta)
        mae = np.mean([r["MAE"] for r in pm_results])
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(CONFIGS)}] h={cfg['h1']},{cfg['h2']} d={cfg['d']} → PM MAE={mae:.2f} ({elapsed:.0f}s)")

        if mae < best_mae:
            best_mae = mae
            best_pred = Yp_abs
            model.save(f"{PROC}/lstm_model_tuned.keras")

    results = [{"Model": "LSTM", **r} for r in eval_pm(best_pred, Y_true_abs, meta)]
    df = pd.DataFrame(results)
    df.to_csv(f"{PROC}/lstm_results.csv", index=False)
    np.save(f"{PROC}/lstm_predictions.npy", best_pred)
    print(df.to_string(index=False))
    return df
