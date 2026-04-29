"""Generate all publication-quality figures for the paper."""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
GRAPH = os.path.join(os.path.dirname(__file__), "..", "data", "graphs")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

import yaml
with open(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")) as f:
    CFG = yaml.safe_load(f)
with open(os.path.join(PROC, "meta.json")) as f:
    META = json.load(f)

RESULTS = pd.read_csv(os.path.join(PROC, "all_results.csv"))
CITIES = META["city_names"]
LOCS = CFG["locations"]

plt.rcParams.update({"font.size": 11, "figure.dpi": 200, "savefig.bbox": "tight",
                      "axes.grid": True, "grid.alpha": 0.3})
COLORS = {"Persistence": "#95a5a6", "HistoricalAvg": "#bdc3c7", "LSTM": "#3498db",
           "CNN-LSTM": "#2ecc71", "WM-STGN-ens3": "#e74c3c", "WA-STGN-best": "#9b59b6",
           "WM-STGN-best": "#e74c3c"}


def fig1_mae_comparison():
    """Bar chart: MAE across models and horizons for PM2.5."""
    df = RESULTS[RESULTS["Target"] == "pm2_5"]
    models = df["Model"].unique()
    horizons = ["1h", "6h", "12h", "24h"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(horizons))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [df[(df["Model"] == m) & (df["Horizon"] == h)]["MAE"].values[0] for h in horizons]
        ax.bar(x + i * w, vals, w, label=m, color=COLORS.get(m, "#777"), alpha=0.85)
    ax.set_xticks(x + w * len(models) / 2)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("PM2.5 Prediction Error by Model and Forecast Horizon")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(f"{FIG_DIR}/fig1_mae_comparison.png")
    plt.close()
    print("✅ fig1_mae_comparison")


def fig2_r2_comparison():
    """Bar chart: R² across models and horizons for PM2.5."""
    df = RESULTS[RESULTS["Target"] == "pm2_5"]
    models = df["Model"].unique()
    horizons = ["1h", "6h", "12h", "24h"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(horizons))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [df[(df["Model"] == m) & (df["Horizon"] == h)]["R2"].values[0] for h in horizons]
        ax.bar(x + i * w, vals, w, label=m, color=COLORS.get(m, "#777"), alpha=0.85)
    ax.set_xticks(x + w * len(models) / 2)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("R²")
    ax.set_title("PM2.5 R² Score by Model and Forecast Horizon")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0.6, 1.0)
    fig.savefig(f"{FIG_DIR}/fig2_r2_comparison.png")
    plt.close()
    print("✅ fig2_r2_comparison")


def fig3_city_graph_map():
    """29 cities on India map with graph edges."""
    adj = np.load(os.path.join(GRAPH, "adj_matrix.npy"))
    fig, ax = plt.subplots(figsize=(8, 10))
    # Plot edges
    for i in range(len(CITIES)):
        for j in range(i + 1, len(CITIES)):
            if adj[i, j] > 0.01:
                ci, cj = CITIES[i], CITIES[j]
                if ci in LOCS and cj in LOCS:
                    ax.plot([LOCS[ci][1], LOCS[cj][1]], [LOCS[ci][0], LOCS[cj][0]],
                            "b-", alpha=min(adj[i, j] * 5, 0.4), linewidth=0.5)
    # Plot cities
    for city in CITIES:
        if city in LOCS:
            lat, lon = LOCS[city]
            ax.scatter(lon, lat, s=60, c="red", zorder=5, edgecolors="black", linewidth=0.5)
            ax.annotate(city, (lon, lat), fontsize=6, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"City Graph: {len(CITIES)} Cities, {(adj > 0.01).sum() - len(CITIES)} Edges")
    fig.savefig(f"{FIG_DIR}/fig3_city_graph.png")
    plt.close()
    print("✅ fig3_city_graph_map")


def fig4_feature_importance():
    """Feature and city importance from permutation analysis."""
    feat = pd.read_csv(os.path.join(PROC, "shap_feature_importance.csv"))
    city = pd.read_csv(os.path.join(PROC, "shap_city_importance.csv"))
    col_f = [c for c in feat.columns if c != feat.columns[0]][0]
    col_c = [c for c in city.columns if c != city.columns[0]][0]
    feat = feat.sort_values(col_f, ascending=True)
    city = city.sort_values(col_c, ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.barh(feat.iloc[:, 0], feat[col_f], color=["#e74c3c" if v > 0 else "#3498db" for v in feat[col_f]])
    ax1.set_xlabel("ΔMAE (higher = more important)")
    ax1.set_title("Feature Importance (Permutation)")
    ax1.axvline(x=0, color="black", linewidth=0.5)

    ax2.barh(city.iloc[:, 0], city[col_c], color=["#e74c3c" if v > 0.5 else "#3498db" for v in city[col_c]])
    ax2.set_xlabel("ΔMAE (higher = more critical)")
    ax2.set_title("City Importance (Ablation)")
    ax2.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig4_feature_importance.png")
    plt.close()
    print("✅ fig4_feature_importance")


def fig5_per_city_forecast():
    """Actual vs predicted for Delhi (polluted) and Shillong (clean) at 24h, with dates."""
    from training.common import inverse_residual
    seq = dict(np.load(os.path.join(PROC, "sequences.npz")))
    stats = dict(np.load(os.path.join(PROC, "norm_stats.npz")))
    last_vals = seq["last_vals_test"]

    Y_true = inverse_residual(seq["Y_test"], last_vals, stats, META)
    Y_pred = np.load(os.path.join(PROC, "wmstgn_predictions.npy"))

    # Get actual dates from combined_data
    combined = pd.read_csv(os.path.join(PROC, "combined_data.csv"))
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    ts_all = sorted(combined[combined["city"] == "Delhi"]["timestamp"].unique())
    n_total = len(ts_all)
    n_test = len(seq["Y_test"])
    test_dates = ts_all[-(n_test + 24):-24]  # approximate test dates

    pm25_idx = META["pm_eval_idx"][0]
    h24_idx = META["horizons"].index(24)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    n = min(100, len(Y_true), len(test_dates))
    dates = test_dates[-n:]

    for ax, city_name in zip(axes, ["Delhi", "Shillong"]):
        ci = CITIES.index(city_name)
        true = Y_true[-n:, h24_idx, ci, pm25_idx]
        pred = Y_pred[-n:, h24_idx, ci, pm25_idx]
        ax.plot(dates[:len(true)], true, "k-", label="Actual (CPCB)", linewidth=1)
        ax.plot(dates[:len(pred)], pred, "r--", label="WM-STGN Predicted", linewidth=1, alpha=0.8)
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title(f"{city_name} — 24h Ahead PM2.5 Forecast")
        ax.legend()
        ax.tick_params(axis="x", rotation=30)
    axes[1].set_xlabel("Date")
    fig.suptitle(f"Test Period: {dates[0].strftime('%b %d')} – {dates[-1].strftime('%b %d, %Y')}", fontsize=10, y=0.02)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig5_per_city_forecast.png")
    plt.close()
    print("✅ fig5_per_city_forecast")


def fig6_residual_impact():
    """Before vs after residual prediction — the biggest improvement."""
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = ["1h", "6h", "12h", "24h"]
    before = [15.71, 18.66, 22.65, 30.32]  # from steering doc (pre-residual WM-STGN)
    after_wm = RESULTS[(RESULTS["Target"] == "pm2_5") &
                        (RESULTS["Model"].str.contains("WM-STGN"))].sort_values("Horizon")["MAE"].values
    x = np.arange(len(horizons))
    ax.bar(x - 0.2, before, 0.35, label="Before (absolute prediction)", color="#e74c3c", alpha=0.7)
    ax.bar(x + 0.2, after_wm, 0.35, label="After (residual prediction)", color="#2ecc71", alpha=0.7)
    for i in range(len(horizons)):
        pct = (before[i] - after_wm[i]) / before[i] * 100
        ax.annotate(f"-{pct:.0f}%", (x[i], max(before[i], after_wm[i]) + 0.5),
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("Impact of Residual Prediction on PM2.5 MAE")
    ax.legend()
    fig.savefig(f"{FIG_DIR}/fig6_residual_impact.png")
    plt.close()
    print("✅ fig6_residual_impact")


def fig7_wind_adjacency_heatmap():
    """Show how wind-aware adjacency changes vs static adjacency."""
    import torch
    from models.wastgn import WASTGN
    adj = np.load(os.path.join(GRAPH, "adj_matrix.npy"))
    seq = dict(np.load(os.path.join(PROC, "sequences.npz")))

    model = WASTGN(29, 18, 8, 16, 24, 4, 7, adj, embed_dim=4)
    x = torch.FloatTensor(seq["X_test"][:1])  # single sample

    # Get static adj
    A_static = model.wind_adj.fixed_adj.numpy()
    # Get dynamic adj for one timestep
    wind_u = x[:, 0:1, :, 12]
    wind_v = x[:, 0:1, :, 13]
    pressure = x[:, 0:1, :, 10]
    with torch.no_grad():
        A_dyn = model.wind_adj(wind_u, wind_v, pressure)[0, 0].numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    im1 = ax1.imshow(A_static, cmap="YlOrRd", aspect="auto")
    ax1.set_title("Static Adjacency\n(distance-based)")
    plt.colorbar(im1, ax=ax1, shrink=0.7)

    im2 = ax2.imshow(A_dyn, cmap="YlOrRd", aspect="auto")
    ax2.set_title("Wind-Aware Adjacency\n(dynamic, one timestep)")
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    diff = A_dyn - A_static
    im3 = ax3.imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-diff.max(), vmax=diff.max())
    ax3.set_title("Difference\n(wind effect on edges)")
    plt.colorbar(im3, ax=ax3, shrink=0.7)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Destination City")
        ax.set_ylabel("Source City")
        step = max(1, len(CITIES) // 10)
        ax.set_xticks(range(0, len(CITIES), step))
        ax.set_xticklabels([CITIES[i][:4] for i in range(0, len(CITIES), step)], rotation=45, fontsize=7)
        ax.set_yticks(range(0, len(CITIES), step))
        ax.set_yticklabels([CITIES[i][:4] for i in range(0, len(CITIES), step)], fontsize=7)

    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig7_wind_adjacency.png")
    plt.close()
    print("✅ fig7_wind_adjacency_heatmap")


def fig8_uncertainty():
    """MC Dropout confidence intervals for Delhi, with dates."""
    pred_mean = np.load(os.path.join(PROC, "wmstgn_pred_mean.npy"))
    pred_std = np.load(os.path.join(PROC, "wmstgn_pred_std.npy"))

    from training.common import inverse_residual
    seq = dict(np.load(os.path.join(PROC, "sequences.npz")))
    stats = dict(np.load(os.path.join(PROC, "norm_stats.npz")))
    Y_true = inverse_residual(seq["Y_test"], seq["last_vals_test"], stats, META)

    combined = pd.read_csv(os.path.join(PROC, "combined_data.csv"))
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    ts_all = sorted(combined[combined["city"] == "Delhi"]["timestamp"].unique())
    n_test = len(seq["Y_test"])
    test_dates = ts_all[-(n_test + 24):-24]

    pm25_idx = META["pm_eval_idx"][0]
    h24_idx = META["horizons"].index(24)
    ci = CITIES.index("Delhi")

    true = Y_true[:, h24_idx, ci, pm25_idx]
    mean = pred_mean[:, h24_idx, ci, pm25_idx]
    std = pred_std[:, h24_idx, ci, pm25_idx]
    n = min(80, len(true), len(test_dates))
    dates = test_dates[-n:]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates[:len(true[-n:])], true[-n:], "k-", label="Actual (CPCB)", linewidth=1.2)
    ax.plot(dates[:len(mean[-n:])], mean[-n:], "r-", label="Predicted (MC mean)", linewidth=1)
    ax.fill_between(dates[:n], (mean - 1.96 * std)[-n:], (mean + 1.96 * std)[-n:],
                     alpha=0.2, color="red", label="95% CI")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"Delhi — 24h PM2.5 Forecast with MC Dropout Uncertainty ({dates[0].strftime('%b %d')} – {dates[-1].strftime('%b %d, %Y')})")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig8_uncertainty.png")
    plt.close()
    print("✅ fig8_uncertainty")


def fig9_horizon_degradation():
    """Line chart showing how each model degrades with horizon."""
    df = RESULTS[RESULTS["Target"] == "pm2_5"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    horizons = [1, 6, 12, 24]
    for m in df["Model"].unique():
        md = df[df["Model"] == m]
        maes = [md[md["Horizon"] == f"{h}h"]["MAE"].values[0] for h in horizons]
        r2s = [md[md["Horizon"] == f"{h}h"]["R2"].values[0] for h in horizons]
        c = COLORS.get(m, "#777")
        ax1.plot(horizons, maes, "o-", label=m, color=c, linewidth=2, markersize=6)
        ax2.plot(horizons, r2s, "o-", label=m, color=c, linewidth=2, markersize=6)
    ax1.set_xlabel("Forecast Horizon (hours)")
    ax1.set_ylabel("MAE (µg/m³)")
    ax1.set_title("PM2.5 MAE vs Forecast Horizon")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Forecast Horizon (hours)")
    ax2.set_ylabel("R²")
    ax2.set_title("PM2.5 R² vs Forecast Horizon")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig9_horizon_degradation.png")
    plt.close()
    print("✅ fig9_horizon_degradation")


if __name__ == "__main__":
    fig1_mae_comparison()
    fig2_r2_comparison()
    fig3_city_graph_map()
    fig4_feature_importance()
    fig5_per_city_forecast()
    fig6_residual_impact()
    fig7_wind_adjacency_heatmap()
    fig8_uncertainty()
    fig9_horizon_degradation()
    print(f"\n✅ All figures saved to {FIG_DIR}/")
