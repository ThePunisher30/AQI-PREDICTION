"""Data Loader v2 — CPCB (real station data) + Open-Meteo (weather). No API keys needed."""

import os, sys, time, json, io, gzip
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

LOCATIONS = CFG["locations"]
LOOKBACK = CFG["data"]["lookback_hours"]
HORIZONS = CFG["data"]["forecast_horizons"]
TRAIN_R = CFG["data"]["train_ratio"]
VAL_R = CFG["data"]["val_ratio"]
TARGET_POLLUTANTS = CFG["target_pollutants"]

POLLUTANT_COLS = ["PM2.5_avg", "PM10_avg", "CO_avg", "NO2_avg", "SO2_avg", "OZONE_avg", "NH3_avg"]
POLLUTANT_NAMES = ["pm2_5", "pm10", "co", "no2", "so2", "o3", "nh3"]
WEATHER_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation"

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", CFG["paths"]["raw_data"])
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", CFG["paths"]["processed_data"])

# CPCB station mapping (one representative station per city from Vayuayan archive)
CPCB_STATIONS = {
    # Clean air
    "Shillong": "meghalaya__shillong__lumpyngngad_shillong_meghalaya_pcb__site_5131.csv.gz",
    "Kohima": "nagaland__kohima__pwd_junction_kohima_npcb__site_5375.csv.gz",
    "Mysuru": "karnataka__mysuru__hebbal_1st_stage_mysuru_kspcb__site_5125.csv.gz",
    "Gangtok": "sikkim__gangtok__zero_point_gici_gangtok_sspcb__site_5590.csv.gz",
    # Moderate
    "Raipur": "chhattisgarh__raipur__aiims_raipur_cecb__site_5652.csv.gz",
    "Varanasi": "uttar_pradesh__varanasi__ardhali_bazar_varanasi_uppcb__site_273.csv.gz",
    "Vijayawada": "andhra_pradesh__vijayawada__hb_colony_vijayawada_appcb__site_5848.csv.gz",
    "Lucknow": "uttar_pradesh__lucknow__b_r_ambedkar_university_lucknow_uppcb__site_5460.csv.gz",
    "Bangalore": "karnataka__bengaluru__btm_layout_bengaluru_cpcb__site_162.csv.gz",
    "Amritsar": "punjab__amritsar__golden_temple_amritsar_ppcb__site_256.csv.gz",
    "Jodhpur": "rajasthan__jodhpur__collectorate_jodhpur_rspcb__site_136.csv.gz",
    "Ahmedabad": "gujarat__ahmedabad__chandkheda_ahmedabad_iitm__site_5453.csv.gz",
    "Guwahati": "assam__guwahati__lgbi_airport_guwahati_pcba__site_5683.csv.gz",
    "Mumbai": "maharashtra__mumbai__bandra_kurla_complex_mumbai_mpcb__site_5810.csv.gz",
    "Chennai": "tamil_nadu__chennai__alandur_bus_depot_chennai_cpcb__site_293.csv.gz",
    "Kanpur": "uttar_pradesh__kanpur__fti_kidwai_nagar_kanpur_uppcb__site_5500.csv.gz",
    "Agra": "uttar_pradesh__agra__manoharpur_agra_uppcb__site_5464.csv.gz",
    "Chandigarh": "chandigarh__chandigarh__sector_22_chandigarh_cpcc__site_5491.csv.gz",
    "Hyderabad": "telangana__hyderabad__bollaram_industrial_area_hyderabad_tspcb__site_199.csv.gz",
    # Polluted
    "Surat": "gujarat__surat__science_center_surat_smc__site_5664.csv.gz",
    "Bhopal": "madhya_pradesh__bhopal__paryavaran_parisar_bhopal_mppcb__site_5650.csv.gz",
    "Nagpur": "maharashtra__nagpur__ambazari_nagpur_mpcb__site_5792.csv.gz",
    "Jaipur": "rajasthan__jaipur__adarsh_nagar_jaipur_rspcb__site_1393.csv.gz",
    "Indore": "madhya_pradesh__indore__airport_area_indore_imc__site_5857.csv.gz",
    "Bhubaneswar": "odisha__bhubaneswar__lingraj_mandir_bhubaneswar_ospcb__site_5940.csv.gz",
    "Kolkata": "west_bengal__kolkata__ballygunge_kolkata_wbpcb__site_5238.csv.gz",
    "Pune": "maharashtra__pune__bhumkar_nagar_pune_iitm__site_5988.csv.gz",
    "Agartala": "tripura__agartala__bardowali_agartala_tripura_spcb__site_5587.csv.gz",
    "Delhi": "delhi__delhi__alipur_delhi_dpcc__site_5024.csv.gz",
}
VAYUAYAN_BASE = "https://saketkc.github.io/vayuayan-archive/data/"


def fetch_cpcb_pollution(city, filename):
    """Fetch CPCB station data from Vayuayan archive."""
    url = VAYUAYAN_BASE + filename
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"CPCB fetch failed for {city}: {r.status_code}")
    df = pd.read_csv(io.BytesIO(r.content), compression="gzip")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Only use columns that exist in this station's data
    rename = dict(zip(POLLUTANT_COLS, POLLUTANT_NAMES))
    available = [c for c in POLLUTANT_COLS if c in df.columns]
    poll_df = df[available].rename(columns=rename)
    # Force numeric (some stations have string values)
    for col in poll_df.columns:
        poll_df[col] = pd.to_numeric(poll_df[col], errors="coerce")
    # Add missing pollutants as NaN (will be filled later)
    for orig, new in zip(POLLUTANT_COLS, POLLUTANT_NAMES):
        if new not in poll_df.columns:
            poll_df[new] = np.nan
    poll_df = poll_df[POLLUTANT_NAMES]
    poll_df = poll_df.resample("1h").mean(numeric_only=True)
    return poll_df


def fetch_weather(city, lat, lon, start_date, end_date):
    """Fetch hourly weather from Open-Meteo Historical API (free, no key)."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": WEATHER_VARS, "timezone": "GMT",
    }
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Weather API error {r.status_code} for {city}")
    hourly = r.json().get("hourly", {})
    times = pd.to_datetime(hourly["time"], utc=True)
    df = pd.DataFrame({"timestamp": times})
    for var in WEATHER_VARS.split(","):
        col = var.replace("_2m", "").replace("_10m", "")
        df[col] = hourly.get(var, [np.nan] * len(times))
    return df.set_index("timestamp").sort_index().resample("1h").mean(numeric_only=True)


def add_features(df):
    """Add wind decomposition + cyclical time features."""
    ts = df.index
    if "wind_speed" in df.columns and "wind_direction" in df.columns:
        rad = np.deg2rad(df["wind_direction"])
        df["wind_u"] = df["wind_speed"] * np.cos(rad)
        df["wind_v"] = df["wind_speed"] * np.sin(rad)
        df.drop(columns=["wind_direction"], inplace=True)
    df["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)
    return df


def fetch_all():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)

    all_dfs = []
    for city in LOCATIONS:
        lat, lon = LOCATIONS[city]
        filename = CPCB_STATIONS.get(city)
        if not filename:
            print(f"  ⚠ No CPCB station for {city}, skipping")
            continue

        print(f"Fetching {city} ...")
        # Pollution from CPCB
        poll_df = fetch_cpcb_pollution(city, filename)
        time.sleep(0.3)

        # Weather from Open-Meteo (match CPCB date range)
        start_date = poll_df.index.min().strftime("%Y-%m-%d")
        end_date = poll_df.index.max().strftime("%Y-%m-%d")
        weather_df = fetch_weather(city, lat, lon, start_date, end_date)
        time.sleep(0.3)

        # Merge
        merged = poll_df.join(weather_df, how="inner")
        merged = add_features(merged)
        merged["city"] = city
        merged = merged.ffill().bfill()
        # Drop rows where target pollutants are still NaN
        merged = merged.dropna(subset=["pm2_5", "pm10"])

        if len(merged) < 500:
            print(f"  ⚠ Only {len(merged)} rows after merge, skipping {city}")
            continue

        merged.to_csv(os.path.join(RAW_DIR, f"{city}_raw.csv"))
        all_dfs.append(merged.reset_index())
        print(f"  → {len(merged)} rows, {merged.shape[1]} cols")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(PROC_DIR, "combined_data.csv"), index=False)

    feature_cols = [c for c in combined.columns if c not in ["timestamp", "city"]]
    print(f"\nSaved: {combined.shape}")
    print(f"Cities: {combined['city'].unique().tolist()}")
    print(f"Time range: {combined['timestamp'].min()} → {combined['timestamp'].max()}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    return combined


def build_sequences(combined_df):
    cities = sorted(combined_df["city"].unique())
    feature_cols = [c for c in combined_df.columns if c not in ["timestamp", "city"]]

    # Improvement #5: Log transform skewed pollutants before normalization
    log_cols = ["pm2_5", "pm10", "co", "no2", "so2", "o3", "nh3"]
    for col in log_cols:
        if col in combined_df.columns:
            combined_df[col] = np.log1p(combined_df[col].clip(lower=0))

    pivoted = {}
    for city in cities:
        cdf = combined_df[combined_df["city"] == city].set_index("timestamp")[feature_cols].sort_index()
        pivoted[city] = cdf

    common_ts = pivoted[cities[0]].index
    for city in cities[1:]:
        common_ts = common_ts.intersection(pivoted[city].index)
    common_ts = common_ts.sort_values()

    num_t, num_c, num_f = len(common_ts), len(cities), len(feature_cols)
    data_3d = np.zeros((num_t, num_c, num_f))
    for i, city in enumerate(cities):
        data_3d[:, i, :] = pivoted[city].loc[common_ts].values

    # Improvement #2: Use all pollutants as targets (multi-task learning)
    all_pollutant_targets = [c for c in log_cols if c in feature_cols]
    tgt_idx = [feature_cols.index(t) for t in all_pollutant_targets]
    # Also track PM2.5/PM10 indices for AQI evaluation
    pm_eval_idx = [all_pollutant_targets.index(t) for t in TARGET_POLLUTANTS]

    max_h = max(HORIZONS)

    X_list, Y_list, last_val_list = [], [], []
    for t in range(LOOKBACK, num_t - max_h):
        x = data_3d[t - LOOKBACK:t, :, :]
        # Current value (last timestep) for target pollutants
        current = data_3d[t - 1, :, :][:, tgt_idx]  # (cities, targets)
        # Future values
        future = np.stack([data_3d[t + h - 1, :, :][:, tgt_idx] for h in HORIZONS])  # (horizons, cities, targets)

        # Improvement #1: RESIDUAL PREDICTION — predict change from current value
        y_delta = future - current[np.newaxis, :, :]  # (horizons, cities, targets)

        X_list.append(x)
        Y_list.append(y_delta)
        last_val_list.append(current)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)  # Now contains DELTAS, not absolute values
    last_vals = np.array(last_val_list, dtype=np.float32)  # For inverse transform

    n = len(X)
    n_train = int(n * TRAIN_R)
    n_val = int(n * VAL_R)

    # Per-city normalization for X
    x_mean = np.nanmean(X[:n_train], axis=0)
    x_std = np.nanstd(X[:n_train], axis=0) + 1e-8
    x_mean = np.nanmean(x_mean, axis=0, keepdims=True)
    x_std = np.nanmean(x_std, axis=0, keepdims=True) + 1e-8
    x_mean = np.nan_to_num(x_mean, nan=0.0)
    x_std = np.nan_to_num(x_std, nan=1.0)

    # For deltas: normalize by training delta stats
    y_mean = np.nanmean(Y[:n_train], axis=0)
    y_std = np.nanstd(Y[:n_train], axis=0) + 1e-8
    y_mean = np.nanmean(y_mean, axis=0, keepdims=True)
    y_std = np.nanmean(y_std, axis=0, keepdims=True) + 1e-8
    y_mean = np.nan_to_num(y_mean, nan=0.0)
    y_std = np.nan_to_num(y_std, nan=1.0)

    X_norm = np.clip(np.nan_to_num((X - x_mean) / x_std, nan=0.0), -10, 10)
    Y_norm = np.clip(np.nan_to_num((Y - y_mean) / y_std, nan=0.0), -10, 10)

    X_train, Y_train = X_norm[:n_train], Y_norm[:n_train]
    X_val, Y_val = X_norm[n_train:n_train + n_val], Y_norm[n_train:n_train + n_val]
    X_test, Y_test = X_norm[n_train + n_val:], Y_norm[n_train + n_val:]
    last_vals_test = last_vals[n_train + n_val:]

    np.savez(os.path.join(PROC_DIR, "norm_stats.npz"),
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    np.savez(os.path.join(PROC_DIR, "sequences.npz"),
             X_train=X_train, Y_train=Y_train,
             X_val=X_val, Y_val=Y_val,
             X_test=X_test, Y_test=Y_test,
             last_vals_test=last_vals_test)

    meta = {"feature_names": feature_cols, "target_names": all_pollutant_targets,
            "pm_eval_idx": pm_eval_idx, "city_names": cities, "horizons": HORIZONS,
            "residual_prediction": True, "log_transformed": True}
    with open(os.path.join(PROC_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSequences (RESIDUAL prediction, log-transformed, multi-target):")
    print(f"  X_train {X_train.shape}, Y_train {Y_train.shape}")
    print(f"  X_val {X_val.shape}, X_test {X_test.shape}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Targets ({len(all_pollutant_targets)}): {all_pollutant_targets}")
    print(f"  PM eval indices: {pm_eval_idx} ({TARGET_POLLUTANTS})")
    print(f"  Common timestamps: {len(common_ts)}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


if __name__ == "__main__":
    combined = fetch_all()
    build_sequences(combined)
    print("\n✅ Data pipeline v2 complete (CPCB + Open-Meteo Weather).")
