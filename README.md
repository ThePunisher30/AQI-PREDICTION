# Air Quality Prediction using Spatio-Temporal Graph Neural Networks

## Overview

An end-to-end system that predicts air quality across **29 Indian cities** up to 24 hours ahead using graph neural networks on real CPCB sensor data, with uncertainty-aware health alerts and an interactive dashboard.

**Key Innovation:** Residual prediction — forecasting the *change* in pollution rather than the absolute value — reduces 1-hour PM2.5 MAE by 90%.

## Results

| Model | 1h MAE | 1h R² | 6h MAE | 12h MAE | 24h MAE | 24h R² |
|-------|--------|-------|--------|---------|---------|--------|
| Persistence | 1.45 | 0.996 | 6.79 | 12.00 | 20.38 | 0.644 |
| LSTM | 1.48 | 0.996 | 6.91 | 12.09 | 20.30 | 0.655 |
| CNN-LSTM | 1.61 | 0.996 | 6.96 | 12.25 | 20.10 | 0.654 |
| **WM-STGN** | **1.49** | **0.996** | **6.71** | **11.74** | **19.44** | **0.699** |
| WA-STGN | 1.55 | 0.996 | 6.73 | 11.75 | 19.62 | 0.696 |

*PM2.5 MAE in µg/m³. WM-STGN beats all models at 6h+ horizons.*

## Quick Start

```bash
pip install -r requirements.txt

# 1. Fetch data (29 cities, CPCB + Open-Meteo weather)
python utils/data_loader.py

# 2. Build city graph (haversine adjacency)
python utils/graph_builder.py

# 3. Train all models (baselines + LSTM + CNN-LSTM + WM-STGN + WA-STGN)
python training/retrain_all.py

# 4. Generate figures for paper
python utils/visualizations.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
aqi/
├── configs/config.yaml              # City coordinates, hyperparameters
├── models/                          # Model definitions
│   ├── baselines.py                 # Persistence + Historical Average
│   ├── lstm.py                      # LSTM model factory
│   ├── cnn_lstm.py                  # CNN-LSTM model factory
│   ├── stgcn.py                     # WM-STGN (static graph GNN)
│   └── wastgn.py                    # WA-STGN (wind-aware dynamic graph GNN)
├── training/                        # Training scripts
│   ├── common.py                    # Shared: load(), inverse_residual(), eval_pm()
│   ├── training_lstm.py             # LSTM: 20 hyperparameter configs
│   ├── training_cnn_lstm.py         # CNN-LSTM: 20 configs
│   ├── training_wmstgn.py           # WM-STGN: 20 configs + ensemble + MC Dropout
│   ├── training_wastgn.py           # WA-STGN: 20 configs + ensemble + MC Dropout
│   └── retrain_all.py               # Orchestrator: runs all models
├── utils/                           # Utilities
│   ├── data_loader.py               # CPCB + Open-Meteo fetch, preprocessing, sequences
│   ├── graph_builder.py             # Haversine adjacency matrix
│   ├── metrics.py                   # MAE, RMSE, R²
│   ├── aqi_calculator.py            # EPA breakpoint AQI + health categories
│   └── visualizations.py            # Publication figure generator (9 figures)
├── dashboard/app.py                 # Streamlit dashboard (5 tabs)
├── figures/                         # Publication-quality figures
├── data/
│   ├── raw/                         # Per-city CPCB + weather CSVs
│   ├── processed/                   # Sequences, model weights, results
│   └── graphs/                      # Adjacency matrices
├── notebooks/                       # Literature review, documentation
├── paper.md                         # Research paper
└── requirements.txt
```

## Data

- **Pollution**: CPCB real sensor data via Vayuayan archive (29 cities, Nov 2025 – Apr 2026)
- **Weather**: Open-Meteo Historical API (temperature, humidity, wind, pressure, precipitation)
- **Features**: 18 per city (7 pollutants + 6 weather + 4 time encodings + wind u/v)
- **Target**: Residual prediction (change in concentration) for 7 pollutants
- **Horizons**: 1h, 6h, 12h, 24h

## Models

| Model | Spatial | Temporal | Key Feature |
|-------|---------|----------|-------------|
| Persistence | ❌ | ❌ | Predicts no change (delta=0) |
| Historical Avg | ❌ | ❌ | Predicts mean training delta |
| LSTM | ❌ | ✅ | Temporal patterns only |
| CNN-LSTM | Grid | ✅ | Conv1D spatial + LSTM temporal |
| WM-STGN | Graph (static) | ✅ | Adaptive adjacency + GLU + city masking |
| WA-STGN | Graph (dynamic) | ✅ | Wind-aware edges + pressure gradient + diffusion |

## Novel Contributions

1. **Residual prediction** for spatio-temporal GNN — 90% MAE reduction
2. **City masking** during training — spatial regularization
3. **MC Dropout health alerts** — uncertainty-aware AQI categories

## Hardware

Optimized for Apple M4 Pro (14 cores, 48GB RAM, MPS GPU). Full training: ~2 hours.
