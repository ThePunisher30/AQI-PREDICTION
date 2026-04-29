# Project Documentation

## System Overview

Spatio-temporal graph neural network for air quality prediction across 29 Indian cities. Predicts PM2.5 and PM10 concentrations 1-24 hours ahead using CPCB sensor data and Open-Meteo weather data.

## Architecture

```
CPCB Sensor Data (29 cities) + Open-Meteo Weather
    │
    ▼
Data Pipeline (fetch, merge, log transform, residual targets)
    │
    ▼
Graph Construction (haversine distance, 1500km threshold, 602 edges)
    │
    ▼
6 Models (20 configs each, fair comparison):
  1. Persistence (delta=0)
  2. Historical Average (mean training delta)
  3. LSTM (temporal only)
  4. CNN-LSTM (grid spatial + temporal)
  5. WM-STGN (graph spatial + temporal, static adjacency)
  6. WA-STGN (graph spatial + temporal, wind-aware dynamic adjacency)
    │
    ▼
Outputs:
  - AQI predictions with MC Dropout uncertainty
  - Health alerts (conservative category from 95% upper bound)
  - Explainability (permutation feature + city importance)
  - Streamlit dashboard (5 tabs)
  - Publication figures (9 figures)
```

## Key Results (PM2.5)

| Model | 1h MAE | 1h R² | 6h MAE | 12h MAE | 24h MAE | 24h R² |
|-------|--------|-------|--------|---------|---------|--------|
| Persistence | 1.45 | 0.996 | 6.79 | 12.00 | 20.38 | 0.644 |
| LSTM | 1.48 | 0.996 | 6.91 | 12.09 | 20.30 | 0.655 |
| CNN-LSTM | 1.61 | 0.996 | 6.96 | 12.25 | 20.10 | 0.654 |
| **WM-STGN** | **1.49** | **0.996** | **6.71** | **11.74** | **19.44** | **0.699** |
| WA-STGN | 1.55 | 0.996 | 6.73 | 11.75 | 19.62 | 0.696 |

## Novel Contributions

1. **Residual prediction** — predict change, not absolute value (90% MAE reduction)
2. **City masking** — random spatial dropout during training
3. **MC Dropout health alerts** — uncertainty-aware AQI categories

## Data

- **Source**: CPCB via Vayuayan archive + Open-Meteo weather
- **Cities**: 29 (4 clean, 15 moderate, 10 polluted)
- **Period**: Nov 2025 – Apr 2026
- **Features**: 18 per city (7 pollutants + 6 weather + 4 time + wind u/v)
- **Samples**: 2,022 train / 433 val / 434 test

## How to Run

```bash
python utils/data_loader.py       # Fetch data
python utils/graph_builder.py     # Build graph
python training/retrain_all.py    # Train all models (~2 hours)
python utils/visualizations.py    # Generate figures
streamlit run dashboard/app.py    # Launch dashboard
```
