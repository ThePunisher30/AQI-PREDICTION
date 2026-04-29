> **Note**: This covers 7 additional papers (2024-2026) not in the original review. Key findings are incorporated into `paper.md`.

# Extended Literature Review & Research Improvement Plan
## Updated: 2026-04-04

---

## 1. Latest State-of-the-Art (2025-2026 Papers)

### 1.1 TransNet (Nature, Feb 2026) — Current SOTA
- **Transport-Informed Graph Neural Network** for PM2.5 forecasting across South Korea
- Learns coupled **Advection-Diffusion-Reaction (ADR) operators** — physics equations embedded in the GNN
- Forecasts up to **+72 hours** ahead
- Key innovation: the graph edges don't just represent distance — they represent **physical transport pathways** (how pollution actually moves through the atmosphere)
- **Why it matters for us:** Our WM-STGN uses distance-based + learnable adjacency. TransNet shows that encoding *physics* (wind-driven transport) into the graph structure is the next frontier. Our "wind-aware" concept was on the right track but didn't go far enough.

### 1.2 CEEMDAN-CNN-IGWO-BiGRU-Attention (Nature Scientific Reports, 2026)
- **Signal decomposition approach**: uses CEEMDAN to break AQI time series into multiple frequency components (trend, seasonal, noise)
- Each component predicted separately by CNN + BiGRU + Attention
- Hyperparameters optimized by Improved Grey Wolf Optimizer (IGWO)
- **Why it matters:** This "decompose then predict" approach is fundamentally different from our direct prediction. It's why they achieve better results — the model doesn't have to learn both the trend AND the fluctuations simultaneously.

### 1.3 Physics-Informed Multimodal Framework (MDPI Systems, 2026)
- Combines **Gaussian plume dispersion model** (physics) with **residual CNN-LSTM** (data-driven)
- Physics model provides baseline prediction, neural network learns the **residual** (correction)
- Uses satellite imagery + ground sensors + meteorological data (multimodal)
- R² > 0.85, 25% lower MAE than pure data-driven baselines
- **Why it matters:** This is exactly the "residual prediction" approach we identified. It's now published and validated. The physics model acts as a better "persistence" baseline, and the neural network only needs to learn corrections.

### 1.4 MSDGNN — Multi-Scale Dynamic Graph Neural Network (PLOS ONE, 2025)
- Multi-scale temporal modeling: **hourly, daily, weekly** patterns captured separately
- Dynamic graph that changes per timestep (not fixed)
- Reduces MAE by 6.77% and RMSE by 8.67% vs best baseline
- **Why it matters:** Validates our earlier idea of multi-scale temporal decomposition. The key insight: different temporal scales need different graph structures (hourly pollution transport ≠ weekly seasonal patterns).

### 1.5 AirPhyNet (ICLR 2024) — Physics-Guided GNN
- Embeds advection and diffusion equations as differential equation networks
- Outperforms SOTA by up to 10% on 24h-72h predictions
- Especially strong for **sudden change prediction** (pollution spikes)
- **Why it matters:** Shows that physics constraints help most at longer horizons and during regime changes — exactly where our model struggles.

### 1.6 TopoFlow (arXiv, Feb 2026) — Physics-Guided for China
- Trained on 6 years of data from 1,400+ stations across China
- PM2.5 RMSE of 9.71 µg/m³
- 71-80% improvement over operational forecasting systems
- 13% improvement over SOTA AI baselines
- **Why it matters:** Shows what's possible with enough data (6 years vs our 3.5 months). Also shows that physics guidance is the key differentiator.

### 1.7 Data Augmentation for PM2.5 (MDPI Atmosphere, 2025)
- Models trained on augmented data **significantly outperform** those on original data
- Especially important for high-pollution scenarios (rare events)
- Techniques: oversampling high-pollution periods, synthetic minority oversampling
- **Why it matters:** We have limited data (3.5 months). Augmentation could help without needing more real data.

---

## 2. Key Research Gaps We Can Address

### Gap 1: No Physics-Guided Residual GNN for Indian Cities
- TransNet (Korea), TopoFlow (China), AirPhyNet (general) all use physics
- **Nobody has applied physics-guided residual learning to Indian city-level AQ prediction**
- India has unique challenges: crop burning, Diwali, extreme temperature inversions, monsoon effects
- Our contribution: adapt the residual prediction approach with Indian-specific physics priors

### Gap 2: Signal Decomposition + Graph Neural Network
- CEEMDAN papers decompose the signal but don't use spatial graphs
- Graph papers use spatial structure but don't decompose the signal
- **Nobody combines signal decomposition with spatio-temporal GNN**
- Our contribution: CEEMDAN decomposition → separate GNN for each frequency component → fusion

### Gap 3: Uncertainty-Aware Health Alerts with Spatial Context
- MC Dropout for uncertainty exists but nobody maps it to health categories with spatial escalation
- No paper considers: "if Delhi is Very Unhealthy and wind blows toward Lucknow, escalate Lucknow's alert"
- Our contribution: spatially-informed uncertainty escalation

### Gap 4: Small-Data Regime for AQ Prediction
- Most papers use 1-6 years of data
- Nobody addresses: how to build a good AQ prediction system with only 3-4 months of data
- This is the reality for many developing countries where monitoring just started
- Our contribution: techniques that work with limited data (masking, augmentation, per-city normalization)

---

## 3. Concrete Improvements for Our Project (Ranked by Impact)

### HIGH IMPACT — Should Implement

#### 3.1 Residual Prediction (from Physics-Informed Multimodal paper)
- **What:** Instead of predicting PM2.5(t+h), predict PM2.5(t+h) - PM2.5(t)
- **Why:** Persistence baseline is hard to beat because pollution is autocorrelated. By predicting the *change*, the model starts from persistence and only needs to learn corrections.
- **Expected improvement:** Should make GNN competitive with or better than persistence at all horizons
- **Implementation:** Change target computation in data_loader.py (5 lines of code)
- **Literature support:** Physics-Informed Multimodal (2026), VARNN (2025)

#### 3.2 Signal Decomposition (from CEEMDAN paper)
- **What:** Decompose PM2.5 time series into trend + seasonal + residual using STL or CEEMDAN, predict each component separately
- **Why:** The model currently tries to learn everything at once. Decomposition separates easy patterns (daily cycle) from hard patterns (sudden spikes).
- **Expected improvement:** 10-20% MAE reduction based on CEEMDAN paper results
- **Implementation:** Add STL decomposition before sequence building (~30 lines)
- **Literature support:** CEEMDAN-CNN-BiGRU (2026), STL-LOESS decomposition (2023)

#### 3.3 Multi-Scale Temporal Modeling (from MSDGNN paper)
- **What:** Process recent (6h), daily (24h), and weekly (168h) patterns with separate encoders, then fuse
- **Why:** Different temporal scales capture different phenomena (rush hour vs seasonal)
- **Expected improvement:** 5-10% based on MSDGNN results
- **Implementation:** Modify sequence builder to create multi-scale inputs (~40 lines)
- **Literature support:** MSDGNN (2025)

### MEDIUM IMPACT — Nice to Have

#### 3.4 Data Augmentation for Rare Events
- **What:** Oversample high-pollution periods (AQI > 200), add jittered copies
- **Why:** Model sees mostly moderate pollution, underperforms on spikes
- **Implementation:** Add augmentation in data_loader.py (~20 lines)
- **Literature support:** Data Augmentation for PM2.5 (2025)

#### 3.5 Dynamic Graph per Timestep
- **What:** Recompute adjacency matrix at each timestep based on current wind conditions
- **Why:** Pollution transport depends on current wind, not just distance
- **Implementation:** Modify WMSTGN forward pass to use wind features for edge weights (~30 lines)
- **Literature support:** MSDGNN (2025), TransNet (2026)

### LOWER IMPACT — Future Work

#### 3.6 Physics-Informed Loss Function
- Add advection-diffusion constraint as regularization term in loss
- Requires implementing simplified atmospheric transport equations
- **Literature support:** AirPhyNet (2024), TransNet (2026)

#### 3.7 Satellite Data Integration
- Add satellite-derived AOD (Aerosol Optical Depth) as additional feature
- Available from NASA MODIS/VIIRS
- **Literature support:** Physics-Informed Multimodal (2026), TopoFlow (2026)

---

## 4. Revised Paper Strategy

### Title (Updated)
**"Residual Spatio-Temporal Graph Networks with Multi-Scale Decomposition and Spatial Masking for Urban Air Quality Forecasting in India"**

### Novel Claims (Updated)
1. **Residual graph prediction** — predict change from current state, not absolute value (distinct from CEEMDAN which decomposes but doesn't use graphs, and from TransNet which uses physics but not residual framing)
2. **Signal decomposition + GNN fusion** — CEEMDAN/STL decomposition with separate graph encoders per component (nobody has combined these)
3. **Spatial masking as training regularization** — random city dropout during training (distinct from AirRadar's inference-time masking)
4. **Uncertainty-escalated health alerts** — MC Dropout → conservative category escalation
5. **Small-data regime analysis** — systematic study of what works with 3.5 months vs what needs years

### Target Venues
- **Tier 1:** Nature Scientific Reports, IEEE TNNLS, Environmental Science & Technology
- **Tier 2:** Frontiers in Environmental Science, MDPI Atmosphere, PLOS ONE
- **Tier 3:** IEEE Access, MDPI Sustainability, regional conferences (AAAI workshop, ACM SIGSPATIAL)

---

## 5. What We Should Do Next (Priority Order)

1. **Implement residual prediction** — single biggest improvement, 5 lines of code
2. **Retrain on 29-city dataset** — interrupted last time, need to complete
3. **Add STL decomposition** — second biggest improvement
4. **Run ablation study** — with vs without each component (residual, masking, decomposition)
5. **Build final dashboard** — already built, just needs retraining results
6. **Write paper** — structure already outlined in master document
