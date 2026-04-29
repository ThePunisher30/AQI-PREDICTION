> **Note**: This is the detailed 25-paper literature review. A condensed version is in `paper.md` Section II.

# Literature Review: Spatio-Temporal Deep Learning for Air Quality Prediction

## 1. Traditional and Statistical Methods

### [1] Box, Jenkins, and Reinsel — Time Series Analysis (2015)
The foundational text on ARIMA and related time series methods. ARIMA models assume linearity and stationarity, making them suitable for short-term univariate forecasting but fundamentally limited for air quality prediction where (a) pollutant dynamics are highly nonlinear, (b) spatial dependencies between monitoring stations are critical, and (c) external factors like weather drive pollution transport. The Box-Jenkins methodology remains relevant as a baseline comparison — our Historical Average baseline is a simplified version of this approach.

### [2] Kumar, Goyal, and Singh — ML for Air Quality (2018)
Demonstrates that machine learning approaches (SVR, Random Forest, ANN) outperform traditional statistical methods for AQ prediction. Key finding: incorporating meteorological features (temperature, humidity, wind speed) alongside pollutant concentrations improves prediction accuracy by 15-25%. **This directly supports our decision to include weather features in the model.** However, these methods treat each station independently, ignoring spatial correlations — a limitation our graph-based approach addresses.

---

## 2. Deep Learning for Temporal Modeling

### [13] Hochreiter and Schmidhuber — LSTM (1997)
The seminal paper introducing Long Short-Term Memory networks. LSTMs solve the vanishing gradient problem in RNNs through gating mechanisms (input, forget, output gates) that control information flow. For air quality, LSTMs capture temporal patterns like diurnal cycles and weekly trends. **Our plain LSTM baseline directly implements this architecture.** Limitation: LSTMs process sequences element-by-element, making them slow for long sequences and unable to capture spatial dependencies without explicit spatial encoding.

### [14] Cho et al. — GRU (2014)
Introduces the Gated Recurrent Unit, a simplified variant of LSTM with only two gates (reset and update) instead of three. GRUs achieve comparable performance to LSTMs with fewer parameters, making them preferable when training data is limited. **Given our small dataset (84 training samples), GRU may be more appropriate than LSTM for the recurrent components of our model.** The AAMGCRN paper (Chen et al., 2024) confirms this — they use GRU inside their graph-recurrent cell.

### [15] Bai, Kolter, and Koltun — TCN vs RNN (2018)
A systematic comparison showing that Temporal Convolutional Networks (TCNs) can match or exceed RNNs/LSTMs for sequence modeling tasks. TCNs offer: (a) parallelizable computation, (b) stable gradients via residual connections, (c) flexible receptive fields via dilated convolutions. **This validates our STGCN's use of temporal convolutions rather than recurrent units for the temporal component.** The original STGCN paper [7] uses gated temporal convolutions (GLU activation) based on this insight.

### [25] Shi et al. — ConvLSTM (2015)
Proposes Convolutional LSTM for precipitation nowcasting, replacing the fully-connected transforms in LSTM gates with convolution operations. The key innovation: by making both input-to-state and state-to-state transitions convolutional, the model captures spatial correlations within the recurrent structure itself. **This is the conceptual foundation for embedding graph convolutions inside recurrent gates** — the approach used by AAMGCRN and recommended for our improved STGCN. ConvLSTM operates on regular grids; our adaptation replaces 2D convolutions with graph convolutions to handle irregular station networks.

---

## 3. Graph Neural Networks

### [11] Kipf and Welling — GCN (2017)
The foundational paper on Graph Convolutional Networks. Proposes a first-order approximation of spectral graph convolutions: H = σ(D̃^(-1/2) Ã D̃^(-1/2) X W), where Ã = A + I (adjacency with self-loops) and D̃ is the degree matrix. This propagation rule aggregates features from 1-hop neighbors. **Our STGCN's spatial component is built on this.** Key limitation: the adjacency matrix A is fixed and predefined, which cannot capture dynamic or hidden spatial dependencies.

### [12] Hamilton, Ying, and Leskovec — GraphSAGE (2017)
Introduces inductive graph learning — the ability to generalize to unseen nodes without retraining. Unlike transductive methods (Kipf's GCN), GraphSAGE learns aggregation functions rather than node-specific embeddings. **This is directly relevant to our spatial interpolation task** (leave-one-station-out), where we need to predict AQI at a city not seen during training. Using learnable node features (rather than fixed node IDs) enables this inductive capability.

### [10] Wu et al. — GNN Survey (2021)
Comprehensive survey classifying GNNs into spectral-based (our Chebyshev approach), spatial-based (message passing), and attention-based (GAT) methods. Key insight for our work: **adaptive/learnable adjacency matrices consistently outperform fixed predefined ones** across all surveyed applications. The survey identifies three strategies for learning adjacency: (a) attention-based (GAT), (b) embedding-based (Graph WaveNet), (c) data-driven (AAMGCRN). We adopt strategy (b) as it's simplest and most effective for small datasets.

---

## 4. Spatio-Temporal Graph Models

### [7] Yu, Yin, and Zhu — STGCN (IJCAI 2018)
**The paper our STGCN implementation is based on.** Proposes a purely convolutional architecture for spatio-temporal forecasting: ST-Conv blocks consisting of temporal gated convolution → spatial graph convolution → temporal gated convolution. Key design choices:
- **Gated Linear Units (GLU)** for temporal convolutions: Γ = (W₁ * X + b₁) ⊙ σ(W₂ * X + b₂), where ⊙ is element-wise multiplication and σ is sigmoid. **Our current implementation uses ReLU instead of GLU — this should be changed.**
- **Chebyshev polynomials** for spatial convolution (K-hop neighborhood aggregation)
- **Residual connections** between ST-Conv blocks
- No recurrent units — faster training than RNN-based alternatives

The paper reports significant improvements over LSTM baselines on traffic data. However, the fixed adjacency matrix limits its ability to capture dynamic spatial dependencies.

### [24] Wu et al. — Graph WaveNet (IJCAI 2019)
Extends STGCN with two critical innovations:
1. **Adaptive adjacency matrix**: A_adp = softmax(ReLU(E₁ · E₂ᵀ)), where E₁, E₂ ∈ ℝ^(N×c) are learnable node embedding matrices. This discovers hidden spatial dependencies without prior knowledge. The final adjacency combines fixed and adaptive: A = A_fixed + A_adp.
2. **Dilated causal convolutions** (from WaveNet) for temporal modeling, with exponentially growing receptive fields.

**This is the most directly applicable improvement for our model.** The adaptive adjacency is simple to implement (~10 lines of code) and consistently improves performance. The dilated convolutions handle long-range temporal dependencies better than our current fixed-kernel temporal convolutions.

### [4] Zhang et al. — DNN for Spatio-Temporal Data (KDD 2018)
Proposes ST-ResNet for citywide crowd flow prediction. Key contribution: **residual learning for spatio-temporal blocks**. Each block's output is: Y = F(X) + X (or Y = F(X) + W·X when dimensions differ). This addresses gradient degradation in deep networks and allows stacking more layers. **Our STGCN currently lacks residual connections — adding them is a one-line change per block that improves gradient flow, especially critical with our small dataset.**

---

## 5. Air Quality Specific Models

### [8] Zheng, Liu, and Hsieh — U-Air (KDD 2013)
Pioneering work on urban air quality inference using spatial correlations. Models AQ as a function of: (a) meteorological data, (b) traffic patterns, (c) POI (Points of Interest), (d) road network structure. **Establishes that multi-source data fusion is essential for AQ prediction.** Our model currently uses only pollutant + time features; adding weather data (as recommended) aligns with this foundational work.

### [9] Yi et al. — Deep Distributed Fusion (KDD 2018)
Multi-group encoder-decoder for AQ prediction that fuses heterogeneous data sources. Key finding: **separate encoding branches for different data modalities (pollution, weather, spatial) with late fusion outperform single-branch architectures.** This supports the dual-path architecture recommendation: one branch for spatio-temporal (GCN+RNN) and another for temporal-only (RNN on weather data), merged via a gating mechanism.

### [5] Chen, Wang, and Zhang — Spatio-Temporal DL for AQ (IEEE TNNLS 2021)
Systematic evaluation showing that models capturing both spatial and temporal dependencies reduce MAE by 15-25% compared to temporal-only models. **Directly validates our STGCN approach over the plain LSTM baseline.** Also demonstrates that prediction accuracy degrades significantly beyond 24h horizons with limited training data — consistent with our observation that all models perform poorly at the 24h horizon.

### [3] Li et al. — Traffic and Air Pollution (AAAI 2019)
Shows that traffic patterns are a leading indicator of air pollution. While we don't have traffic data, this paper's methodology of using auxiliary time-series as predictive features supports our use of cyclical time encodings (hour_sin, hour_cos, dow_sin, dow_cos) to capture traffic-related pollution patterns implicitly.

### [6] Wang, Duan, and Wang — GNN for AQ Survey (IEEE Access 2022)
Survey of graph neural network approaches for AQ prediction. Identifies three types of spatial dependencies:
1. **Geographic proximity** (our haversine-based adjacency)
2. **Functional similarity** (similar land use/POI — we don't have this data)
3. **Temporal pattern similarity** (correlated pollution time series)

Recommends combining multiple graph types. **For our project, we can construct a temporal correlation graph from the training data (Pearson correlation between cities' pollution time series) and combine it with the distance graph.** This is a lightweight way to capture functional similarity without POI data.

---

## 6. Hybrid and Advanced Approaches

### [16] Wu, Tan, and Qin — Hybrid DL for AQ (2019)
Proposes combining CNN for spatial feature extraction with LSTM for temporal modeling. Shows that the hybrid approach outperforms either component alone. **Our STGCN is a more principled version of this — using graph convolutions instead of regular CNNs to handle the non-Euclidean structure of monitoring station networks.**

### [17] Qin, Liu, and Zhang — CNN-LSTM for AQ (2019)
Demonstrates that CNN-LSTM hybrids achieve good results for single-station AQ prediction. However, the CNN component operates on a regular grid, which is suboptimal for irregularly spaced monitoring stations. **This limitation is exactly what graph-based approaches (our STGCN) solve.**

### [18] Pak, Kim, and Park — Urban AQ Forecasting (2020)
Evaluates multiple deep learning architectures for urban AQ. Key finding: **model performance is highly sensitive to the lookback window length.** Optimal lookback varies by pollutant: PM2.5 benefits from longer windows (48-72h) while O3 benefits from shorter windows (12-24h). **This suggests our fixed 24h lookback may be suboptimal — we should experiment with 48h.**

### [19] Li, Zheng, and Zhang — DL for Urban AQ (IJCAI 2020)
Proposes attention-based temporal weighting to learn which past timesteps are most informative. Shows that recent hours (1-6h) dominate for short-term prediction, while diurnal patterns (24h, 48h ago) matter for longer horizons. **This supports using attention mechanisms in the temporal component, though for our small dataset, the simpler GLU gating from [7] may be more appropriate.**

### [20] Gao and Li — ST-GNN for AQ (IEEE TKDE 2021)
Comprehensive ST-GNN framework for AQ. Key architectural insight: **the spatial graph should be dynamic (changing per timestep) rather than static.** Pollution transport depends on current wind conditions, so the effective spatial connectivity changes over time. While a fully dynamic graph is complex, the adaptive adjacency from [24] partially addresses this by learning a data-driven spatial structure.

### [21] Guo, Zhang, and Wang — GAT for AQ (IEEE Access 2020)
Uses Graph Attention Networks instead of spectral GCN for spatial modeling. GAT learns attention weights between nodes dynamically, effectively creating a soft adaptive adjacency. **For our small dataset, the simpler Graph WaveNet-style adaptive adjacency [24] is preferable to GAT, which has more parameters to learn.**

### [22] Zheng — Cross-Domain Data Fusion (IEEE TBD 2015)
Overview of methodologies for fusing heterogeneous urban data. Establishes principles for combining pollution, weather, traffic, and geographic data. **Validates our multi-feature approach and suggests that even simple feature concatenation (our current method) is effective when combined with appropriate spatial-temporal modeling.**

### [23] Zhang, Wang, and Zheng — ST Learning for Environmental Data (AAAI 2021)
Proposes spatio-temporal learning specifically for environmental monitoring. Key contribution: **transfer learning between cities** — training on data-rich cities and fine-tuning on data-poor ones. While we skip transfer learning today, this is a strong future work direction for our project.

---

## 7. Summary of Key Insights for Our Model

| Insight | Source Papers | Impact on Our Model |
|---------|--------------|-------------------|
| Weather features are essential for AQ prediction | [2], [8], [9], [5] | **Add temp, humidity, wind_speed, pressure** |
| Adaptive/learnable adjacency outperforms fixed | [10], [24], [20], [21] | **Implement Graph WaveNet-style adaptive adj** |
| GLU activation > ReLU for temporal convolutions | [7], [15] | **Replace ReLU with GLU in temporal conv layers** |
| Residual connections improve deep ST models | [4], [7] | **Add skip connections in STConvBlocks** |
| GRU preferred over LSTM for small datasets | [14], [13] | **Use GRU in graph-recurrent variant** |
| Embedding conv inside RNN gates captures ST jointly | [25] | **GCN-GRU architecture (ConvLSTM concept)** |
| Dual-path (ST + temporal-only) with fusion | [9], [16] | **Add parallel temporal branch with gating** |
| Lookback window affects performance significantly | [18] | **Experiment with 48h lookback** |
| Temporal correlation graph adds spatial info | [6] | **Combine distance + correlation adjacency** |
| Inductive learning enables spatial interpolation | [12] | **Use learnable node features for LOO** |
