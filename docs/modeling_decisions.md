# Modeling Decisions & Methodology Rationale

## 1. Multi-Task Target Formulation
We formulate outage risk prediction under two complementary tasks:
1. **Task A: Severe-Outage Classification ($P(\text{severe} \mid X_t)$)**:
   $$\text{Target}_{i,t}^H = \mathbb{I}\left( \max_{h \in [1, H]} \text{outage\_fraction}_{i, t+h} \ge \tau \right)$$
   where horizon $H \in \{6, 12, 24\}$ hours, default severe threshold $\tau = 0.01$ (1% of county customer base), with sensitivity ablations across $\tau \in \{0.005, 0.01, 0.02, 0.05\}$.
2. **Task B: Outage-Severity Regression & Quantiles**:
   $$Y_{i,t}^H = \max_{h \in [1, H]} \text{outage\_fraction}_{i, t+h}$$
   Estimated using quantile gradient boosting and conformal prediction intervals at nominal coverage levels $\alpha \in \{0.80, 0.90, 0.95\}$.

## 2. Model Taxonomy
- **Statistical Baselines**: Climatological empirical rates, Persistence rule (recent status), Regularized Logistic / Ridge GLM.
- **Gradient Boosted Decision Trees**: LightGBM and XGBoost with histogram-based splitting, imbalance handling (scale_pos_weight), and quantile loss functions.
- **Temporal Neural Network**: Uni-directional PyTorch GRU / LSTM / TCN encoders over causal historical sequence windows $L \in \{6, 12, 24, 48\}$ hours.
- **Spatiotemporal Graph Neural Network**: GRU temporal encoder coupled with GraphSAGE / GCN spatial message passing over county contiguity graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, A)$.

## 3. Probability Calibration & Conformalization
Because severe outages are extreme rare events ($<2\%$ prevalence), raw model scores often exhibit miscalibration. We implement:
- Isotonic regression and Platt scaling on held-out validation predictions.
- Split Conformal Prediction for distribution-free finite-sample coverage guarantees on regression prediction intervals.
