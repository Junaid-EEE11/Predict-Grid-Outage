# Research Protocol & Pre-Registration Plan

**Project Title**: From Outage Prediction to Grid Decisions: Uncertainty-Calibrated Spatiotemporal Learning and Physics-Based Resilience Assessment for Weather-Driven Distribution Outages  
**Status**: Pre-registered before model development  
**Date**: September 2026  

## 1. Research Questions (RQ1?RQ6)

- **RQ1**: How accurately can county-level severe outage risk be predicted 6, 12, and 24 hours ahead using strictly information available at the prediction cutoff $t$?
- **RQ2**: Do spatial relationships between neighboring counties provide statistically meaningful predictive improvement beyond strong non-spatial statistical and machine-learning baselines?
- **RQ3**: Do temporal deep-learning models provide meaningful improvement over persistence, regularized generalized linear models, and gradient-boosted trees after leakage is removed?
- **RQ4**: How well calibrated are the predicted probabilities and prediction intervals, particularly during rare high-impact events?
- **RQ5**: How robust are models under temporal distribution shift, extreme-weather events, and geographic transfer?
- **RQ6**: How does predictive uncertainty affect downstream estimates of distribution-grid resilience when outage-risk scenarios are propagated through an IEEE 123-bus OpenDSS feeder?

## 2. Pre-Registered Hypotheses (H1?H6)

- **H1**: Weather, lagged outage behavior, and temporal features will outperform simple persistence/climatology for 6?24 h outage-risk prediction.
- **H2**: Spatial information will improve performance primarily during geographically widespread weather events, rather than uniformly across all observations.
- **H3**: A complex spatiotemporal neural model will not necessarily outperform gradient boosting after strict leakage prevention; this is an empirical question.
- **H4**: Uncalibrated high-capacity models will exhibit poorer probability calibration than their discrimination metrics suggest.
- **H5**: Calibration procedures using validation data only will improve Brier score and expected calibration error (ECE) without using test-set information.
- **H6**: Forecast uncertainty will materially widen the distribution of simulated feeder resilience outcomes (energy not served, restoration trajectory) during high-risk events.

## 3. Experimental Design & Splitting Protocol

1. **Chronological Splitting**:
   - **Training Set**: Years 2017?2020 (temporal fitting only)
   - **Validation Set**: Year 2021 (hyperparameter selection, probability calibration fitting, conformal quantile calibration)
   - **Held-out Test Set**: Year 2022 (final blind evaluation)
   - **Purge Buffer**: 24-hour purge between chronological partitions to eliminate target overlap across boundary points.
2. **Event-Held-Out Splitting**:
   - Stratified evaluation using NOAA Storm Events episode IDs to evaluate severe episode generalization.
3. **Geographic Transfer Splitting**:
   - Held-out transfer state/region (e.g., Virginia region) never seen during training or validation.

## 4. Primary Decision Metrics

- **Classification**: Precision-Recall AUC (PR-AUC), Brier Score, Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), ROC-AUC, F1 at fixed operating thresholds. (Accuracy is strictly prohibited as a primary metric due to severe class imbalance).
- **Regression**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Continuous Ranked Probability Score (CRPS), Empirical Interval Coverage (80%, 90%, 95%), Mean Interval Width (MPIW).
- **Inference**: Event-aware block bootstrap with 1,000 resamples for 95% confidence intervals on all test metrics.

## 5. Non-Fabrication Commitment
All results reported in tables, figures, and manuscripts must originate from executable scripts and recorded experiment outputs. If external observational data are inaccessible in a specific environment, self-contained reproducible synthetic fixtures conforming to identical schemas are used and explicitly documented.
