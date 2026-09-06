# Scientific and Operational Limitations

1. **County vs. Feeder Boundary Discrepancy**: County-level outage aggregates from EAGLE-I reflect administrative boundaries, whereas utility distribution circuits span arbitrary geographical boundaries and utility operating territories.
2. **EAGLE-I Data Missingness & Reporting Latency**: Outage data collection depends on utility status page scraping; temporary network drops or utility page failures may introduce reporting gaps.
3. **Observational vs. Numerical Weather Forecasts**: In historical retrospective analysis, observational station data serve as the operational proxy. In real-time deployment, Numerical Weather Prediction (NWP) forecasts with inherent forecast error would be supplied at cutoff $t$.
4. **Synthetic Nature of Downstream IEEE 123-Bus Simulation**: The IEEE 123-bus test feeder is a standard benchmark circuit used to demonstrate methodological uncertainty propagation, not a literal representation of any single physical county.
5. **Non-Causal Interpretability**: Feature importances (SHAP, permutation importance) identify predictive associations under severe weather conditions and do not establish direct mechanical causality.
