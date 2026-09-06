# Leakage Audit & Guardrails

## 1. Strict Cutoff Timestamp Rule ($t$)
For any prediction generated at forecast cutoff timestamp $t$:
- All input features $X_t$ must be strictly computed from observations $s \le t$.
- Future observations $s > t$ (weather observations, customer counts, outage logs) are strictly prohibited.
- Preprocessing scalers (mean, variance, min-max) and quantile conformers must be fit **exclusively on the training dataset** ($t \in \mathcal{T}_{\text{train}}$) and applied without modification to validation and test datasets.
- Probability calibrators (Isotonic, Platt scaling) must be fit **exclusively on the validation dataset** ($t \in \mathcal{T}_{\text{val}}$) and evaluated on the test dataset ($t \in \mathcal{T}_{\text{test}}$).

## 2. Feature-by-Feature Temporal Audit Table
| Feature Category | Features Included | Aggregation Window | Known at $t$? | Leakage Risk & Prevention |
| :--- | :--- | :--- | :--- | :--- |
| **Lagged Outages** | `outage_fraction_lag_1h`, `lag_2h`, `lag_6h`, `lag_24h`, `lag_48h` | $t-k$ where $k \ge 1$ | Yes | Only prior hourly maximums; no forward filling across prediction horizons. |
| **Rolling Outages** | `outage_roll_mean_6h`, `roll_max_12h`, `roll_max_24h` | $[t-W, t]$ | Yes | Causal rolling windows ending at $t$. |
| **Current Weather** | `wind_speed_ms`, `wind_gust_ms`, `precip_1h`, `temp_c`, `pressure_hpa` | Observation at $t$ | Yes | Observational data valid at or before $t$. |
| **Rolling Weather** | `precip_accum_6h`, `precip_accum_24h`, `max_gust_12h` | $[t-W, t]$ | Yes | Historical causal accumulation ending at $t$. |
| **Weather Deltas** | `pressure_diff_3h`, `temp_diff_6h`, `wind_diff_3h` | $(t) - (t-k)$ | Yes | Difference between observation at $t$ and observation at $t-k$. |
| **Spatial Neighbors** | `neighbor_outage_mean_lag1`, `neighbor_wind_max_lag0` | $[t-1, t]$ over neighbors | Yes | Neighbor graph aggregation uses only contemporaneous or lagged neighbor states. |
| **Calendar/Static** | `hour_sin`, `month_cos`, `log_modeled_customers`, `lat`, `lon` | Fixed at $t$ | Yes | Deterministic astronomical time and invariant geographic coordinates. |

## 3. Forbidden Information Audit
- **Full-event maximums**: Target values $Y_{t+1:t+H}$ must never enter feature vectors.
- **NOAA Storm damage reports / narrative text**: Post-hoc storm event characteristics are strictly reserved for post-evaluation stratification.
- **Global dataset normalization**: Scalers fit on the entire timeseries leak future variance; our pipeline enforces strict fit-transform separation across chronological splits.
