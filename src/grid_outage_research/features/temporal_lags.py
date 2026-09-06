import logging
from typing import List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class TemporalLagExtractor:
    """Generates strictly causal historical outage lags and rolling statistics."""

    def __init__(self, lag_hours: List[int] = None, rolling_windows: List[int] = None):
        self.lag_hours = lag_hours or [1, 2, 3, 6, 12, 24, 48]
        self.rolling_windows = rolling_windows or [3, 6, 12, 24, 48]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute causal historical lags strictly ending at t or before."""
        df_out = df.copy().sort_values(["fips", "timestamp"]).reset_index(drop=True)

        for lag in self.lag_hours:
            df_out[f"outage_fraction_lag_{lag}h"] = df_out.groupby("fips")["outage_fraction"].shift(lag)
            df_out[f"customers_out_lag_{lag}h"] = df_out.groupby("fips")["customers_out"].shift(lag)

        # Causal rolling windows (shifted by 1 to represent history strictly up to t-1, or unshifted if including t)
        # In EAGLE-I, the snapshot at t is the current reading at prediction cutoff.
        for w in self.rolling_windows:
            df_out[f"outage_roll_mean_{w}h"] = df_out.groupby("fips")["outage_fraction"].transform(
                lambda s: s.rolling(window=w, min_periods=1).mean()
            )
            df_out[f"outage_roll_max_{w}h"] = df_out.groupby("fips")["outage_fraction"].transform(
                lambda s: s.rolling(window=w, min_periods=1).max()
            )
            df_out[f"outage_roll_std_{w}h"] = df_out.groupby("fips")["outage_fraction"].transform(
                lambda s: s.rolling(window=w, min_periods=1).std().fillna(0.0)
            )

        return df_out
