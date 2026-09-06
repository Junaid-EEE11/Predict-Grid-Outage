import logging
from typing import List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class WeatherFeatureExtractor:
    """Computes atmospheric rate-of-change, storm severity indicators, and interactions."""

    def __init__(self, diff_hours: List[int] = None):
        self.diff_hours = diff_hours or [1, 3, 6]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute pressure drops, wind surges, and non-linear power-grid fragility interactions."""
        df_out = df.copy().sort_values(["fips", "timestamp"]).reset_index(drop=True)

        # Atmospheric pressure change (rapid drops signify approaching convective / cyclonic systems)
        for d in self.diff_hours:
            if "surface_pressure_hpa" in df_out.columns:
                df_out[f"pressure_diff_{d}h"] = df_out.groupby("fips")["surface_pressure_hpa"].diff(d)
            if "temperature_c" in df_out.columns:
                df_out[f"temp_diff_{d}h"] = df_out.groupby("fips")["temperature_c"].diff(d)
            if "wind_speed_ms" in df_out.columns:
                df_out[f"wind_speed_diff_{d}h"] = df_out.groupby("fips")["wind_speed_ms"].diff(d)

        # Non-linear wind power law proxy: wind force scales with (wind_speed)^2 or (wind_speed)^3
        if "wind_speed_ms" in df_out.columns:
            df_out["wind_speed_sq"] = df_out["wind_speed_ms"] ** 2
            df_out["wind_speed_cube"] = df_out["wind_speed_ms"] ** 3
        if "wind_gust_ms" in df_out.columns:
            df_out["wind_gust_sq"] = df_out["wind_gust_ms"] ** 2

        # Combined wind-rain severity index
        if "wind_speed_ms" in df_out.columns and "precipitation_accum_6h_mm" in df_out.columns:
            df_out["wind_rain_compound_index"] = df_out["wind_speed_ms"] * np.log1p(df_out["precipitation_accum_6h_mm"])

        return df_out
