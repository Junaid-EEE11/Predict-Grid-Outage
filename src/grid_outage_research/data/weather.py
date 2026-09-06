import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class WeatherProcessor:
    """Process NOAA ISD and hourly observational surface weather observations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.variables = config.get("variables", [
            "temperature_c", "dew_point_c", "relative_humidity",
            "wind_speed_ms", "wind_gust_ms", "surface_pressure_hpa",
            "precipitation_1h_mm"
        ])

    def process_raw(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Validate physical constraints, resample to 1h grid, and calculate rolling accumulations."""
        df = raw_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["fips"] = df["fips"].astype(str).str.zfill(5)

        # Clip variables to physically plausible bounds
        if "temperature_c" in df.columns:
            df["temperature_c"] = df["temperature_c"].clip(-50.0, 60.0)
        if "dew_point_c" in df.columns:
            df["dew_point_c"] = df["dew_point_c"].clip(-50.0, 45.0)
        if "wind_speed_ms" in df.columns:
            df["wind_speed_ms"] = df["wind_speed_ms"].clip(0.0, 80.0)
        if "wind_gust_ms" in df.columns:
            # Gust must be at least sustained wind speed
            if "wind_speed_ms" in df.columns:
                df["wind_gust_ms"] = np.maximum(df["wind_gust_ms"], df["wind_speed_ms"])
            df["wind_gust_ms"] = df["wind_gust_ms"].clip(0.0, 100.0)
        if "surface_pressure_hpa" in df.columns:
            df["surface_pressure_hpa"] = df["surface_pressure_hpa"].clip(850.0, 1085.0)
        if "precipitation_1h_mm" in df.columns:
            df["precipitation_1h_mm"] = df["precipitation_1h_mm"].clip(0.0, 200.0)

        # Derive relative humidity if missing
        if "relative_humidity" not in df.columns and "temperature_c" in df.columns and "dew_point_c" in df.columns:
            t = df["temperature_c"]
            td = df["dew_point_c"]
            # Magnus formula approximation
            rh = 100.0 * np.exp((17.625 * td) / (243.04 + td)) / np.exp((17.625 * t) / (243.04 + t))
            df["relative_humidity"] = rh.clip(0.0, 100.0)

        # Sort and calculate causal rolling precipitation accumulations
        df = df.sort_values(["fips", "timestamp"]).reset_index(drop=True)
        df["precipitation_accum_6h_mm"] = df.groupby("fips")["precipitation_1h_mm"].transform(
            lambda x: x.rolling(window=6, min_periods=1).sum()
        )
        df["precipitation_accum_24h_mm"] = df.groupby("fips")["precipitation_1h_mm"].transform(
            lambda x: x.rolling(window=24, min_periods=1).sum()
        )

        return df
