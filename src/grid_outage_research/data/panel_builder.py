import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from grid_outage_research.data.eaglei import EagleIProcessor
from grid_outage_research.data.weather import WeatherProcessor
from grid_outage_research.data.noaa_storm_events import NOAAStormEventsProcessor
from grid_outage_research.data.validator import DataValidator

logger = logging.getLogger(__name__)

class PanelDatasetBuilder:
    """Integrates outage records, weather grids, and county metadata into a verified hourly panel."""

    def __init__(self, data_config: Dict[str, Any], features_config: Dict[str, Any]):
        self.data_config = data_config
        self.features_config = features_config
        self.horizons = features_config.get("horizons", [6, 12, 24])
        self.severe_threshold = features_config.get("default_severe_threshold", 0.01)

    def build_panel(
        self,
        eaglei_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        events_df: Optional[pd.DataFrame] = None,
        county_metadata: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Merge sources, calculate causal targets for horizons H, and validate data integrity."""
        logger.info("Aligning EAGLE-I outage and weather data...")
        
        # Ensure timestamp formats
        eaglei_df["timestamp"] = pd.to_datetime(eaglei_df["timestamp"], utc=True)
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True)
        eaglei_df["fips"] = eaglei_df["fips"].astype(str).str.zfill(5)
        weather_df["fips"] = weather_df["fips"].astype(str).str.zfill(5)

        # Merge outage and weather on (fips, timestamp)
        merged = pd.merge(eaglei_df, weather_df, on=["fips", "timestamp"], how="inner")
        merged = merged.sort_values(["fips", "timestamp"]).reset_index(drop=True)

        # Attach county metadata if present
        if county_metadata is not None and "fips" in county_metadata.columns:
            county_metadata["fips"] = county_metadata["fips"].astype(str).str.zfill(5)
            merged = pd.merge(merged, county_metadata, on="fips", how="left")

        # Attach NOAA event evaluation metadata if provided
        if events_df is not None and not events_df.empty:
            noaa_proc = NOAAStormEventsProcessor(self.data_config.get("noaa_events", {}))
            merged = noaa_proc.attach_event_metadata(merged, events_df)

        # Vectorized target calculation per county
        logger.info(f"Computing forward targets for horizons {self.horizons}h...")
        for H in self.horizons:
            # Shift by 1 forward to ensure target is strictly future (t+1 .. t+H)
            merged[f"target_max_fraction_h{H}"] = (
                merged.groupby("fips")["outage_fraction"]
                .transform(lambda s: s.iloc[::-1].rolling(window=H, min_periods=1).max().iloc[::-1].shift(-1))
            )
            merged[f"target_severe_h{H}"] = (
                merged[f"target_max_fraction_h{H}"] >= self.severe_threshold
            ).astype(int)

            merged[f"target_max_log_customers_h{H}"] = (
                merged.groupby("fips")["customers_out"]
                .transform(lambda s: np.log1p(s.iloc[::-1].rolling(window=H, min_periods=1).max().iloc[::-1].shift(-1)))
            )

        max_h = max(self.horizons)
        valid_mask = merged.groupby("fips")[f"target_max_fraction_h{max_h}"].transform(lambda s: s.notnull())
        panel = merged[valid_mask].copy().reset_index(drop=True)

        # Validate final panel schema
        report = DataValidator.validate_panel_schema(
            panel,
            required_cols=["fips", "timestamp", "outage_fraction", "wind_speed_ms", "precipitation_1h_mm"]
        )
        logger.info(f"Panel built successfully: {report['total_rows']} rows across {report['unique_counties']} counties.")
        return panel
