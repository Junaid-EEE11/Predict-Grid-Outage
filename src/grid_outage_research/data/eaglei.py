import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class EagleIProcessor:
    """Process raw 15-minute EAGLE-I outage records to hourly aligned county time-series."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.time_col = config.get("timestamp_col", "run_start_time")
        self.fips_col = config.get("fips_col", "fips_code")
        self.cust_out_col = config.get("customers_out_col", "sum")
        self.tot_cust_col = config.get("total_customers_col", "total_customers")
        self.max_gap_hours = config.get("max_short_gap_hours", 3)

    def process_raw(
        self,
        raw_df: pd.DataFrame,
        county_metadata: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Clean, resample to hourly maximum, and compute outage fractions."""
        df = raw_df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True)
        df[self.fips_col] = df[self.fips_col].astype(str).str.zfill(5)

        # Standardize customer counts
        df[self.cust_out_col] = pd.to_numeric(df[self.cust_out_col], errors="coerce").fillna(0).clip(lower=0)

        # Merge modeled total customer denominators if available
        if county_metadata is not None and "fips" in county_metadata.columns and "modeled_customers" in county_metadata.columns:
            county_metadata["fips"] = county_metadata["fips"].astype(str).str.zfill(5)
            df = df.merge(
                county_metadata[["fips", "modeled_customers"]],
                left_on=self.fips_col,
                right_on="fips",
                how="left"
            )
            df[self.tot_cust_col] = df["modeled_customers"].fillna(10000)
        elif self.tot_cust_col not in df.columns:
            df[self.tot_cust_col] = 50000  # Default fallback denominator

        # Hourly aggregation (maximum customers out during the hour)
        df["hourly_timestamp"] = df[self.time_col].dt.floor("1h")
        grouped = df.groupby([self.fips_col, "hourly_timestamp"]).agg(
            customers_out=(self.cust_out_col, "max"),
            total_customers=(self.tot_cust_col, "first")
        ).reset_index()

        grouped.rename(columns={
            self.fips_col: "fips",
            "hourly_timestamp": "timestamp"
        }, inplace=True)

        # Compute outage fraction
        grouped["outage_fraction"] = (
            grouped["customers_out"] / grouped["total_customers"].clip(lower=1)
        ).clip(0.0, 1.0)

        # Add explicit missingness / imputation flag
        grouped["is_imputed"] = False
        grouped["coverage_flag"] = 1.0

        return grouped.sort_values(["fips", "timestamp"]).reset_index(drop=True)
