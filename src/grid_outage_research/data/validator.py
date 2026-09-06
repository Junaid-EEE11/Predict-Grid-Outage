import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DataValidator:
    """Automated schema and data-integrity verification for spatiotemporal grid panels."""

    @staticmethod
    def validate_panel_schema(df: pd.DataFrame, required_cols: List[str]) -> Dict[str, Any]:
        """Verify required columns, detect missingness, and check physical boundaries."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Panel schema violation: Missing required columns: {missing_cols}")

        report = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "date_min": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
            "date_max": str(df["timestamp"].max()) if "timestamp" in df.columns else None,
            "unique_counties": int(df["fips"].nunique()) if "fips" in df.columns else 0,
            "null_counts": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated(subset=["fips", "timestamp"]).sum()) if ("fips" in df.columns and "timestamp" in df.columns) else 0,
            "bounds_violations": {}
        }

        # Check outage fraction boundaries [0.0, 1.0]
        if "outage_fraction" in df.columns:
            neg_out = (df["outage_fraction"] < 0).sum()
            over_out = (df["outage_fraction"] > 1.0).sum()
            if neg_out > 0 or over_out > 0:
                report["bounds_violations"]["outage_fraction"] = {
                    "negative_count": int(neg_out),
                    "greater_than_one_count": int(over_out)
                }
                logger.warning(f"Outage fraction bounds violation: {neg_out} negative, {over_out} > 1.0")

        # Check physical weather bounds
        if "wind_speed_ms" in df.columns:
            neg_wind = (df["wind_speed_ms"] < 0).sum()
            extreme_wind = (df["wind_speed_ms"] > 100).sum()
            if neg_wind > 0 or extreme_wind > 0:
                report["bounds_violations"]["wind_speed_ms"] = {
                    "negative_count": int(neg_wind),
                    "extreme_count": int(extreme_wind)
                }

        return report

    @staticmethod
    def check_temporal_continuity(df: pd.DataFrame, expected_freq: str = "1h") -> pd.DataFrame:
        """Detect missing timestamp gaps per county FIPS."""
        gaps = []
        for fips, group in df.groupby("fips"):
            group = group.sort_values("timestamp")
            time_diffs = group["timestamp"].diff()
            expected_delta = pd.Timedelta(expected_freq)
            gap_rows = group[time_diffs > expected_delta]
            if not gap_rows.empty:
                gaps.append({
                    "fips": fips,
                    "gap_count": len(gap_rows),
                    "max_gap": str(time_diffs.max())
                })
        return pd.DataFrame(gaps)
