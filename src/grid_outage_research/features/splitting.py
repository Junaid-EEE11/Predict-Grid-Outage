import logging
from typing import Dict, Any, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class ChronologicalDataSplitter:
    """Strict chronological train/val/test data partitioner with boundary purge buffers."""

    def __init__(
        self,
        train_years: List[int] = None,
        val_years: List[int] = None,
        test_years: List[int] = None,
        purge_hours: int = 24
    ):
        self.train_years = train_years or [2017, 2018, 2019, 2020]
        self.val_years = val_years or [2021]
        self.test_years = test_years or [2022]
        self.purge_hours = purge_hours

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Partition dataset by year while purging overlap at boundaries."""
        df = df.copy()
        df["year"] = df["timestamp"].dt.year

        train_raw = df[df["year"].isin(self.train_years)].copy()
        val_raw = df[df["year"].isin(self.val_years)].copy()
        test_raw = df[df["year"].isin(self.test_years)].copy()

        # Purge boundary records (remove initial purge_hours from val and test to prevent target overlap)
        if not val_raw.empty and self.purge_hours > 0:
            val_start = val_raw["timestamp"].min() + pd.Timedelta(hours=self.purge_hours)
            val_df = val_raw[val_raw["timestamp"] >= val_start].copy()
        else:
            val_df = val_raw

        if not test_raw.empty and self.purge_hours > 0:
            test_start = test_raw["timestamp"].min() + pd.Timedelta(hours=self.purge_hours)
            test_df = test_raw[test_raw["timestamp"] >= test_start].copy()
        else:
            test_df = test_raw

        train_df = train_raw.drop(columns=["year"], errors="ignore")
        val_df = val_df.drop(columns=["year"], errors="ignore")
        test_df = test_df.drop(columns=["year"], errors="ignore")

        logger.info(
            f"Chronological split completed: Train={len(train_df)} ({self.train_years}), "
            f"Val={len(val_df)} ({self.val_years}), Test={len(test_df)} ({self.test_years})"
        )
        return train_df, val_df, test_df
