import logging
from typing import Dict, Any, List
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SpatialFeatureExtractor:
    """Extracts graph neighbor summary statistics strictly without temporal lookahead."""

    def __init__(self, G: nx.Graph):
        self.G = G

    def compute_spatial_lags(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized neighbor lag computations using matrix operations."""
        if len(panel_df) == 0:
            return panel_df.copy()
        df = panel_df.copy()
        
        # Pivot by timestamp x fips for fast matrix operations
        outage_pivot = df.pivot(index="timestamp", columns="fips", values="outage_fraction").sort_index()
        wind_pivot = df.pivot(index="timestamp", columns="fips", values="wind_speed_ms").sort_index()
        precip_pivot = df.pivot(index="timestamp", columns="fips", values="precipitation_1h_mm").sort_index()

        out_frames = []
        for fips in outage_pivot.columns:
            neighbors = [n for n in self.G.neighbors(fips) if n in outage_pivot.columns] if fips in self.G else []
            if not neighbors:
                neighbors = [fips]

            # Neighbor lagged outage (t-1)
            n_out_mean_lag1 = outage_pivot[neighbors].shift(1).mean(axis=1)
            n_out_max_lag1 = outage_pivot[neighbors].shift(1).max(axis=1)

            # Neighbor contemporaneous weather (t)
            n_wind_max = wind_pivot[neighbors].max(axis=1)
            n_precip_max = precip_pivot[neighbors].max(axis=1)

            fips_df = pd.DataFrame({
                "timestamp": outage_pivot.index,
                "fips": fips,
                "neighbor_outage_mean_lag1": n_out_mean_lag1.values,
                "neighbor_outage_max_lag1": n_out_max_lag1.values,
                "neighbor_wind_max_t": n_wind_max.values,
                "neighbor_precip_max_t": n_precip_max.values
            })
            out_frames.append(fips_df)

        spatial_all = pd.concat(out_frames, ignore_index=True)
        merged = pd.merge(df, spatial_all, on=["fips", "timestamp"], how="left")
        return merged
