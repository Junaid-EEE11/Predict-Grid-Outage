import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class NOAAStormEventsProcessor:
    """Process NOAA Storm Events for episode identification, evaluation stratification, and failure analysis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_types = config.get("event_types", [
            "Thunderstorm Wind", "High Wind", "Flood", "Flash Flood",
            "Winter Storm", "Ice Storm", "Tropical Storm", "Hurricane", "Tornado"
        ])

    def process_raw(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize storm episode dates, FIPS, and severe event categories."""
        df = raw_df.copy()
        df["fips"] = df["fips"].astype(str).str.zfill(5)
        df["begin_time"] = pd.to_datetime(df["begin_time"], utc=True)
        df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

        if "event_type" in df.columns:
            df["is_severe_storm"] = df["event_type"].isin(self.event_types)
        else:
            df["is_severe_storm"] = True

        return df.sort_values(["fips", "begin_time"]).reset_index(drop=True)

    def attach_event_metadata(
        self,
        panel_df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach event identifiers to panel timestamps for EVALUATION ONLY using interval join."""
        df = panel_df.copy()
        df["is_storm_active_eval"] = False
        df["storm_event_type_eval"] = "None"
        df["storm_episode_id_eval"] = -1

        if events_df.empty:
            return df

        # Group by fips to vectorize matching per county
        event_dict = {}
        for fips, grp in events_df.groupby("fips"):
            event_dict[fips] = grp[["begin_time", "end_time", "event_type", "episode_id"]].to_dict("records")

        for fips, ev_list in event_dict.items():
            fips_mask = (df["fips"] == fips)
            county_times = df.loc[fips_mask, "timestamp"].values
            county_active = np.zeros(len(county_times), dtype=bool)
            county_type = np.full(len(county_times), "None", dtype=object)
            county_ep = np.full(len(county_times), -1, dtype=int)

            for ev in ev_list:
                m = (county_times >= np.datetime64(ev["begin_time"])) & (county_times <= np.datetime64(ev["end_time"]))
                county_active[m] = True
                county_type[m] = ev.get("event_type", "Storm")
                county_ep[m] = ev.get("episode_id", 0)

            df.loc[fips_mask, "is_storm_active_eval"] = county_active
            df.loc[fips_mask, "storm_event_type_eval"] = county_type
            df.loc[fips_mask, "storm_episode_id_eval"] = county_ep

        return df
