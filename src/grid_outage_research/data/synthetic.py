import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generates reproducible, physics-inspired synthetic fixtures for CI, smoke tests, and offline execution."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_benchmark_dataset(
        self,
        counties: List[Dict[str, Any]],
        start_date: str = "2017-01-01",
        end_date: str = "2022-12-31",
        freq: str = "1h"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate synthetic EAGLE-I, NOAA weather, and NOAA Storm Events matching exact production schemas."""
        timestamps = pd.date_range(start=start_date, end=end_date, freq=freq, tz=timezone.utc)
        n_times = len(timestamps)
        n_counties = len(counties)
        
        logger.info(f"Generating synthetic panel: {n_counties} counties, {n_times} timestamps ({start_date} to {end_date})")

        records_eaglei = []
        records_weather = []
        events_list = []

        # Generate realistic periodic climate patterns + episodic severe storm systems
        # 1. Base annual and diurnal temperature cycles
        day_of_year = timestamps.dayofyear.values
        hour_of_day = timestamps.hour.values

        # Base seasonal temp in Celsius (mean ~15C, summer ~28C, winter ~2C)
        base_temp = 15.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25) + 4.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)

        # Base pressure
        base_pressure = 1013.25 + 5.0 * np.cos(2 * np.pi * (day_of_year) / 365.25)

        # Introduce recurring severe weather episodes (tropical storms in Aug-Oct, winter storms in Jan-Feb, convective storms in May-Jul)
        # Episode generator
        np.random.seed(self.seed)
        n_episodes = int((n_times / (24 * 365)) * 12)  # ~12 storm episodes per year
        episode_starts = np.sort(np.random.choice(np.arange(100, n_times - 100), size=n_episodes, replace=False))
        
        storm_intensity = np.zeros(n_times)
        storm_precip = np.zeros(n_times)
        storm_type_map = {}

        for ep_idx, ep_start in enumerate(episode_starts):
            ep_dur = np.random.randint(12, 48)  # 12-48 hour storm
            ep_end = min(ep_start + ep_dur, n_times - 1)
            intensity = np.random.uniform(18.0, 42.0)  # wind speed peak m/s
            rain_rate = np.random.uniform(5.0, 35.0)   # mm/h
            
            # Half-sine storm envelope
            t_span = np.arange(ep_start, ep_end)
            if len(t_span) > 0:
                envelope = np.sin(np.pi * (t_span - ep_start) / len(t_span))
                storm_intensity[t_span] = np.maximum(storm_intensity[t_span], intensity * envelope)
                storm_precip[t_span] = np.maximum(storm_precip[t_span], rain_rate * envelope)
            
            # Store metadata
            ep_type = np.random.choice(["Tropical Storm", "Thunderstorm Wind", "Winter Storm", "High Wind"])
            storm_type_map[ep_idx] = {
                "episode_id": 1000 + ep_idx,
                "event_type": ep_type,
                "begin_time": timestamps[ep_start],
                "end_time": timestamps[ep_end]
            }

        # Build records for each county
        for c_idx, c in enumerate(counties):
            fips = c["fips"]
            tot_cust = c.get("modeled_customers", 50000)
            lat = c.get("lat", 35.0)
            lon = c.get("lon", -79.0)

            # Spatial phase shift for storm arrival
            spatial_lag = int((lon + 80.0) * 4)  # 0-8 hour storm progression

            # Generate county weather
            c_noise_wind = self.rng.normal(0, 1.2, n_times)
            c_noise_temp = self.rng.normal(0, 1.5, n_times)
            c_noise_press = self.rng.normal(0, 2.0, n_times)

            # Shifted storm profile
            c_storm_wind = np.roll(storm_intensity, spatial_lag)
            c_storm_precip = np.roll(storm_precip, spatial_lag)

            wind_speed = np.maximum(1.5, 4.0 + c_noise_wind + c_storm_wind)
            wind_gust = wind_speed * self.rng.uniform(1.2, 1.65, n_times)
            precip = np.maximum(0.0, np.where(self.rng.random(n_times) > 0.85, self.rng.exponential(1.5, n_times), 0.0) + c_storm_precip)
            temp = base_temp + c_noise_temp - (lat - 35.0) * 1.5
            dew_point = temp - self.rng.uniform(1.0, 7.0, n_times)
            pressure = base_pressure + c_noise_press - 0.4 * c_storm_wind

            df_w = pd.DataFrame({
                "timestamp": timestamps,
                "fips": fips,
                "temperature_c": np.round(temp, 2),
                "dew_point_c": np.round(dew_point, 2),
                "wind_speed_ms": np.round(wind_speed, 2),
                "wind_gust_ms": np.round(wind_gust, 2),
                "surface_pressure_hpa": np.round(pressure, 2),
                "precipitation_1h_mm": np.round(precip, 2)
            })
            records_weather.append(df_w)

            # Physics-inspired fragility & outage generation:
            # Outage rate follows sigmoid threshold of wind gust and accumulated precipitation
            # Normal background outage: ~0.02% of customers
            base_outage_rate = self.rng.uniform(0.0001, 0.0004, n_times)
            # Fragility component
            wind_fragility = 1.0 / (1.0 + np.exp(-(wind_gust - 22.0) / 3.5))
            precip_fragility = 1.0 / (1.0 + np.exp(-(precip - 15.0) / 4.0))
            
            storm_outage_fraction = 0.45 * (wind_fragility ** 1.8) + 0.15 * (precip_fragility ** 1.2)
            combined_outage_fraction = np.clip(base_outage_rate + storm_outage_fraction, 0.0, 1.0)
            
            # Smooth outage evolution with realistic restoration decay
            outage_series = np.zeros(n_times)
            curr_val = combined_outage_fraction[0]
            for t in range(n_times):
                target_val = combined_outage_fraction[t]
                if target_val > curr_val:
                    # Rapid failure onset
                    curr_val = 0.7 * curr_val + 0.3 * target_val
                else:
                    # Exponential restoration decay
                    curr_val = 0.94 * curr_val + 0.06 * target_val
                outage_series[t] = curr_val

            customers_out = np.round(outage_series * tot_cust).astype(int)

            df_o = pd.DataFrame({
                "timestamp": timestamps,
                "fips": fips,
                "customers_out": customers_out,
                "total_customers": tot_cust,
                "outage_fraction": np.round(outage_series, 6),
                "is_imputed": False,
                "coverage_flag": 1.0
            })
            records_eaglei.append(df_o)

            # Record storm events for county
            for ep_idx, ep_meta in storm_type_map.items():
                if np.max(c_storm_wind) > 15.0:
                    events_list.append({
                        "episode_id": ep_meta["episode_id"],
                        "event_id": ep_meta["episode_id"] * 100 + c_idx,
                        "fips": fips,
                        "event_type": ep_meta["event_type"],
                        "begin_time": ep_meta["begin_time"],
                        "end_time": ep_meta["end_time"]
                    })

        all_weather = pd.concat(records_weather, ignore_index=True)
        all_eaglei = pd.concat(records_eaglei, ignore_index=True)
        all_events = pd.DataFrame(events_list).drop_duplicates()

        return all_eaglei, all_weather, all_events
