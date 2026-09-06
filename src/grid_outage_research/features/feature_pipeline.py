import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from grid_outage_research.features.temporal_lags import TemporalLagExtractor
from grid_outage_research.features.weather_features import WeatherFeatureExtractor
from grid_outage_research.geography.spatial_features import SpatialFeatureExtractor
import networkx as nx

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """End-to-end feature engineering pipeline enforcing train-only transformation fitting."""

    def __init__(self, features_config: Dict[str, Any], G: Optional[nx.Graph] = None):
        self.config = features_config
        self.G = G
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.is_fitted = False

    def engineer_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all deterministic lags, rolling stats, weather differentials, and calendar features."""
        df_feat = df.copy()

        # 1. Temporal outage lags & rolling windows
        lag_extractor = TemporalLagExtractor(
            lag_hours=self.config.get("lag_windows_hours", [1, 2, 3, 6, 12, 24, 48]),
            rolling_windows=self.config.get("rolling_windows_hours", [3, 6, 12, 24, 48])
        )
        df_feat = lag_extractor.transform(df_feat)

        # 2. Weather physical features & dynamics
        weather_extractor = WeatherFeatureExtractor(
            diff_hours=self.config.get("weather_diff_hours", [1, 3, 6])
        )
        df_feat = weather_extractor.transform(df_feat)

        # 3. Spatial neighbor lags if graph is provided
        if self.G is not None:
            spatial_extractor = SpatialFeatureExtractor(self.G)
            df_feat = spatial_extractor.compute_spatial_lags(df_feat)

        # 4. Deterministic calendar features
        timestamps = df_feat["timestamp"]
        hour = timestamps.dt.hour.values
        dow = timestamps.dt.dayofweek.values
        month = timestamps.dt.month.values

        df_feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df_feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df_feat["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df_feat["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7.0)
        df_feat["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
        df_feat["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
        df_feat["is_weekend"] = (dow >= 5).astype(float)

        # 5. Static county features
        if "modeled_customers" in df_feat.columns:
            df_feat["log_modeled_customers"] = np.log1p(df_feat["modeled_customers"])
        if "lat" in df_feat.columns:
            df_feat["centroid_lat"] = df_feat["lat"]
        if "lon" in df_feat.columns:
            df_feat["centroid_lon"] = df_feat["lon"]

        return df_feat

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Identify predictive input columns (excluding targets, metadata, and timestamps)."""
        exclude_patterns = [
            "target_", "timestamp", "fips", "county", "state", "begin_time", "end_time",
            "_eval", "is_imputed", "coverage_flag", "customers_out", "total_customers", "modeled_customers",
            "lat", "lon", "name"
        ]
        feature_cols = [
            c for c in df.columns
            if not any(pat in c for pat in exclude_patterns)
            and np.issubdtype(df[c].dtype, np.number)
        ]
        return sorted(feature_cols)

    def fit_transform_train(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Fit scalers strictly on training data and return scaled feature matrix."""
        df_feat = self.engineer_raw_features(train_df)
        self.feature_columns = self.get_feature_names(df_feat)
        
        # Fill short startup NAs caused by initial lags with 0.0 or forward fill
        X_raw = df_feat[self.feature_columns].fillna(0.0).values
        X_scaled = self.scaler.fit_transform(X_raw)
        self.is_fitted = True

        logger.info(f"Feature pipeline fitted on training data with {len(self.feature_columns)} features.")
        return df_feat, X_scaled, self.feature_columns

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Transform validation or test data using scalers fitted strictly on training data."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted on training data before transforming.")
        df_feat = self.engineer_raw_features(df)
        X_raw = df_feat[self.feature_columns].fillna(0.0).values
        X_scaled = self.scaler.transform(X_raw)
        return df_feat, X_scaled
