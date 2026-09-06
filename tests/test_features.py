import numpy as np
import pandas as pd
import pytest
from grid_outage_research.features.temporal_lags import TemporalLagExtractor
from grid_outage_research.features.weather_features import WeatherFeatureExtractor
from grid_outage_research.features.feature_pipeline import FeaturePipeline

def test_temporal_lags(synthetic_panel):
    extractor = TemporalLagExtractor(lag_hours=[1, 2, 6], rolling_windows=[6, 24])
    df_lags = extractor.transform(synthetic_panel)
    
    assert "outage_fraction_lag_1h" in df_lags.columns
    assert "outage_roll_max_6h" in df_lags.columns
    assert "outage_roll_max_24h" in df_lags.columns
    
    # Test strict causal shift: lag1 at t must equal t-1 value
    county_df = df_lags[df_lags["fips"] == "37001"].sort_values("timestamp").reset_index(drop=True)
    assert county_df["outage_fraction_lag_1h"].iloc[1] == pytest.approx(county_df["outage_fraction"].iloc[0])

def test_weather_features(synthetic_panel):
    extractor = WeatherFeatureExtractor()
    df_w = extractor.transform(synthetic_panel)
    
    assert "temp_diff_3h" in df_w.columns
    assert "wind_speed_sq" in df_w.columns
    assert "pressure_diff_3h" in df_w.columns

def test_feature_pipeline_fit_transform(synthetic_panel, sample_graph, sample_features_config):
    train_mask = synthetic_panel["timestamp"] < "2020-03-01"
    train_df = synthetic_panel[train_mask].copy()
    test_df = synthetic_panel[~train_mask].copy()

    pipeline = FeaturePipeline(sample_features_config, sample_graph)
    _, X_train, cols = pipeline.fit_transform_train(train_df)
    _, X_test = pipeline.transform(test_df)

    assert X_train.shape[1] == len(cols)
    assert X_test.shape[1] == len(cols)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
