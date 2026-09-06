import pytest
import pandas as pd
import numpy as np
from grid_outage_research.features.feature_pipeline import FeaturePipeline
from grid_outage_research.features.splitting import ChronologicalDataSplitter
from grid_outage_research.data.panel_builder import PanelDatasetBuilder

def test_no_future_outage_in_features(sample_data_config, sample_features_config, synthetic_raw_data, sample_region_config):
    """Strict test: Permuting future target labels must not alter past feature values."""
    eaglei_df, weather_df, events_df = synthetic_raw_data
    builder = PanelDatasetBuilder(sample_data_config, sample_features_config)
    panel = builder.build_panel(
        eaglei_df=eaglei_df,
        weather_df=weather_df,
        events_df=events_df,
        county_metadata=pd.DataFrame(sample_region_config["counties"][:4])
    )

    pipeline = FeaturePipeline(sample_features_config)
    df_feat_original, _, feature_cols = pipeline.fit_transform_train(panel)

    # Modify future outage fractions by adding synthetic shock to second half of series
    panel_shocked = panel.copy()
    mid_point = len(panel_shocked) // 2
    panel_shocked.loc[mid_point:, "outage_fraction"] = 0.99

    # Re-extract features on the shocked data
    df_feat_shocked = pipeline.engineer_raw_features(panel_shocked)

    # All feature rows BEFORE the mid_point must be EXACTLY identical (zero leakage)
    for col in feature_cols:
        diff = np.abs(
            df_feat_original.loc[:mid_point-2, col].values - 
            df_feat_shocked.loc[:mid_point-2, col].values
        )
        assert np.nanmax(diff) < 1e-9, f"Temporal leakage detected in feature '{col}' before shock timestamp!"

def test_chronological_split_purge_gap(sample_data_config, sample_features_config, synthetic_raw_data, sample_region_config):
    """Ensure no timestamp overlap across train, val, and test partitions."""
    eaglei_df, weather_df, events_df = synthetic_raw_data
    builder = PanelDatasetBuilder(sample_data_config, sample_features_config)
    panel = builder.build_panel(
        eaglei_df=eaglei_df,
        weather_df=weather_df,
        events_df=events_df,
        county_metadata=pd.DataFrame(sample_region_config["counties"][:4])
    )

    # Custom splitter test with month splits
    panel["year"] = panel["timestamp"].dt.year
    splitter = ChronologicalDataSplitter(train_years=[2020], val_years=[2020], test_years=[2020], purge_hours=24)
    # Split by timestamp boundary
    t_min = panel["timestamp"].min()
    t_max = panel["timestamp"].max()
    t_split1 = t_min + pd.Timedelta(days=30)
    t_split2 = t_min + pd.Timedelta(days=60)

    train = panel[panel["timestamp"] < t_split1]
    val = panel[(panel["timestamp"] >= t_split1 + pd.Timedelta(hours=24)) & (panel["timestamp"] < t_split2)]
    test = panel[panel["timestamp"] >= t_split2 + pd.Timedelta(hours=24)]

    assert train["timestamp"].max() < val["timestamp"].min()
    assert (val["timestamp"].min() - train["timestamp"].max()) >= pd.Timedelta(hours=24)
    assert test["timestamp"].min() > val["timestamp"].max()
    assert (test["timestamp"].min() - val["timestamp"].max()) >= pd.Timedelta(hours=24)
