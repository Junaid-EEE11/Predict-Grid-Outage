import pytest
import pandas as pd
import numpy as np
from grid_outage_research.data.validator import DataValidator
from grid_outage_research.data.panel_builder import PanelDatasetBuilder

def test_validator_detects_bounds_violation():
    bad_df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=5, freq="1h"),
        "fips": ["37183"] * 5,
        "outage_fraction": [0.0, 0.05, 1.25, -0.1, 0.02], # contains >1 and <0
        "wind_speed_ms": [5.0, 10.0, -2.0, 150.0, 8.0],
        "precipitation_1h_mm": [0.0, 1.0, 0.0, 2.0, 0.0]
    })
    report = DataValidator.validate_panel_schema(bad_df, required_cols=["fips", "timestamp", "outage_fraction"])
    assert "outage_fraction" in report["bounds_violations"]
    assert report["bounds_violations"]["outage_fraction"]["negative_count"] == 1
    assert report["bounds_violations"]["outage_fraction"]["greater_than_one_count"] == 1

def test_panel_builder(sample_data_config, sample_features_config, synthetic_raw_data, sample_region_config):
    eaglei_df, weather_df, events_df = synthetic_raw_data
    builder = PanelDatasetBuilder(sample_data_config, sample_features_config)
    panel = builder.build_panel(
        eaglei_df=eaglei_df,
        weather_df=weather_df,
        events_df=events_df,
        county_metadata=pd.DataFrame(sample_region_config["counties"][:4])
    )
    assert not panel.empty
    assert "target_severe_h6" in panel.columns
    assert "target_severe_h12" in panel.columns
    assert "target_severe_h24" in panel.columns
    assert "target_max_fraction_h6" in panel.columns
    assert panel["outage_fraction"].max() <= 1.0
    assert panel["outage_fraction"].min() >= 0.0
