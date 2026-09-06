import pytest
import pandas as pd
from grid_outage_research.data.synthetic import SyntheticDataGenerator
from grid_outage_research.data.panel_builder import PanelDatasetBuilder
from grid_outage_research.geography.county_graph import CountyGraphBuilder
from grid_outage_research.utils.config import load_yaml

@pytest.fixture(scope="session")
def sample_region_config():
    return load_yaml("configs/region.yaml")["primary_region"]

@pytest.fixture(scope="session")
def sample_data_config():
    return load_yaml("configs/data.yaml")

@pytest.fixture(scope="session")
def sample_features_config():
    return load_yaml("configs/features.yaml")

@pytest.fixture(scope="session")
def sample_grid_config():
    return load_yaml("configs/grid_resilience.yaml")

@pytest.fixture(scope="session")
def synthetic_raw_data(sample_region_config):
    generator = SyntheticDataGenerator(seed=42)
    eaglei_df, weather_df, events_df = generator.generate_benchmark_dataset(
        counties=sample_region_config["counties"][:4],
        start_date="2020-01-01",
        end_date="2020-03-31",
        freq="1h"
    )
    return eaglei_df, weather_df, events_df

@pytest.fixture(scope="session")
def synthetic_panel(synthetic_raw_data, sample_data_config, sample_features_config):
    eaglei_df, weather_df, events_df = synthetic_raw_data
    builder = PanelDatasetBuilder(data_config=sample_data_config, features_config=sample_features_config)
    panel = builder.build_panel(eaglei_df, weather_df, events_df)
    return panel

@pytest.fixture(scope="session")
def sample_graph(sample_region_config):
    builder = CountyGraphBuilder(region_config=sample_region_config)
    G = builder.build_graph_from_coordinates(max_distance_km=150.0)
    return G
