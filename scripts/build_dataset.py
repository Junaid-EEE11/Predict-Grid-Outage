import logging
from pathlib import Path
import pandas as pd
from grid_outage_research.data.synthetic import SyntheticDataGenerator
from grid_outage_research.data.panel_builder import PanelDatasetBuilder
from grid_outage_research.geography.county_graph import CountyGraphBuilder
from grid_outage_research.features.splitting import ChronologicalDataSplitter
from grid_outage_research.features.feature_pipeline import FeaturePipeline
from grid_outage_research.utils.config import load_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_dataset")

def main():
    data_config = load_yaml("configs/data.yaml")
    reg_config = load_yaml("configs/region.yaml")["primary_region"]
    feat_config = load_yaml("configs/features.yaml")

    # 1. Generate or load base data
    synth_dir = Path("data/synthetic_fixtures")
    if not (synth_dir / "synthetic_eaglei.parquet").exists():
        logger.info("Generating base synthetic fixtures...")
        gen = SyntheticDataGenerator(seed=42)
        eaglei_df, weather_df, events_df = gen.generate_benchmark_dataset(
            counties=reg_config["counties"],
            start_date="2017-01-01",
            end_date="2022-12-31"
        )
        synth_dir.mkdir(parents=True, exist_ok=True)
        eaglei_df.to_parquet(synth_dir / "synthetic_eaglei.parquet")
        weather_df.to_parquet(synth_dir / "synthetic_weather.parquet")
        events_df.to_parquet(synth_dir / "synthetic_noaa_events.parquet")
    else:
        logger.info("Loading existing synthetic fixtures...")
        eaglei_df = pd.read_parquet(synth_dir / "synthetic_eaglei.parquet")
        weather_df = pd.read_parquet(synth_dir / "synthetic_weather.parquet")
        events_df = pd.read_parquet(synth_dir / "synthetic_noaa_events.parquet")

    # 2. Build County Spatial Graph
    graph_builder = CountyGraphBuilder(reg_config)
    G = graph_builder.build_graph_from_coordinates()
    A_norm = graph_builder.get_adjacency_matrix(G, normalize=True)
    
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.save(interim_dir / "adjacency_normalized.npy", A_norm)

    # 3. Build Panel Dataset
    builder = PanelDatasetBuilder(data_config, feat_config)
    panel_df = builder.build_panel(
        eaglei_df=eaglei_df,
        weather_df=weather_df,
        events_df=events_df,
        county_metadata=pd.DataFrame(reg_config["counties"])
    )

    # 4. Chronological Splitting
    splitter = ChronologicalDataSplitter(
        train_years=data_config["splitting"]["train_years"],
        val_years=data_config["splitting"]["val_years"],
        test_years=data_config["splitting"]["test_years"],
        purge_hours=data_config["splitting"]["purge_hours_between_splits"]
    )
    train_df, val_df, test_df = splitter.split(panel_df)

    # 5. Feature Engineering Pipeline
    pipeline = FeaturePipeline(feat_config, G=G)
    train_feat, X_train, feature_cols = pipeline.fit_transform_train(train_df)
    val_feat, X_val = pipeline.transform(val_df)
    test_feat, X_test = pipeline.transform(test_df)

    # 6. Save Processed Datasets
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    panel_df.to_parquet(proc_dir / "panel_full.parquet")
    train_df.to_parquet(proc_dir / "train_raw.parquet")
    val_df.to_parquet(proc_dir / "val_raw.parquet")
    test_df.to_parquet(proc_dir / "test_raw.parquet")

    np.save(proc_dir / "X_train.npy", X_train)
    np.save(proc_dir / "X_val.npy", X_val)
    np.save(proc_dir / "X_test.npy", X_test)

    # Save target vectors for each horizon
    for H in feat_config["horizons"]:
        np.save(proc_dir / f"y_train_severe_h{H}.npy", train_df[f"target_severe_h{H}"].values)
        np.save(proc_dir / f"y_val_severe_h{H}.npy", val_df[f"target_severe_h{H}"].values)
        np.save(proc_dir / f"y_test_severe_h{H}.npy", test_df[f"target_severe_h{H}"].values)

        np.save(proc_dir / f"y_train_fraction_h{H}.npy", train_df[f"target_max_fraction_h{H}"].values)
        np.save(proc_dir / f"y_val_fraction_h{H}.npy", val_df[f"target_max_fraction_h{H}"].values)
        np.save(proc_dir / f"y_test_fraction_h{H}.npy", test_df[f"target_max_fraction_h{H}"].values)

    import json
    with open(proc_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    logger.info(f"Dataset build complete. {len(feature_cols)} features processed and saved to {proc_dir}")

if __name__ == "__main__":
    main()
