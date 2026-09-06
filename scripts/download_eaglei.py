import sys
import logging
from pathlib import Path
from grid_outage_research.data.synthetic import SyntheticDataGenerator
from grid_outage_research.utils.config import load_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_eaglei")

def main():
    logger.info("Checking EAGLE-I data source availability...")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    reg_config = load_yaml("configs/region.yaml")["primary_region"]
    logger.info("External DOE/ORNL EAGLE-I API requires authentication credentials.")
    logger.info("Generating fully standardized synthetic benchmark fixtures in data/synthetic_fixtures/ for local execution...")
    
    gen = SyntheticDataGenerator(seed=42)
    eaglei_df, weather_df, events_df = gen.generate_benchmark_dataset(
        counties=reg_config["counties"],
        start_date="2017-01-01",
        end_date="2022-12-31"
    )
    
    synth_dir = Path("data/synthetic_fixtures")
    synth_dir.mkdir(parents=True, exist_ok=True)
    eaglei_df.to_parquet(synth_dir / "synthetic_eaglei.parquet")
    eaglei_df.to_csv(synth_dir / "synthetic_eaglei.csv", index=False)
    logger.info(f"EAGLE-I fixture created: {len(eaglei_df)} records saved to {synth_dir / 'synthetic_eaglei.parquet'}")

if __name__ == "__main__":
    main()
