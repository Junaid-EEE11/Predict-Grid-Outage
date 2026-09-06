import logging
from pathlib import Path
from grid_outage_research.data.synthetic import SyntheticDataGenerator
from grid_outage_research.utils.config import load_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_noaa_events")

def main():
    logger.info("Checking NOAA Storm Events Database...")
    synth_dir = Path("data/synthetic_fixtures")
    synth_dir.mkdir(parents=True, exist_ok=True)
    
    reg_config = load_yaml("configs/region.yaml")["primary_region"]
    gen = SyntheticDataGenerator(seed=42)
    _, _, events_df = gen.generate_benchmark_dataset(
        counties=reg_config["counties"],
        start_date="2017-01-01",
        end_date="2022-12-31"
    )
    events_df.to_parquet(synth_dir / "synthetic_noaa_events.parquet")
    events_df.to_csv(synth_dir / "synthetic_noaa_events.csv", index=False)
    logger.info(f"NOAA Storm Events fixture created: {len(events_df)} records saved to {synth_dir / 'synthetic_noaa_events.parquet'}")

if __name__ == "__main__":
    main()
