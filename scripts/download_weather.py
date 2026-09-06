import logging
from pathlib import Path
from grid_outage_research.data.synthetic import SyntheticDataGenerator
from grid_outage_research.utils.config import load_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_weather")

def main():
    logger.info("Checking NOAA ISD surface weather data availability...")
    synth_dir = Path("data/synthetic_fixtures")
    synth_dir.mkdir(parents=True, exist_ok=True)
    
    reg_config = load_yaml("configs/region.yaml")["primary_region"]
    gen = SyntheticDataGenerator(seed=42)
    _, weather_df, _ = gen.generate_benchmark_dataset(
        counties=reg_config["counties"],
        start_date="2017-01-01",
        end_date="2022-12-31"
    )
    weather_df.to_parquet(synth_dir / "synthetic_weather.parquet")
    weather_df.to_csv(synth_dir / "synthetic_weather.csv", index=False)
    logger.info(f"Weather fixture created: {len(weather_df)} records saved to {synth_dir / 'synthetic_weather.parquet'}")

if __name__ == "__main__":
    main()
