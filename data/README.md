# Data Directory

This directory stores the raw, interim, processed datasets and synthetic fixtures used in the research project.

## Directory Structure
- `raw/`: Unmodified downloaded raw data files (EAGLE-I outage records, NOAA weather ISD/hourly data, NOAA Storm Events database, US Census county shapefiles).
- `interim/`: Intermediate aligned hourly files, cleaned weather matrices, county contiguity graphs.
- `processed/`: Final panel dataset (`panel_features.parquet` / `panel_features.csv`), train/val/test splits, and preprocessed arrays.
- `synthetic_fixtures/`: Explicitly labeled synthetic fixtures mirroring the real data schema used for automated testing, CI smoke tests, and offline reproducibility.

## Data Provenance & Integrity Rules
1. Raw data must never be edited manually.
2. Missingness in EAGLE-I records must be handled according to `docs/data_dictionary.md` and `docs/modeling_decisions.md` (never silently treated as zero without coverage flags).
3. All predictive features strictly obey the prediction cutoff timestamp $t$ (no lookahead or leakage).
