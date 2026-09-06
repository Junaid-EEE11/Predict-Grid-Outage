import logging
from pathlib import Path
import numpy as np
import pandas as pd
from grid_outage_research.pipeline.failure_analysis import FailureAnalysisEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_failure_analysis")

def main():
    proc_dir = Path("data/processed")
    pred_dir = Path("results/predictions")
    calib_dir = Path("results/calibrated")
    
    test_df = pd.read_parquet(proc_dir / "test_raw.parquet")
    train_df = pd.read_parquet(proc_dir / "train_raw.parquet")
    
    import json
    with open(proc_dir / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    y_test_cls = np.load(proc_dir / "y_test_severe_h24.npy")
    y_test_reg = np.load(proc_dir / "y_test_fraction_h24.npy")
    lgb_test_iso = np.load(calib_dir / "LightGBM_prob_test_isotonic_h24.npy")
    lgb_test_reg = np.load(pred_dir / "LightGBM_reg_pred_test_h24.npy")

    logger.info("Running Failure and Distribution Shift Analysis...")
    errors = FailureAnalysisEngine.identify_worst_errors(
        test_df=test_df,
        y_true=y_test_cls,
        y_prob=lgb_test_iso,
        y_pred_sev=lgb_test_reg,
        top_n=10
    )

    shifts = FailureAnalysisEngine.evaluate_distribution_shift(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols
    )

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    errors["worst_false_negatives"].to_csv(out_dir / "table07_worst_false_negatives.csv", index=False)
    errors["worst_false_positives"].to_csv(out_dir / "table08_worst_false_positives.csv", index=False)
    shifts.to_csv(out_dir / "distribution_shifts_train_vs_test.csv", index=False)
    logger.info("Failure analysis exported successfully.")

if __name__ == "__main__":
    main()
