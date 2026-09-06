import logging
from pathlib import Path
import numpy as np
from grid_outage_research.calibration.calibrator import ProbabilityCalibrator
from grid_outage_research.calibration.quantile_conformal import SplitConformalRegressor
from grid_outage_research.evaluation.metrics import MetricsEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calibrate_models")

def main():
    proc_dir = Path("data/processed")
    pred_dir = Path("results/predictions")
    calib_dir = Path("results/calibrated")
    calib_dir.mkdir(parents=True, exist_ok=True)

    H = 24
    logger.info(f"--- Running Post-Hoc Calibration on Validation Data (H={H}h) ---")
    y_val_cls = np.load(proc_dir / f"y_val_severe_h{H}.npy")
    y_test_cls = np.load(proc_dir / f"y_test_severe_h{H}.npy")
    y_val_reg = np.load(proc_dir / f"y_val_fraction_h{H}.npy")
    y_test_reg = np.load(proc_dir / f"y_test_fraction_h{H}.npy")

    # 1. Classification Probability Calibration (Isotonic & Platt)
    lgb_val = np.load(pred_dir / f"LightGBM_prob_val_h{H}.npy")
    lgb_test = np.load(pred_dir / f"LightGBM_prob_test_h{H}.npy")

    # Uncalibrated baseline metrics
    m_uncal = MetricsEngine.compute_classification_metrics(y_test_cls, lgb_test)
    logger.info(f"[Uncalibrated LightGBM] Brier={m_uncal['brier_score']:.5f}, ECE={m_uncal['ece']:.5f}")

    # Isotonic Calibrator
    cal_iso = ProbabilityCalibrator(method="isotonic").fit(lgb_val, y_val_cls)
    lgb_test_iso = cal_iso.transform(lgb_test)
    m_iso = MetricsEngine.compute_classification_metrics(y_test_cls, lgb_test_iso)
    logger.info(f"[Isotonic Calibrated LightGBM] Brier={m_iso['brier_score']:.5f}, ECE={m_iso['ece']:.5f}")

    # Platt Scaling
    cal_platt = ProbabilityCalibrator(method="platt").fit(lgb_val, y_val_cls)
    lgb_test_platt = cal_platt.transform(lgb_test)
    m_platt = MetricsEngine.compute_classification_metrics(y_test_cls, lgb_test_platt)
    logger.info(f"[Platt Calibrated LightGBM] Brier={m_platt['brier_score']:.5f}, ECE={m_platt['ece']:.5f}")

    np.save(calib_dir / f"LightGBM_prob_test_isotonic_h{H}.npy", lgb_test_iso)
    np.save(calib_dir / f"LightGBM_prob_test_platt_h{H}.npy", lgb_test_platt)

    # 2. Regression Conformal Calibration
    quant_val = np.load(pred_dir / f"LightGBM_reg_quant_val_h{H}.npy")
    quant_test = np.load(pred_dir / f"LightGBM_reg_quant_test_h{H}.npy")
    
    conformal = SplitConformalRegressor(alpha=0.10)
    conformal.calibrate(quant_val[:, 0], quant_val[:, 1], y_val_reg)
    c_low, c_high = conformal.predict_interval(quant_test[:, 0], quant_test[:, 1])

    cov_eval = SplitConformalRegressor.evaluate_coverage(c_low, c_high, y_test_reg)
    logger.info(f"[Conformalized Prediction Intervals (Nominal 90%)] Empirical Coverage={cov_eval['empirical_coverage']*100:.2f}%, Mean Width={cov_eval['mean_interval_width']:.4f}")

    np.save(calib_dir / f"conformal_interval_low_h{H}.npy", c_low)
    np.save(calib_dir / f"conformal_interval_high_h{H}.npy", c_high)

if __name__ == "__main__":
    main()
