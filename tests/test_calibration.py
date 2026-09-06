import numpy as np
import pytest
from grid_outage_research.calibration.calibrator import ProbabilityCalibrator
from grid_outage_research.calibration.reliability import ReliabilityEvaluator
from grid_outage_research.calibration.quantile_conformal import SplitConformalRegressor

def test_probability_calibrator():
    np.random.seed(42)
    y_true = np.array([0]*80 + [1]*20)
    y_prob = np.clip(y_true * 0.7 + np.random.uniform(0, 0.3, size=100), 0.01, 0.99)

    cal_iso = ProbabilityCalibrator(method="isotonic").fit(y_prob, y_true)
    cal_probs = cal_iso.transform(y_prob)

    assert cal_probs.shape == y_prob.shape
    assert (cal_probs >= 0).all() and (cal_probs <= 1).all()

    ece_uncal, _ = ReliabilityEvaluator.compute_ece(y_prob, y_true)
    ece_cal, _ = ReliabilityEvaluator.compute_ece(cal_probs, y_true)
    assert ece_cal <= ece_uncal + 1e-4

def test_split_conformal_regression():
    np.random.seed(42)
    y_val = np.random.uniform(0.01, 0.10, size=200)
    q_low = y_val - 0.02
    q_high = y_val + 0.02

    conformal = SplitConformalRegressor(alpha=0.10)
    conformal.calibrate(q_low, q_high, y_val)

    y_test = np.random.uniform(0.01, 0.10, size=100)
    t_low = y_test - 0.02
    t_high = y_test + 0.02
    c_low, c_high = conformal.predict_interval(t_low, t_high)

    res = SplitConformalRegressor.evaluate_coverage(c_low, c_high, y_test)
    assert res["empirical_coverage"] >= 0.85
