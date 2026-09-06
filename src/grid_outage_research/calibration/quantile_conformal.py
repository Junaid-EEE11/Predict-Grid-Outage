import numpy as np
from typing import Tuple, Optional

class SplitConformalRegressor:
    """Distribution-free Split Conformal Prediction for valid finite-sample prediction intervals."""

    def __init__(self, alpha: float = 0.10):
        """alpha: error rate (e.g. alpha=0.10 for 90% coverage interval)."""
        self.alpha = alpha
        self.q_hat = 0.0
        self.is_calibrated = False

    def calibrate(self, y_val_pred_low: np.ndarray, y_val_pred_high: np.ndarray, y_val_true: np.ndarray) -> "SplitConformalRegressor":
        """Compute nonconformity scores on held-out validation dataset."""
        # Nonconformity score s_i = max(pred_low - y, y - pred_high)
        scores = np.maximum(y_val_pred_low - y_val_true, y_val_true - y_val_pred_high)
        n = len(scores)
        # Empirical quantile at level ceil((n+1)*(1-alpha))/n
        q_level = np.clip(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0)
        self.q_hat = float(np.quantile(scores, q_level, method="higher"))
        self.is_calibrated = True
        return self

    def predict_interval(self, y_pred_low: np.ndarray, y_pred_high: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Adjust test prediction bounds by calibrated conformal correction."""
        if not self.is_calibrated:
            raise RuntimeError("SplitConformalRegressor must be calibrated on validation set first.")
        c_low = np.clip(y_pred_low - self.q_hat, 0.0, 1.0)
        c_high = np.clip(y_pred_high + self.q_hat, 0.0, 1.0)
        return c_low, c_high

    @staticmethod
    def evaluate_coverage(y_low: np.ndarray, y_high: np.ndarray, y_true: np.ndarray) -> dict:
        """Compute empirical coverage, mean width, and Winkler score."""
        covered = (y_true >= y_low) & (y_true <= y_high)
        empirical_coverage = float(np.mean(covered))
        widths = y_high - y_low
        mean_width = float(np.mean(widths))
        return {
            "empirical_coverage": empirical_coverage,
            "mean_interval_width": mean_width,
            "coverage_gap": empirical_coverage - (1.0 - 0.10)
        }
