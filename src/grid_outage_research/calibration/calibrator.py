import logging
from typing import Literal, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    """Probability calibration strictly fit on validation set predictions."""

    def __init__(self, method: Literal["isotonic", "platt", "temperature"] = "isotonic"):
        self.method = method
        self.calibrator = None
        self.temp = 1.0

    def fit(self, y_val_prob: np.ndarray, y_val_true: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibration map using validation probabilities and true binary labels."""
        y_val_prob = np.clip(y_val_prob, 1e-6, 1.0 - 1e-6)
        
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(y_val_prob, y_val_true)
        elif self.method == "platt":
            # Platt scaling via univariate logistic regression on log-odds
            log_odds = np.log(y_val_prob / (1.0 - y_val_prob)).reshape(-1, 1)
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(log_odds, y_val_true)
        elif self.method == "temperature":
            # Optimize single temperature parameter on cross-entropy
            from scipy.optimize import minimize_scalar
            def nll(t):
                if t <= 0.01: return 1e9
                scaled_probs = 1.0 / (1.0 + np.exp(-np.log(y_val_prob / (1.0 - y_val_prob)) / t))
                scaled_probs = np.clip(scaled_probs, 1e-6, 1.0 - 1e-6)
                return -np.mean(y_val_true * np.log(scaled_probs) + (1.0 - y_val_true) * np.log(1.0 - scaled_probs))
            res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
            self.temp = float(res.x)

        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply fitted calibration map to test probabilities."""
        y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.calibrator.predict(y_prob), 0.0, 1.0)
        elif self.method == "platt":
            log_odds = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
            return self.calibrator.predict_proba(log_odds)[:, 1]
        elif self.method == "temperature":
            log_odds = np.log(y_prob / (1.0 - y_prob))
            return 1.0 / (1.0 + np.exp(-log_odds / self.temp))
        return y_prob
