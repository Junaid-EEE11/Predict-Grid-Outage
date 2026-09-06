from typing import Optional, Dict
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error
)
from grid_outage_research.calibration.reliability import ReliabilityEvaluator

class MetricsEngine:
    """Comprehensive metric computation for classification and regression tasks."""

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        operating_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute discrimination and calibration metrics."""
        y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        y_pred = (y_prob >= operating_threshold).astype(int)

        # PR-AUC / Average Precision
        pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
        # ROC-AUC
        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        # Brier Score
        brier = float(brier_score_loss(y_true, y_prob))
        # Log Loss (Negative Log-Likelihood)
        nll = float(log_loss(y_true, y_prob))
        # ECE
        ece, _ = ReliabilityEvaluator.compute_ece(y_prob, y_true, n_bins=10)

        # Operating point metrics
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        prevalence = float(np.mean(y_true))

        return {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "ece": ece,
            "nll": nll,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "prevalence": prevalence
        }

    @staticmethod
    def compute_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_low: Optional[np.ndarray] = None,
        y_high: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute point and interval regression error metrics."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        res = {
            "mae": mae,
            "rmse": rmse
        }
        if y_low is not None and y_high is not None:
            covered = (y_true >= y_low) & (y_true <= y_high)
            res["empirical_coverage"] = float(np.mean(covered))
            res["mean_interval_width"] = float(np.mean(y_high - y_low))
        return res
