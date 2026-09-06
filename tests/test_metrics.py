import numpy as np
import pytest
from grid_outage_research.evaluation.metrics import MetricsEngine

def test_metrics_classification():
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.15, 0.85, 0.90, 0.05, 0.75, 0.3])
    
    m = MetricsEngine.compute_classification_metrics(y_true, y_prob)
    assert "pr_auc" in m
    assert "roc_auc" in m
    assert "brier_score" in m
    assert "ece" in m
    assert m["pr_auc"] > 0.7
    assert m["roc_auc"] > 0.8

def test_metrics_regression():
    y_true = np.array([0.01, 0.05, 0.02, 0.10])
    y_pred = np.array([0.012, 0.048, 0.022, 0.095])
    y_low = y_pred - 0.01
    y_high = y_pred + 0.01

    m = MetricsEngine.compute_regression_metrics(y_true, y_pred, y_low, y_high)
    assert m["mae"] < 0.01
    assert m["empirical_coverage"] == 1.0
