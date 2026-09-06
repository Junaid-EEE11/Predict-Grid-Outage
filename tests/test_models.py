import numpy as np
import pytest
from grid_outage_research.models.baselines import (
    ClimatologyModel,
    PersistenceModel,
    RegularizedLogisticRegressionModel,
    RidgeRegressionModel,
    LightGBMClassifierModel
)

def test_climatology_model():
    y_tr = np.array([0, 0, 1, 0, 0])
    model = ClimatologyModel().fit(np.zeros((5, 2)), y_tr)
    probs = model.predict_proba(np.zeros((3, 2)))
    assert probs.shape == (3, 2)
    assert probs[0, 1] == pytest.approx(0.20)

def test_persistence_model():
    X = np.array([[0.05], [0.0], [0.02]])
    model = PersistenceModel(recent_outage_col_idx=0, threshold=0.01)
    preds = model.predict(X)
    assert np.array_equal(preds, np.array([1, 0, 1]))

def test_lightgbm_classifier():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    model = LightGBMClassifierModel(n_estimators=10)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100, 2)
    assert (probs >= 0).all() and (probs <= 1).all()
