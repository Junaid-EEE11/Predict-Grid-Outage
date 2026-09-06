import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
import lightgbm as lgb
import xgboost as xgb
from grid_outage_research.models.base import BaseOutageModel

logger = logging.getLogger(__name__)

class ClimatologyModel(BaseOutageModel):
    """Predicts empirical historical base rate from training data."""

    def __init__(self):
        self.base_rate = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClimatologyModel":
        self.base_rate = float(np.mean(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = np.full((len(X), 1), self.base_rate)
        p0 = 1.0 - p1
        return np.hstack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class PersistenceModel(BaseOutageModel):
    """Predicts severe outage if current outage fraction at cutoff >= threshold."""

    def __init__(self, recent_outage_col_idx: int = 0, threshold: float = 0.01):
        self.recent_outage_col_idx = recent_outage_col_idx
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceModel":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        current_val = X[:, self.recent_outage_col_idx]
        p1 = np.clip(current_val / max(self.threshold, 1e-4), 0.0, 1.0)[:, None]
        p0 = 1.0 - p1
        return np.hstack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, self.recent_outage_col_idx] >= self.threshold).astype(int)

class RegularizedLogisticRegressionModel(BaseOutageModel):
    """L2 regularized Logistic Regression GLM baseline."""

    def __init__(self, C: float = 1.0, max_iter: int = 200, random_state: int = 42):
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, class_weight="balanced")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularizedLogisticRegressionModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class RidgeRegressionModel(BaseOutageModel):
    """Ridge linear regression for outage severity."""

    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.model = Ridge(alpha=alpha, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(X), 0.0, None)

class LightGBMClassifierModel(BaseOutageModel):
    """Gradient-boosted decision trees for severe outage classification."""

    def __init__(self, n_estimators: int = 60, learning_rate: float = 0.08, max_depth: int = 6, random_state: int = 42):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=4,
            verbose=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMClassifierModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class LightGBMRegressorModel(BaseOutageModel):
    """LightGBM Regressor with fast multi-quantile regression capability."""

    def __init__(self, n_estimators: int = 60, learning_rate: float = 0.08, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=4,
            verbose=-1
        )
        self.quantile_models = {}

    def fit(self, X: np.ndarray, y: np.ndarray, fit_quantiles: Optional[List[float]] = None) -> "LightGBMRegressorModel":
        self.model.fit(X, y)
        if fit_quantiles:
            # Subsample for blazing fast quantile tree fitting
            sub_idx = np.random.RandomState(self.random_state).choice(len(X), size=min(len(X), 50000), replace=False)
            X_sub, y_sub = X[sub_idx], y[sub_idx]
            for q in fit_quantiles:
                q_model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    n_jobs=4,
                    verbose=-1
                )
                q_model.fit(X_sub, y_sub)
                self.quantile_models[q] = q_model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(X), 0.0, None)

    def predict_quantiles(self, X: np.ndarray, quantiles: Tuple[float, float] = (0.05, 0.95)) -> np.ndarray:
        low_q, high_q = quantiles
        if low_q not in self.quantile_models or high_q not in self.quantile_models:
            raise ValueError(f"Quantiles {quantiles} not fitted. Fitted quantiles: {list(self.quantile_models.keys())}")
        pred_low = np.clip(self.quantile_models[low_q].predict(X), 0.0, None)
        pred_high = np.clip(self.quantile_models[high_q].predict(X), 0.0, None)
        return np.vstack([pred_low, pred_high]).T

class XGBoostClassifierModel(BaseOutageModel):
    """XGBoost baseline for severe outage classification."""

    def __init__(self, n_estimators: int = 60, learning_rate: float = 0.08, max_depth: int = 6, random_state: int = 42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            tree_method="hist",
            n_jobs=4,
            eval_metric="logloss"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifierModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
