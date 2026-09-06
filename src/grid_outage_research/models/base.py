from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

class BaseOutageModel(ABC):
    """Abstract base interface for all classification, regression, and uncertainty models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseOutageModel":
        """Fit model on training feature matrix X and target labels y."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions."""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probabilities [P(0), P(1)]."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_proba.")

    def predict_quantiles(self, X: np.ndarray, quantiles: Tuple[float, ...]) -> np.ndarray:
        """Generate conditional quantile predictions."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_quantiles.")
