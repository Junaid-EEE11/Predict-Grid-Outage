import logging
from typing import Callable, Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EventAwareBootstrap:
    """Block and event-aware paired bootstrap for rigorous statistical hypothesis testing."""

    def __init__(self, n_resamples: int = 1000, block_size_hours: int = 24, seed: int = 42):
        self.n_resamples = n_resamples
        self.block_size = block_size_hours
        self.rng = np.random.default_rng(seed)

    def paired_bootstrap_confidence_interval(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Compute bootstrap confidence intervals for model A, model B, and their paired difference (A - B)."""
        N = len(y_true)
        n_blocks = max(1, N // self.block_size)
        block_indices = [np.arange(i * self.block_size, min((i + 1) * self.block_size, N)) for i in range(n_blocks)]

        scores_a = []
        scores_b = []
        diffs = []

        for _ in range(self.n_resamples):
            sampled_block_idx = self.rng.choice(n_blocks, size=n_blocks, replace=True)
            sampled_indices = np.concatenate([block_indices[b] for b in sampled_block_idx])
            
            y_t_sample = y_true[sampled_indices]
            if len(np.unique(y_t_sample)) < 2:
                continue

            score_a = metric_fn(y_t_sample, y_pred_a[sampled_indices])
            score_b = metric_fn(y_t_sample, y_pred_b[sampled_indices])
            scores_a.append(score_a)
            scores_b.append(score_b)
            diffs.append(score_a - score_b)

        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        diffs = np.array(diffs)

        ci_a = (float(np.quantile(scores_a, alpha / 2)), float(np.quantile(scores_a, 1.0 - alpha / 2)))
        ci_b = (float(np.quantile(scores_b, alpha / 2)), float(np.quantile(scores_b, 1.0 - alpha / 2)))
        ci_diff = (float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1.0 - alpha / 2)))

        # Two-sided empirical p-value for hypothesis H0: diff == 0
        p_val = float(2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))

        return {
            "mean_a": float(np.mean(scores_a)),
            "ci_a": ci_a,
            "mean_b": float(np.mean(scores_b)),
            "ci_b": ci_b,
            "mean_diff": float(np.mean(diffs)),
            "ci_diff": ci_diff,
            "p_value": p_val,
            "is_significant_05": p_val < 0.05
        }
