import numpy as np
from typing import Dict, Any, Tuple

class ReliabilityEvaluator:
    """Computes expected calibration error (ECE) and reliability diagram statistics."""

    @staticmethod
    def compute_ece(y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict[str, np.ndarray]]:
        """Compute Expected Calibration Error and bin-level empirical statistics."""
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_lowers = bin_edges[:-1]
        bin_uppers = bin_edges[1:]

        ece = 0.0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        N = len(y_prob)
        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob >= lower) & (y_prob < upper if upper < 1.0 else y_prob <= upper)
            count = np.sum(in_bin)
            bin_counts.append(count)
            if count > 0:
                acc = np.mean(y_true[in_bin])
                conf = np.mean(y_prob[in_bin])
                bin_accuracies.append(acc)
                bin_confidences.append(conf)
                ece += (count / N) * np.abs(acc - conf)
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append((lower + upper) / 2.0)

        diag_data = {
            "bin_lowers": bin_lowers,
            "bin_uppers": bin_uppers,
            "bin_accuracies": np.array(bin_accuracies),
            "bin_confidences": np.array(bin_confidences),
            "bin_counts": np.array(bin_counts)
        }
        return float(ece), diag_data

    @staticmethod
    def compute_brier_score(y_prob: np.ndarray, y_true: np.ndarray) -> float:
        """Mean squared probability error."""
        return float(np.mean((y_prob - y_true) ** 2))
