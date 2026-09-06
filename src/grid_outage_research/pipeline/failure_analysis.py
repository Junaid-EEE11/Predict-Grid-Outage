from typing import Optional, Dict, Any, List
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class FailureAnalysisEngine:
    """Investigates extreme prediction errors, severe storm underpredictions, and distribution shifts."""

    @staticmethod
    def identify_worst_errors(
        test_df: pd.DataFrame,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred_sev: Optional[np.ndarray] = None,
        top_n: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """Identify top false negatives (missed severe outages) and false positives (false alarms)."""
        df = test_df.copy().reset_index(drop=True)
        df["y_true"] = y_true
        df["y_prob"] = y_prob
        df["error"] = np.abs(df["y_true"] - df["y_prob"])
        
        # False negatives: true severe outage (y_true=1) with lowest predicted probabilities
        fn_df = df[df["y_true"] == 1].sort_values("y_prob", ascending=True).head(top_n)
        
        # False positives: non-severe (y_true=0) with highest predicted probabilities
        fp_df = df[df["y_true"] == 0].sort_values("y_prob", ascending=False).head(top_n)

        # Severely underpredicted regression events
        if y_pred_sev is not None:
            df["y_pred_sev"] = y_pred_sev
            df["underpred_magnitude"] = df["outage_fraction"] - df["y_pred_sev"]
            underpred_df = df.sort_values("underpred_magnitude", ascending=False).head(top_n)
        else:
            underpred_df = pd.DataFrame()

        return {
            "worst_false_negatives": fn_df,
            "worst_false_positives": fp_df,
            "worst_underpredicted_events": underpred_df
        }

    @staticmethod
    def evaluate_distribution_shift(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """Compute Kolmogorov-Smirnov test and Wasserstein shift across chronological partitions."""
        from scipy.stats import ks_2samp, wasserstein_distance
        shifts = []
        for col in feature_cols:
            if col in train_df.columns and col in test_df.columns:
                v_tr = train_df[col].dropna().values
                v_te = test_df[col].dropna().values
                if len(v_tr) > 0 and len(v_te) > 0:
                    ks_stat, p_val = ks_2samp(v_tr, v_te)
                    w_dist = wasserstein_distance(v_tr, v_te)
                    shifts.append({
                        "feature": col,
                        "ks_stat": round(float(ks_stat), 4),
                        "ks_p_value": float(p_val),
                        "wasserstein_distance": round(float(w_dist), 4),
                        "train_mean": round(float(np.mean(v_tr)), 4),
                        "test_mean": round(float(np.mean(v_te)), 4)
                    })
        return pd.DataFrame(shifts).sort_values("ks_stat", ascending=False)
