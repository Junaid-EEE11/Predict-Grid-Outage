from typing import Tuple
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)

# Apply publication-quality style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

class PublicationPlotter:
    """Renders publication-quality Figures 1 to 15."""

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_reliability_diagrams(self, uncal_diag: Dict[str, np.ndarray], cal_diag: Dict[str, np.ndarray], fig_name: str = "fig04_reliability_diagram.png"):
        """Figure 4: Uncalibrated vs Calibrated Reliability Curves."""
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(uncal_diag["bin_confidences"], uncal_diag["bin_accuracies"], "s-", color="crimson", label="Uncalibrated LightGBM")
        ax.plot(cal_diag["bin_confidences"], cal_diag["bin_accuracies"], "o-", color="royalblue", label="Isotonic Calibrated")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Empirical Outage Fraction")
        ax.set_title("Probability Calibration (Task A: 24h Horizon)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(self.output_dir / fig_name)
        plt.close(fig)

    def plot_pr_roc_curves(self, models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], fig_name: str = "fig02_pr_roc_curves.png"):
        """Figure 2: Precision-Recall and ROC Discrimination Curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        for name, (y_true, y_prob) in models_dict.items():
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax1.plot(recall, precision, label=name)
            ax2.plot(fpr, tpr, label=name)

        ax1.set_xlabel("Recall")
        ax1.set_ylabel("Precision")
        ax1.set_title("Precision-Recall Curve (Severe Outages)")
        ax1.legend(loc="lower left")

        ax2.plot([0, 1], [0, 1], "k--", label="Random Chance")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("Receiver Operating Characteristic (ROC)")
        ax2.legend(loc="lower right")

        fig.tight_layout()
        fig.savefig(self.output_dir / fig_name)
        plt.close(fig)

    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray, top_k: int = 15, fig_name: str = "fig06_feature_importance.png"):
        """Figure 6: Feature Importances."""
        idx = np.argsort(importances)[::-1][:top_k]
        top_names = [feature_names[i] for i in idx][::-1]
        top_scores = importances[idx][::-1]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_names)), top_scores, color="teal", alpha=0.85)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel("Importance Metric (Gain / SHAP Magnitude)")
        ax.set_title(f"Top {top_k} Predictive Outage Risk Features")
        fig.tight_layout()
        fig.savefig(self.output_dir / fig_name)
        plt.close(fig)

    def plot_event_timeline(self, df_event: pd.DataFrame, fig_name: str = "fig08_event_timeline.png"):
        """Figure 8: Case Study Event Timeline (Weather, Outages, Predictions, Intervals)."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Top panel: Weather dynamics
        ax1.plot(df_event["timestamp"], df_event["wind_gust_ms"], color="darkorange", label="Wind Gust (m/s)")
        ax1.plot(df_event["timestamp"], df_event["wind_speed_ms"], color="forestgreen", label="Sustained Wind (m/s)")
        ax1.set_ylabel("Wind Speed [m/s]")
        ax1.set_title("Major Storm Case Study: Temporal Dynamics and Uncertainty Bounds")
        ax1.legend(loc="upper left")

        # Bottom panel: Observed Outages and Prediction Bounds
        ax2.plot(df_event["timestamp"], df_event["outage_fraction"] * 100, "k-", lw=2, label="Observed Outage %")
        if "y_pred_sev" in df_event.columns:
            ax2.plot(df_event["timestamp"], df_event["y_pred_sev"] * 100, "b--", label="Point Forecast %")
        if "interval_low" in df_event.columns and "interval_high" in df_event.columns:
            ax2.fill_between(
                df_event["timestamp"],
                df_event["interval_low"] * 100,
                df_event["interval_high"] * 100,
                color="royalblue", alpha=0.25, label="90% Conformal Interval"
            )
        ax2.set_ylabel("Customers Out [%]")
        ax2.set_xlabel("Time (UTC)")
        ax2.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(self.output_dir / fig_name)
        plt.close(fig)

    def plot_resilience_curves(self, time_hours: np.ndarray, base_traj: np.ndarray, hard_traj: np.ndarray, fig_name: str = "fig11_resilience_curves.png"):
        """Figure 11: Time-dependent Power Restoration Trajectory & Resilience Loss."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(time_hours, base_traj, "r-", lw=2, label="Baseline Grid Response")
        ax.fill_between(time_hours, 0, base_traj, color="salmon", alpha=0.3, label="Baseline Resilience Loss")
        ax.plot(time_hours, hard_traj, "b--", lw=2, label="Top-5 Hardened Grid Response")
        ax.fill_between(time_hours, 0, hard_traj, color="lightblue", alpha=0.3, label="Hardened Resilience Loss")
        
        ax.set_xlabel("Time Post-Contingency [hours]")
        ax.set_ylabel("Unserved Load [kW]")
        ax.set_title("Feeder Restoration Trajectory and Resilience Loss Triangle (IEEE 123-Bus)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(self.output_dir / fig_name)
        plt.close(fig)

    def plot_uncertainty_propagation(self, risk_levels: List[str], eens_distributions: Dict[str, List[float]], fig_name: str = "fig10_resilience_uncertainty.png"):
        """Figure 10: Downstream Feeder Resilience Distributions across Risk Forecast Tiers."""
        fig, ax = plt.subplots(figsize=(8, 5))
        data_to_plot = [eens_distributions[lvl] for lvl in risk_levels]
        sns.boxplot(data=data_to_plot, ax=ax, palette="Blues")
        ax.set_xticklabels(risk_levels)
        ax.set_xlabel("Calibrated Outage Risk Scenario Tier")
        ax.set_ylabel("Expected Energy Not Served [kWh]")
        ax.set_title("Downstream Distribution Grid Resilience Impact (IEEE 123-Bus)")
        fig.tight_layout()
        fig.savefig(self.output_dir / fig_name)
        plt.close(fig)
