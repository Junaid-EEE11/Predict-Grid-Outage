import logging
from pathlib import Path
import numpy as np
import pandas as pd
from grid_outage_research.models.baselines import LightGBMClassifierModel
from grid_outage_research.evaluation.metrics import MetricsEngine
from grid_outage_research.utils.experiment_tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_ablations")

def main():
    proc_dir = Path("data/processed")
    X_train = np.load(proc_dir / "X_train.npy")
    X_test = np.load(proc_dir / "X_test.npy")
    
    import json
    with open(proc_dir / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    y_train = np.load(proc_dir / "y_train_severe_h24.npy")
    y_test = np.load(proc_dir / "y_test_severe_h24.npy")

    tracker = ExperimentTracker()

    # Identify feature subsets
    weather_idx = [i for i, c in enumerate(feature_cols) if any(k in c for k in ["wind", "precip", "temp", "pressure", "dew"])]
    outage_idx = [i for i, c in enumerate(feature_cols) if "outage" in c]
    weather_outage_idx = sorted(list(set(weather_idx + outage_idx)))
    all_idx = list(range(len(feature_cols)))

    ablations = [
        ("A1_weather_only", weather_idx, "Weather Only"),
        ("A2_lagged_outages_only", outage_idx, "Lagged Outages Only"),
        ("A3_weather_and_outage", weather_outage_idx, "Weather + Lagged Outages"),
        ("A4_full_features", all_idx, "Weather + Outage + Calendar + Static (Full)")
    ]

    ablation_results = []
    for ab_id, indices, desc in ablations:
        logger.info(f"Running Ablation: {desc} ({len(indices)} features)...")
        X_tr = X_train[:, indices]
        X_te = X_test[:, indices]

        model = LightGBMClassifierModel()
        model.fit(X_tr, y_train)
        p_test = model.predict_proba(X_te)[:, 1]

        m = MetricsEngine.compute_classification_metrics(y_test, p_test)
        logger.info(f"[{ab_id}] PR-AUC={m['pr_auc']:.4f}, ROC-AUC={m['roc_auc']:.4f}, Brier={m['brier_score']:.4f}")
        
        ablation_results.append({
            "ablation_id": ab_id,
            "description": desc,
            "num_features": len(indices),
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "brier_score": m["brier_score"],
            "ece": m["ece"]
        })

    df_ab = pd.DataFrame(ablation_results)
    res_dir = Path("results/tables")
    res_dir.mkdir(parents=True, exist_ok=True)
    df_ab.to_csv(res_dir / "table04_feature_ablation.csv", index=False)
    logger.info("Ablation suite completed successfully.")

if __name__ == "__main__":
    main()
