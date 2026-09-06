import logging
import time
from pathlib import Path
import numpy as np
from grid_outage_research.models.temporal_nn import TemporalNNModel
from grid_outage_research.evaluation.metrics import MetricsEngine
from grid_outage_research.utils.experiment_tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_temporal")

def main():
    proc_dir = Path("data/processed")
    X_train = np.load(proc_dir / "X_train.npy")
    X_val = np.load(proc_dir / "X_val.npy")
    X_test = np.load(proc_dir / "X_test.npy")

    pred_dir = Path("results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker()

    H = 24
    logger.info(f"Training Temporal GRU Model for Horizon H={H}h...")
    y_train_cls = np.load(proc_dir / f"y_train_severe_h{H}.npy")
    y_val_cls = np.load(proc_dir / f"y_val_severe_h{H}.npy")
    y_test_cls = np.load(proc_dir / f"y_test_severe_h{H}.npy")

    t0 = time.time()
    temp_model = TemporalNNModel(
        input_dim=X_train.shape[1],
        seq_len=12,
        hidden_dim=32,
        lr=0.005,
        epochs=5,
        batch_size=512
    )
    temp_model.fit(X_train, y_train_cls)
    runtime = time.time() - t0

    p_val = temp_model.predict_proba(X_val)[:, 1]
    p_test = temp_model.predict_proba(X_test)[:, 1]

    metrics = MetricsEngine.compute_classification_metrics(y_test_cls, p_test)
    logger.info(f"[Temporal GRU H={H}h] PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, Brier={metrics['brier_score']:.4f}")

    tracker.log_run(
        experiment_id="temporal_gru",
        model_name="TemporalGRU",
        task="severe_classification",
        horizon=H,
        hyperparameters={"seq_len": 12, "hidden_dim": 32, "epochs": 5},
        metrics=metrics,
        runtime_seconds=runtime
    )
    np.save(pred_dir / f"TemporalGRU_prob_val_h{H}.npy", p_val)
    np.save(pred_dir / f"TemporalGRU_prob_test_h{H}.npy", p_test)
    logger.info("Temporal training completed successfully.")

if __name__ == "__main__":
    main()
