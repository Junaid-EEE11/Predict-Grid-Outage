import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from grid_outage_research.models.graph_nn import SpatiotemporalGNNModel
from grid_outage_research.evaluation.metrics import MetricsEngine
from grid_outage_research.utils.experiment_tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_graph")

def main():
    proc_dir = Path("data/processed")
    interim_dir = Path("data/interim")
    A_norm = np.load(interim_dir / "adjacency_normalized.npy")
    N = A_norm.shape[0]

    # Reshape features to (T, N, D)
    X_train = np.load(proc_dir / "X_train.npy")
    X_val = np.load(proc_dir / "X_val.npy")
    X_test = np.load(proc_dir / "X_test.npy")

    T_tr = len(X_train) // N
    T_val = len(X_val) // N
    T_te = len(X_test) // N
    D = X_train.shape[1]

    X_train_3d = X_train[:T_tr * N].reshape(T_tr, N, D)
    X_val_3d = X_val[:T_val * N].reshape(T_val, N, D)
    X_test_3d = X_test[:T_te * N].reshape(T_te, N, D)

    H = 24
    logger.info(f"Training Spatiotemporal GNN Model for Horizon H={H}h...")
    y_train_cls = np.load(proc_dir / f"y_train_severe_h{H}.npy")[:T_tr * N].reshape(T_tr, N)
    y_val_cls = np.load(proc_dir / f"y_val_severe_h{H}.npy")[:T_val * N].reshape(T_val, N)
    y_test_cls = np.load(proc_dir / f"y_test_severe_h{H}.npy")[:T_te * N].reshape(T_te, N)

    tracker = ExperimentTracker()
    pred_dir = Path("results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spatiotemporal GNN (With Graph Message Passing)
    t0 = time.time()
    gnn_model = SpatiotemporalGNNModel(
        num_nodes=N,
        input_dim=D,
        A_norm=A_norm,
        seq_len=24,
        hidden_dim=32,
        use_graph_passing=True,
        epochs=6,
        batch_size=64
    )
    gnn_model.fit(X_train_3d, y_train_cls)
    runtime = time.time() - t0

    p_val = gnn_model.predict_proba(X_val_3d)[:, 1]
    p_test = gnn_model.predict_proba(X_test_3d)[:, 1]

    metrics = MetricsEngine.compute_classification_metrics(y_test_cls.flatten(), p_test)
    logger.info(f"[Spatiotemporal GNN H={H}h] PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, Brier={metrics['brier_score']:.4f}")

    tracker.log_run(
        experiment_id="spatiotemporal_gnn",
        model_name="SpatiotemporalGNN",
        task="severe_classification",
        horizon=H,
        hyperparameters={"seq_len": 24, "hidden_dim": 32, "epochs": 6, "use_graph": True},
        metrics=metrics,
        runtime_seconds=runtime
    )
    np.save(pred_dir / f"SpatiotemporalGNN_prob_val_h{H}.npy", p_val)
    np.save(pred_dir / f"SpatiotemporalGNN_prob_test_h{H}.npy", p_test)

    # 2. Ablation A5: Same Temporal Architecture WITHOUT Graph Message Passing
    gnn_no_graph = SpatiotemporalGNNModel(
        num_nodes=N,
        input_dim=D,
        A_norm=A_norm,
        seq_len=24,
        hidden_dim=32,
        use_graph_passing=False,
        epochs=6,
        batch_size=64
    )
    gnn_no_graph.fit(X_train_3d, y_train_cls)
    p_test_nograph = gnn_no_graph.predict_proba(X_test_3d)[:, 1]
    metrics_nograph = MetricsEngine.compute_classification_metrics(y_test_cls.flatten(), p_test_nograph)
    logger.info(f"[Ablation A5: No-Graph Model H={H}h] PR-AUC={metrics_nograph['pr_auc']:.4f}")

    np.save(pred_dir / f"NoGraph_prob_test_h{H}.npy", p_test_nograph)
    logger.info("Graph model training completed successfully.")

if __name__ == "__main__":
    main()
