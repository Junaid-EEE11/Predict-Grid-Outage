import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from grid_outage_research.models.baselines import (
    ClimatologyModel,
    PersistenceModel,
    RegularizedLogisticRegressionModel,
    RidgeRegressionModel,
    LightGBMClassifierModel,
    LightGBMRegressorModel,
    XGBoostClassifierModel
)
from grid_outage_research.evaluation.metrics import MetricsEngine
from grid_outage_research.utils.experiment_tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_baselines")

def main():
    proc_dir = Path("data/processed")
    X_train = np.load(proc_dir / "X_train.npy")
    X_val = np.load(proc_dir / "X_val.npy")
    X_test = np.load(proc_dir / "X_test.npy")

    tracker = ExperimentTracker()
    pred_dir = Path("results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    horizons = [6, 12, 24]
    for H in horizons:
        logger.info(f"--- Training Baselines for Horizon H={H}h ---")
        y_train_cls = np.load(proc_dir / f"y_train_severe_h{H}.npy")
        y_val_cls = np.load(proc_dir / f"y_val_severe_h{H}.npy")
        y_test_cls = np.load(proc_dir / f"y_test_severe_h{H}.npy")

        y_train_reg = np.load(proc_dir / f"y_train_fraction_h{H}.npy")
        y_val_reg = np.load(proc_dir / f"y_val_fraction_h{H}.npy")
        y_test_reg = np.load(proc_dir / f"y_test_fraction_h{H}.npy")

        models = {
            "Climatology": ClimatologyModel(),
            "Persistence": PersistenceModel(recent_outage_col_idx=0),
            "LogisticRegression": RegularizedLogisticRegressionModel(),
            "LightGBM": LightGBMClassifierModel(),
            "XGBoost": XGBoostClassifierModel()
        }

        for name, model in models.items():
            t0 = time.time()
            model.fit(X_train, y_train_cls)
            t_fit = time.time() - t0
            
            p_val = model.predict_proba(X_val)[:, 1]
            p_test = model.predict_proba(X_test)[:, 1]

            metrics_test = MetricsEngine.compute_classification_metrics(y_test_cls, p_test)
            logger.info(f"[{name} H={H}h] PR-AUC={metrics_test['pr_auc']:.4f}, ROC-AUC={metrics_test['roc_auc']:.4f}, Brier={metrics_test['brier_score']:.4f}")

            tracker.log_run(
                experiment_id=f"baseline_{name.lower()}",
                model_name=name,
                task="severe_classification",
                horizon=H,
                hyperparameters={},
                metrics=metrics_test,
                runtime_seconds=t_fit
            )
            np.save(pred_dir / f"{name}_prob_val_h{H}.npy", p_val)
            np.save(pred_dir / f"{name}_prob_test_h{H}.npy", p_test)

        # Train Regressors (LightGBM Regressor with Quantiles)
        lgb_reg = LightGBMRegressorModel()
        lgb_reg.fit(X_train, y_train_reg, fit_quantiles=[0.05, 0.95])
        pred_val_reg = lgb_reg.predict(X_val)
        pred_test_reg = lgb_reg.predict(X_test)
        quant_val = lgb_reg.predict_quantiles(X_val, quantiles=(0.05, 0.95))
        quant_test = lgb_reg.predict_quantiles(X_test, quantiles=(0.05, 0.95))

        m_reg = MetricsEngine.compute_regression_metrics(
            y_test_reg, pred_test_reg, y_low=quant_test[:, 0], y_high=quant_test[:, 1]
        )
        logger.info(f"[LightGBM Regressor H={H}h] MAE={m_reg['mae']:.5f}, RMSE={m_reg['rmse']:.5f}, Coverage={m_reg.get('empirical_coverage', 0):.3f}")
        
        np.save(pred_dir / f"LightGBM_reg_pred_val_h{H}.npy", pred_val_reg)
        np.save(pred_dir / f"LightGBM_reg_pred_test_h{H}.npy", pred_test_reg)
        np.save(pred_dir / f"LightGBM_reg_quant_val_h{H}.npy", quant_val)
        np.save(pred_dir / f"LightGBM_reg_quant_test_h{H}.npy", quant_test)

    logger.info("Baseline training completed successfully.")

if __name__ == "__main__":
    main()
