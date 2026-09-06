import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

class ExperimentTracker:
    """Reproducible experiment registry saving run metadata, parameters, and metrics."""
    def __init__(self, output_dir: str | Path = "results/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.output_dir / "experiment_registry.csv"

    @staticmethod
    def get_git_commit() -> str:
        """Retrieve the current git commit hash if available."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True
            )
            return res.stdout.strip()
        except Exception:
            return "unknown_git_rev"

    def log_run(
        self,
        experiment_id: str,
        model_name: str,
        task: str,
        horizon: int,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        runtime_seconds: float,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Record a completed experimental trial to JSON and CSV registry."""
        run_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "git_commit": self.get_git_commit(),
            "model_name": model_name,
            "task": task,
            "horizon_hours": horizon,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "runtime_seconds": round(runtime_seconds, 3),
            "notes": notes
        }
        # Save individual JSON artifact
        run_file = self.output_dir / f"{experiment_id}_{model_name}_h{horizon}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2)

        # Flatten for tabular registry
        flat_record = {
            "timestamp": run_record["timestamp"],
            "experiment_id": experiment_id,
            "git_commit": run_record["git_commit"],
            "model_name": model_name,
            "task": task,
            "horizon_hours": horizon,
            "runtime_seconds": run_record["runtime_seconds"],
            "notes": notes
        }
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                flat_record[f"metric_{k}"] = v

        df_row = pd.DataFrame([flat_record])
        if self.registry_file.exists():
            df_existing = pd.read_csv(self.registry_file)
            df_all = pd.concat([df_existing, df_row], ignore_index=True)
            df_all.to_csv(self.registry_file, index=False)
        else:
            df_row.to_csv(self.registry_file, index=False)

        return run_record
