import logging
from typing import Dict, Any, List, Optional
import numpy as np
import opendssdirect as dss
from grid_outage_research.grid_resilience.dss_feeder import IEEE123Feeder
from grid_outage_research.grid_resilience.resilience_metrics import ResilienceMetricsCalculator

logger = logging.getLogger(__name__)

class FeederResilienceSimulator:
    """Executes Monte Carlo contingency stress simulations in OpenDSS."""

    def __init__(self, feeder: Optional[IEEE123Feeder] = None):
        self.feeder = feeder or IEEE123Feeder()
        self.feeder.build_case()
        self.base_stats = self.feeder.validate_base_feeder()
        self.base_load_kw = self.base_stats["total_load_kw"]
        self.total_loads_count = self.base_stats["num_loads"]

    def simulate_contingency(self, faulted_lines: List[str]) -> Dict[str, float]:
        """Disconnect faulted lines, solve power flow, and measure unserved load."""
        self.feeder.build_case()

        # Open circuit switches for faulted lines
        for line in faulted_lines:
            dss.Lines.Name(line)
            dss.run_command(f"open line.{line} 1")
            dss.run_command(f"open line.{line} 2")

        dss.run_command("solve")

        if dss.Solution.Converged():
            total_power = dss.Circuit.TotalPower()
            served_kw = float(max(0.0, -total_power[0]))
            
            # Count buses with voltage violations (<0.95 or >1.05 pu)
            v_pu = dss.Circuit.AllBusMagPu()
            v_arr = np.array(v_pu, dtype=float)
            violations = int(np.sum((v_arr < 0.95) | (v_arr > 1.05))) if len(v_arr) > 0 else 0
            
            # Count connected active loads
            served_loads = int(np.round((served_kw / max(1.0, self.base_load_kw)) * self.total_loads_count))
        else:
            # Complete feeder blackout / island collapse
            served_kw = 0.0
            violations = len(dss.Circuit.AllBusMagPu())
            served_loads = 0

        metrics = ResilienceMetricsCalculator.compute_single_contingency_metrics(
            base_load_kw=self.base_load_kw,
            served_load_kw=served_kw,
            total_loads=self.total_loads_count,
            served_loads=served_loads,
            voltage_violations=violations
        )
        return metrics

    def run_monte_carlo_evaluation(self, scenarios: List[List[str]]) -> Dict[str, Any]:
        """Evaluate an ensemble of contingency scenarios and extract resilience distribution."""
        results = []
        for sc in scenarios:
            m = self.simulate_contingency(sc)
            traj = ResilienceMetricsCalculator.compute_restoration_trajectory(m["unserved_load_kw"])
            m["eens_kwh"] = traj["expected_energy_not_served_kwh"]
            m["resilience_loss"] = traj["resilience_loss_index"]
            results.append(m)

        unserved_arr = np.array([r["unserved_load_kw"] for r in results])
        pct_served_arr = np.array([r["percentage_load_served"] for r in results])
        eens_arr = np.array([r["eens_kwh"] for r in results])
        loss_arr = np.array([r["resilience_loss"] for r in results])

        return {
            "n_scenarios": len(scenarios),
            "mean_unserved_kw": float(np.mean(unserved_arr)),
            "median_unserved_kw": float(np.median(unserved_arr)),
            "q95_unserved_kw": float(np.quantile(unserved_arr, 0.95)),
            "mean_pct_served": float(np.mean(pct_served_arr)),
            "mean_eens_kwh": float(np.mean(eens_arr)),
            "q95_eens_kwh": float(np.quantile(eens_arr, 0.95)),
            "mean_resilience_loss": float(np.mean(loss_arr)),
            "raw_results": results
        }
