import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import opendssdirect as dss
from grid_outage_research.grid_resilience.simulator import FeederResilienceSimulator
from grid_outage_research.grid_resilience.contingency_generator import MonteCarloContingencyGenerator

logger = logging.getLogger(__name__)

class GridHardeningDecisionEngine:
    """Compares baseline feeder risk against top-k targeted line hardening strategies."""

    def __init__(self, simulator: FeederResilienceSimulator, config: Dict[str, Any]):
        self.sim = simulator
        self.config = config
        self.all_lines = dss.Lines.AllNames()

    def identify_critical_lines(self) -> List[Tuple[str, float]]:
        """Compute outage criticality index for each distribution line (N-1 vulnerability)."""
        criticality = []
        for line in self.all_lines:
            res = self.sim.simulate_contingency([line])
            unserved = res["unserved_load_kw"]
            criticality.append((line, unserved))

        # Sort descending by unserved kW impact
        criticality.sort(key=lambda x: x[1], reverse=True)
        return criticality

    def evaluate_hardening_policy(
        self,
        top_k: int = 5,
        n_scenarios: int = 150,
        stress_level: str = "high"
    ) -> Dict[str, Any]:
        """Evaluate resilience gain from hardening top-k critical lines."""
        critical_lines = self.identify_critical_lines()
        hardened_lines = [line for line, _ in critical_lines[:top_k]]
        
        gen = MonteCarloContingencyGenerator(self.config, seed=42)
        
        # Generate baseline scenarios
        scenarios_base = gen.generate_scenario_batch(
            line_names=self.all_lines,
            n_scenarios=n_scenarios,
            stress_level=stress_level,
            hardened_lines=None
        )
        base_res = self.sim.run_monte_carlo_evaluation(scenarios_base)

        # Generate scenarios with hardened lines (reduced failure probability)
        scenarios_hardened = gen.generate_scenario_batch(
            line_names=self.all_lines,
            n_scenarios=n_scenarios,
            stress_level=stress_level,
            hardened_lines=hardened_lines
        )
        hardened_res = self.sim.run_monte_carlo_evaluation(scenarios_hardened)

        avoided_eens = base_res["mean_eens_kwh"] - hardened_res["mean_eens_kwh"]
        risk_reduction_pct = (avoided_eens / max(1.0, base_res["mean_eens_kwh"])) * 100.0

        return {
            "top_k": top_k,
            "hardened_lines": hardened_lines,
            "baseline_mean_eens_kwh": base_res["mean_eens_kwh"],
            "baseline_q95_eens_kwh": base_res["q95_eens_kwh"],
            "hardened_mean_eens_kwh": hardened_res["mean_eens_kwh"],
            "hardened_q95_eens_kwh": hardened_res["q95_eens_kwh"],
            "avoided_eens_kwh": avoided_eens,
            "risk_reduction_percentage": risk_reduction_pct
        }
