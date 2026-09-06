import logging
from typing import Dict, Any, List, Tuple
import numpy as np
from grid_outage_research.grid_resilience.simulator import FeederResilienceSimulator
from grid_outage_research.grid_resilience.contingency_generator import MonteCarloContingencyGenerator

logger = logging.getLogger(__name__)

class UncertaintyPropagationEngine:
    """Propagates calibrated statistical predictive uncertainty into OpenDSS distribution resilience risk."""

    def __init__(self, simulator: FeederResilienceSimulator, grid_config: Dict[str, Any]):
        self.sim = simulator
        self.config = grid_config
        import opendssdirect as dss
        self.all_lines = dss.Lines.AllNames()

    def map_prediction_to_resilience_distribution(
        self,
        predicted_prob: float,
        interval_low: float,
        interval_high: float,
        n_mc_samples: int = 150
    ) -> Dict[str, Any]:
        """Demonstrate how predictive uncertainty widens downstream grid resilience distributions."""
        # Map predicted severe probability and intervals to line failure rates
        # Base mapping: line_failure_rate = 0.02 + 0.35 * outage_risk
        p_point = float(np.clip(0.02 + 0.35 * predicted_prob, 0.01, 0.50))
        p_low = float(np.clip(0.02 + 0.35 * interval_low, 0.01, 0.50))
        p_high = float(np.clip(0.02 + 0.35 * interval_high, 0.01, 0.50))

        gen = MonteCarloContingencyGenerator(self.config, seed=42)

        # 1. Point prediction resilience
        sc_point = gen.generate_scenario_batch(self.all_lines, n_scenarios=n_mc_samples, custom_failure_prob=p_point)
        res_point = self.sim.run_monte_carlo_evaluation(sc_point)

        # 2. Lower bound scenario resilience
        sc_low = gen.generate_scenario_batch(self.all_lines, n_scenarios=n_mc_samples, custom_failure_prob=p_low)
        res_low = self.sim.run_monte_carlo_evaluation(sc_low)

        # 3. Upper bound scenario resilience (tail vulnerability)
        sc_high = gen.generate_scenario_batch(self.all_lines, n_scenarios=n_mc_samples, custom_failure_prob=p_high)
        res_high = self.sim.run_monte_carlo_evaluation(sc_high)

        return {
            "predicted_prob": predicted_prob,
            "interval_low": interval_low,
            "interval_high": interval_high,
            "point_mean_unserved_kw": res_point["mean_unserved_kw"],
            "point_q95_unserved_kw": res_point["q95_unserved_kw"],
            "point_mean_eens_kwh": res_point["mean_eens_kwh"],
            "uncertainty_interval_eens_kwh": (res_low["mean_eens_kwh"], res_high["mean_eens_kwh"]),
            "tail_risk_q95_eens_kwh": res_high["q95_eens_kwh"],
            "pct_served_mean": res_point["mean_pct_served"]
        }
