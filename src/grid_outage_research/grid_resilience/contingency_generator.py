import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class MonteCarloContingencyGenerator:
    """Generates reproducible line failure scenarios parameterized by predictive outage risk."""

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        self.config = config
        self.stress_levels = config.get("stress_levels", {})
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample_contingency(
        self,
        line_names: List[str],
        stress_level: str = "moderate",
        custom_failure_prob: Optional[float] = None,
        hardened_lines: Optional[List[str]] = None
    ) -> List[str]:
        """Sample a set of faulted line components under given stress severity."""
        params = self.stress_levels.get(stress_level, {"line_failure_rate": 0.05})
        base_p = custom_failure_prob if custom_failure_prob is not None else params.get("line_failure_rate", 0.05)
        
        failed_lines = []
        hardened_set = set(hardened_lines) if hardened_lines else set()

        for line in line_names:
            p = base_p * 0.1 if line in hardened_set else base_p # Hardening reduces failure probability by 90%
            if self.rng.random() < p:
                failed_lines.append(line)

        return failed_lines

    def generate_scenario_batch(
        self,
        line_names: List[str],
        n_scenarios: int = 100,
        stress_level: str = "moderate",
        custom_failure_prob: Optional[float] = None,
        hardened_lines: Optional[List[str]] = None
    ) -> List[List[str]]:
        """Generate a batch of Monte Carlo line contingency scenarios."""
        scenarios = []
        for _ in range(n_scenarios):
            failed = self.sample_contingency(
                line_names=line_names,
                stress_level=stress_level,
                custom_failure_prob=custom_failure_prob,
                hardened_lines=hardened_lines
            )
            scenarios.append(failed)
        return scenarios
