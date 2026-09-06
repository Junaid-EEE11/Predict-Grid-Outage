from typing import Dict, Any, List
import numpy as np

class ResilienceMetricsCalculator:
    """Calculates power-system resilience indicators (unserved energy, loss triangle, load served)."""

    @staticmethod
    def compute_single_contingency_metrics(
        base_load_kw: float,
        served_load_kw: float,
        total_loads: int,
        served_loads: int,
        voltage_violations: int
    ) -> Dict[str, float]:
        """Instantaneous post-contingency performance metrics."""
        unserved_kw = max(0.0, base_load_kw - served_load_kw)
        load_served_fraction = float(served_load_kw / max(1.0, base_load_kw))
        customer_interruption_fraction = float(max(0, total_loads - served_loads) / max(1, total_loads))

        return {
            "served_load_kw": served_load_kw,
            "unserved_load_kw": unserved_kw,
            "percentage_load_served": load_served_fraction * 100.0,
            "customer_interruption_fraction": customer_interruption_fraction,
            "voltage_violations": float(voltage_violations)
        }

    @staticmethod
    def compute_restoration_trajectory(
        unserved_kw_initial: float,
        time_steps_hours: int = 24,
        repair_rate_per_hour: float = 0.15
    ) -> Dict[str, Any]:
        """Simulate time-dependent restoration curve and compute resilience loss triangle."""
        t = np.arange(time_steps_hours + 1)
        # Exponential restoration profile: unserved(t) = unserved_0 * exp(-lambda * t)
        unserved_t = unserved_kw_initial * np.exp(-repair_rate_per_hour * t)
        
        # Total Expected Energy Not Served (EENS in kWh)
        try:
            eens_kwh = float(np.trapezoid(unserved_t, t))
        except AttributeError:
            eens_kwh = float(np.trapz(unserved_t, t))
        # Normalized Resilience Loss (integral of performance degradation)
        resilience_loss = float(eens_kwh / max(1.0, unserved_kw_initial * time_steps_hours))

        return {
            "time_hours": t,
            "unserved_kw_trajectory": unserved_t,
            "expected_energy_not_served_kwh": eens_kwh,
            "resilience_loss_index": resilience_loss
        }
