import logging
from pathlib import Path
import numpy as np
import pandas as pd
from grid_outage_research.grid_resilience.dss_feeder import IEEE123Feeder
from grid_outage_research.grid_resilience.simulator import FeederResilienceSimulator
from grid_outage_research.grid_resilience.decision_hardening import GridHardeningDecisionEngine
from grid_outage_research.uncertainty.propagation import UncertaintyPropagationEngine
from grid_outage_research.utils.config import load_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_grid_simulations")

def main():
    grid_config = load_yaml("configs/grid_resilience.yaml")
    
    logger.info("Initializing OpenDSS IEEE 123-Bus Feeder...")
    feeder = IEEE123Feeder()
    feeder.build_case()
    base_report = feeder.validate_base_feeder()

    sim = FeederResilienceSimulator(feeder)

    # 1. Uncertainty Propagation across Low, Moderate, High, Extreme Outage Risk Cases
    logger.info("Propagating predictive uncertainty into feeder resilience distributions...")
    engine = UncertaintyPropagationEngine(sim, grid_config)
    
    case_studies = [
        ("Low Risk Scenario", 0.05, 0.01, 0.10),
        ("Moderate Risk Scenario", 0.25, 0.15, 0.40),
        ("High Risk Scenario", 0.65, 0.45, 0.85),
        ("Extreme Storm Scenario", 0.90, 0.75, 0.99)
    ]

    propagation_results = []
    for name, p_mid, p_low, p_high in case_studies:
        res = engine.map_prediction_to_resilience_distribution(
            predicted_prob=p_mid,
            interval_low=p_low,
            interval_high=p_high,
            n_mc_samples=100
        )
        res["scenario_tier"] = name
        propagation_results.append(res)
        logger.info(f"[{name}] Point EENS={res['point_mean_eens_kwh']:.1f} kWh, 90% Uncertainty Span={res['uncertainty_interval_eens_kwh'][0]:.1f} - {res['uncertainty_interval_eens_kwh'][1]:.1f} kWh")

    # 2. Decision Hardening Experiment
    logger.info("Running Targeted Distribution Line Hardening Decision Experiment...")
    decision_engine = GridHardeningDecisionEngine(sim, grid_config)
    hardening_res = decision_engine.evaluate_hardening_policy(top_k=5, n_scenarios=100, stress_level="high")
    logger.info(f"Hardening Top-5 Lines Avoided {hardening_res['avoided_eens_kwh']:.1f} kWh EENS ({hardening_res['risk_reduction_percentage']:.2f}% risk reduction).")

    # Save simulation results
    sim_dir = Path("results/simulations")
    sim_dir.mkdir(parents=True, exist_ok=True)
    
    df_prop = pd.DataFrame(propagation_results)
    df_prop.to_csv(sim_dir / "uncertainty_propagation_summary.csv", index=False)
    
    import json
    with open(sim_dir / "hardening_experiment.json", "w", encoding="utf-8") as f:
        json.dump(hardening_res, f, indent=2)

    logger.info("Grid simulations completed successfully.")

if __name__ == "__main__":
    main()
