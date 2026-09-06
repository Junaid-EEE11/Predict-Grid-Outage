import pytest
import opendssdirect as dss
from grid_outage_research.grid_resilience.dss_feeder import IEEE123Feeder
from grid_outage_research.grid_resilience.contingency_generator import MonteCarloContingencyGenerator
from grid_outage_research.grid_resilience.simulator import FeederResilienceSimulator

def test_ieee123_feeder_validation():
    feeder = IEEE123Feeder()
    feeder.build_case()
    report = feeder.validate_base_feeder()
    
    assert report["converged"] is True
    assert report["num_lines"] == 50
    assert report["total_load_kw"] > 1000.0

def test_monte_carlo_and_resilience_simulation(sample_grid_config):
    feeder = IEEE123Feeder()
    feeder.build_case()
    all_lines = dss.Lines.AllNames()
    sim = FeederResilienceSimulator(feeder)

    mc = MonteCarloContingencyGenerator(config=sample_grid_config, seed=42)
    scenarios = mc.generate_scenario_batch(line_names=all_lines, stress_level="moderate", n_scenarios=5)

    assert len(scenarios) == 5
    for sc in scenarios:
        assert isinstance(sc, list)

    sim_res = sim.run_monte_carlo_evaluation(scenarios)
    assert "mean_pct_served" in sim_res
    assert "mean_unserved_kw" in sim_res
    assert "mean_eens_kwh" in sim_res
