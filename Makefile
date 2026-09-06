.PHONY: setup data baselines temporal graph calibrate ablations failure grid figures paper test all clean

PYTHON := python
PIP := pip
PYTEST := pytest

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PYTHON) scripts/build_dataset.py

baselines:
	$(PYTHON) scripts/train_baselines.py

temporal:
	$(PYTHON) scripts/train_temporal.py

graph:
	$(PYTHON) scripts/train_graph.py

calibrate:
	$(PYTHON) scripts/calibrate_models.py

ablations:
	$(PYTHON) scripts/run_ablations.py

failure:
	$(PYTHON) scripts/run_failure_analysis.py

grid:
	$(PYTHON) scripts/run_grid_simulations.py

figures:
	$(PYTHON) scripts/generate_report_assets.py

paper:
	$(PYTHON) -c "print('Manuscript assets updated in paper/')"

test:
	$(PYTEST) tests/

all: data baselines temporal graph calibrate ablations failure grid figures test paper

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/
