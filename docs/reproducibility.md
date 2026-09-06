# Reproducibility Guide

## Environment Setup
1. Python 3.11+ is required.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
3. Run the complete automated test suite:
   ```bash
   pytest tests/
   ```

## End-to-End Pipeline Execution
Run the full pipeline using `make` commands:
```bash
make data        # Builds the verified hourly panel dataset
make baselines   # Fits statistical and machine learning baselines
make temporal    # Trains temporal deep learning models (GRU/LSTM)
make graph       # Trains spatiotemporal GNN models
make calibrate   # Calibrates probabilities and generates conformal intervals
make ablations   # Runs comprehensive A1-A11 ablation experiments
make failure     # Executes failure analysis on extreme false negatives/positives
make grid        # Executes OpenDSS IEEE-123 Monte Carlo resilience simulations
make figures     # Renders publication-quality figures and tables
make paper       # Compiles paper assets
make all         # Runs the entire pipeline end-to-end
```
