#!/usr/bin/env bash
# Runs the full offline pipeline on Laptop 1: data -> models -> tuning + HMM. Start the API afterwards.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
$PY -m mac1.data.eda
$PY -m mac1.data.folium_maps
$PY -m mac1.data.build_dataset
$PY -m mac1.models.train_models
$PY -m mac1.tuning_hmm.bayes_opt 30
$PY -m mac1.tuning_hmm.hmm_experiments
$PY -m mac1.tuning_hmm.route_maps
echo "Done. Start the API with: uvicorn mac1.api.main:app --host 0.0.0.0 --port 8000"
