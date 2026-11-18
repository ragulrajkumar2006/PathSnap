# PathSnap: Distributed Map-Matching and Route Reconstruction

UE24CS352A Machine Learning Hackathon (PES University).

## Team Members

| Name | SRN |
|---|---|
| Rhea Nair | PES2UG24AM133 |
| Riya Harugop | PES2UG24AM136 |
| Ragul Rajkumar S | PES2UG24AM128 |
| Vinayak Kalal | PES2UG25AM809 |

PathSnap matches noisy GPS fixes to a road network. It generates candidate road segments for each fix, scores them with a classifier, and decodes a connected route with an HMM and the Viterbi algorithm.

## Project Pipeline

```text
raw trip ──► clean + project ──► candidate arcs ──► features ──► classifier P(match)
                                                                     │
                     route + metrics ◄── stitch ◄── Viterbi (HMM) ◄──┘
```

## Two-Laptop Deployment

```text
 Laptop 1: model + API                       Laptop 2: UI
 ┌──────────────────────────────┐  same    ┌──────────────────────────┐
 │ full pipeline, trained model │  Wi-Fi / │ Streamlit UI (:8501)     │
 │ HMM, road networks, .route   │◄hotspot─►│ pick trip → map+metrics  │
 │ FastAPI (:8000) + request log│          │ talks only to the API    │
 └──────────────────────────────┘          └──────────────────────────┘
```

## Repository Layout

```text
.
├── laptop1_model_api/          Laptop 1: everything for training, tuning and serving
│   ├── pathsnap_core/          shared library (loading, cleaning, candidates, labels, features, metrics, HMM, predictor)
│   ├── mac1/data/              EDA, data-quality checks, Folium maps, dataset building
│   ├── mac1/models/            8 models, group-aware CV, comparison plots, model selection
│   ├── mac1/tuning_hmm/        Bayesian optimisation (Optuna TPE), HMM tuning, classifier vs HMM, route maps
│   ├── mac1/api/               FastAPI backend + all-trips API test
│   ├── artifacts/              saved model weights and configs (final + all untuned models, HMM params)
│   ├── results/                CSVs, plots, Folium HTML maps, experiment logs, deployment screenshots
│   ├── data/raw/               put the course dataset here (not committed, see its README)
│   ├── requirements.txt
│   ├── run_all.sh
│   └── start_api.command
│
├── laptop2_ui/                 Laptop 2: Streamlit + Folium evaluator UI (no model, no data)
│
└── docs/HOW_IT_WORKS.md        method, results and design explanations
```

## Where Each Deliverable Lives

| Required component | Location |
|---|---|
| Data processing | `pathsnap_core/{io,data,preprocess,geo}.py`, `mac1/data/eda.py`, `mac1/data/build_dataset.py` |
| Feature engineering | `pathsnap_core/{candidates,labels,features}.py` |
| Model training | `mac1/models/model_zoo.py`, `mac1/models/train_models.py` |
| Model evaluation | `pathsnap_core/{metrics,evaluation}.py`, `results/models/` (comparison CSV, CV folds, confusion matrices, plots) |
| Hyperparameter optimisation | `mac1/tuning_hmm/bayes_opt.py`, `results/tuning_hmm/bayes_trials.csv`, `bayes_tuned_vs_untuned.csv` |
| HMM and Viterbi | `pathsnap_core/hmm.py`, `mac1/tuning_hmm/hmm_experiments.py`, `results/tuning_hmm/hmm_*.csv` |
| API | `mac1/api/main.py`, `pathsnap_core/predictor.py`, `results/api/` (API test results incl. Evaluation 2 run) |
| UI | `laptop2_ui/app.py` |
| Deployment | `requirements.txt` files, `start_api.command`, this README, `laptop2_ui/README.md` |
| Visualisations | `mac1/data/folium_maps.py`, `mac1/tuning_hmm/route_maps.py`, `results/data/maps/`, `results/tuning_hmm/maps/` |
| Saved model | `artifacts/final_model.joblib`, `final_model.json`, `hmm_params.json`, `model_selection.json` |
| API/UI screenshots | `results/deployment_screenshots/` (Laptop 1 API logs, Laptop 2 UI) |

Paths in the table are relative to `laptop1_model_api/`, except `laptop2_ui/`.

## Laptop 1 Setup (Python 3.11–3.13)

```bash
cd laptop1_model_api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the dataset into:

```text
data/raw/train/
data/raw/student_test/
```

See `data/raw/README.md` for dataset placement instructions.

## Offline Pipeline

Run the complete pipeline using:

```bash
./run_all.sh
```

Approximate total runtime: 1 hour.

Individual stages can also be executed separately:

```bash
python -m mac1.data.eda                    # ~1 min
python -m mac1.data.folium_maps            # ~10 s
python -m mac1.data.build_dataset          # ~20 s
python -m mac1.models.train_models         # ~4 min
python -m mac1.tuning_hmm.bayes_opt 30     # ~45 min
python -m mac1.tuning_hmm.hmm_experiments  # ~5 min
python -m mac1.tuning_hmm.route_maps       # ~30 s
```

The trained artifacts are already committed, so you can start the API without rerunning the pipeline.

## Start the API (Laptop 1)

```bash
uvicorn mac1.api.main:app --host 0.0.0.0 --port 8000
ipconfig getifaddr en0                         # Laptop 1's IP, give it to Laptop 2
python -m mac1.api.test_all_trips              # optional: runs every trip through the API
```

On macOS you can also double-click `start_api.command`. It prints the URL to give to Laptop 2.

## Laptop 2 Setup

Laptop 2 only needs the `laptop2_ui/` folder.

```bash
cd laptop2_ui
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py -- --api http://<LAPTOP1_IP>:8000
```

See `laptop2_ui/README.md` for troubleshooting.

## Evaluation Day

1. Connect both laptops to the **same phone hotspot**. Campus Wi-Fi often blocks connections between devices.
2. On Laptop 1, copy the evaluator trips to `data/raw/evaluator_test/<trip_id>/`, then (re)start the API. It serves `evaluator_test` automatically when that folder exists, otherwise `student_test`. You can override this with `PATHSNAP_TRIPS_DIR=/path`.
3. On Laptop 2, start the UI and check that the dropdown lists 10 trips.
4. Keep Laptop 1's terminal visible. Every request is logged with client IP, trip, Top-1 and time.

The UI never loads the model or any `.route` file. The API reads the reference route only after the prediction is finished, to compute accuracy, and never returns it.

## Method Summary

- **Cleaning:** Fixes that are non-finite, have non-increasing timestamps, or imply a speed above 85 m/s are dropped.
- **Projection:** Trips are spread worldwide, so each trip uses a local equirectangular projection centred on its own GPS mean.
- **Candidates:** Arcs within 60 m of a fix, at most 12, found with a KD-tree over arcs densified every 10 m.
- **Labels:** `.route` has no per-fix timing, so a monotone dynamic programme aligns fixes to route positions. Fixes more than 60 m from their aligned arc are marked `label_valid=0` and excluded from training and scoring.
- **Features (21):** Distance and rank, distance relative to the nearest candidate, projection fraction, heading difference, windowed distance/heading over ±3 fixes, connectivity with neighbouring fixes' nearest arc, reverse-twin flag, node degrees, speed and time gap.
- **Models:** Logistic Regression, Gaussian NB, KNN, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, MLP. 5-fold GroupKFold by trip, with student-test used for comparison.
- **HMM:** States are candidate arcs. Emission is `w·log p(classifier)`. Transition is `-|d_route − d_gc| / (β√dt)`, using a bounded Dijkstra on the trip's road graph. Chains restart when no transition is reachable. β and w are tuned on out-of-fold train predictions.

See `docs/HOW_IT_WORKS.md` for detailed results and explanations.

## Reproducibility

Seed 42 everywhere (`laptop1_model_api/pathsnap_core/config.py`). All reported numbers come from the CSVs in `laptop1_model_api/results/`.
