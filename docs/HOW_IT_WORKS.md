# How PathSnap works: results and viva preparation

All numbers below come from the reference run (seed 42) and can be found in `results/`.

---

## 1. Data (`mac1/data/`, results in `results/data/`)

**Files:** `.track` (lon, lat, t), `.nodes` (lon, lat, where the row number is the node id), `.arcs` (from, to, where the row number is the arc id, directed), `.route` (arc ids in travel order, **no per-fix timing**). Each trip has its own local road network.

**Findings**
- 80 train trips (208,417 fixes) and 10 student-test trips (19,160 fixes). There are no NaNs and no non-increasing timestamps.
- Trips are spread **worldwide** (Europe, Russia, the US, Vietnam and more), so the code uses a local equirectangular projection for each trip rather than one UTM zone.
- About 94% of steps are exactly 1 s apart. There are 70 gaps longer than 60 s, the largest being 153,064 s (train 00000090).
- 61 speed glitches above 85 m/s. Train 00000001 has a real 50–63 m/s Autobahn drive, which is why the threshold is not 60.
- Train 00000002: 12% of its fixes are more than 60 m from every reference arc, because the reference route doesn't cover part of the drive. These fixes are marked `label_valid=0`.
- Candidate miss rate: 0.98% on train, 0.10% on student-test.

**Things to be able to explain**
- **Label alignment DP:** `D[i,j] = cost(i,j) + min_{j'≤j} D[i−1,j']`. It is monotone (the car only moves forward along the route) and it can skip short arcs between two fixes.
- **Candidates:** arcs are densified every 10 m into a KD-tree, then the search keeps arcs within 60 m, at most 12 per fix.
- **The reverse-twin problem:** a two-way road is stored as u→v and v→u at identical distance. That's why snapping to the nearest arc is only about 50% accurate, and why the heading features matter.

## 2. Models (`mac1/models/`, results in `results/models/`)

The task is framed as **binary classification of (fix, candidate) pairs**. Top-1 picks the highest-probability candidate for each fix. Evaluation uses GroupKFold (5 folds) by trip, so no trip is split across folds. Training uses every 5th fix, because 1 Hz fixes are nearly duplicates. KNN is capped at 60k rows because its prediction cost grows with training size.

| Model | CV macro F1 | Test Top-1 | Test macro F1 | Log loss | Continuity |
|---|---|---|---|---|---|
| **Random Forest (selected)** | 0.9372 | 0.8964 | 0.9389 | 0.0642 | 0.796 |
| HistGradientBoosting | 0.9369 | 0.8937 | 0.9368 | 0.0660 | 0.809 |
| Extra Trees | 0.9351 | 0.8893 | 0.9347 | 0.0665 | 0.815 |
| MLP | 0.9339 | 0.8909 | 0.9341 | 0.0699 | 0.805 |
| Logistic Regression | 0.9336 | 0.8890 | 0.9332 | 0.0776 | 0.806 |
| Decision Tree | 0.9312 | 0.8861 | 0.9339 | 0.0946 | 0.779 |
| KNN | 0.9272 | 0.8826 | 0.9296 | 0.1045 | 0.786 |
| Gaussian NB | 0.8992 | 0.8615 | 0.8918 | 0.4084 | 0.792 |

- **Selection rule:** highest CV macro F1, tie-broken by student-test Top-1. RF and HGB differ by only 0.0003, which is within the fold standard deviation; HGB is about 3x faster at inference.
- Fold 3 is the weakest for every model, so some trips are simply harder.
- Gaussian NB's log loss is poor because its features are correlated, which breaks the independence assumption and makes its probabilities over-confident.
- Only about 12% of candidates are positive, so macro F1 is a better guide than accuracy.

## 3. Bayesian optimisation (`mac1/tuning_hmm/bayes_opt.py`, results in `results/tuning_hmm/`)

- **Optuna TPE:** it splits past trials into good and bad groups, fits a density l(x) to the good ones and g(x) to the bad ones, and proposes the x that maximises l(x)/g(x). A median pruner stops trials that are clearly worse early.
- **Objective:** mean macro F1 over 3-fold GroupKFold on the train trips. **Trial 0 is the untuned configuration**, so both are compared on the same folds.
- **Search space (RF):** n_estimators 100–300, max_depth 8–40, min_samples_leaf 2–30 (log scale), max_features 0.2–1.0.
- **Result:** 30 trials, 13 pruned, 2,674 s. Best: `n_estimators=300, max_depth=15, min_samples_leaf=3, max_features≈0.35`.

| | CV F1 (3-fold) | Test Top-1 | Test F1 | Log loss | Inference |
|---|---|---|---|---|---|
| untuned | 0.9382 | 0.8964 | 0.9389 | 0.0642 | 0.37 s |
| tuned | 0.9384 | 0.8962 | 0.9379 | 0.0633 | 0.43 s |

Tuning gave **no meaningful gain**. The default RF was already at the limit of what these features allow, and the real improvement comes from the HMM.

## 4. HMM and Viterbi (`pathsnap_core/hmm.py`, `mac1/tuning_hmm/hmm_experiments.py`)

- **Hidden states:** the candidate arcs of each fix.
- **Emission:** `w · log p_classifier`. The distance-only baseline uses `−½(d/σ)²`.
- **Transition:** `−|d_route − d_gc| / (β·√dt)`. `d_route` is the network distance between the projected positions on the two arcs, from a bounded Dijkstra. Up to 15 m of backward movement on the same arc is allowed, to absorb GPS noise.
- **Viterbi:** `δ_t(j) = max_i [δ_{t−1}(i) + log A_ij] + log B_j(o_t)`, with back-pointers, all in log space.
- **Gaps and breaks:** fixes with no candidates are skipped. If no candidate is reachable from the previous fix, the chain restarts from the emissions and the break is counted. The Dijkstra bound grows with the gap, up to 5 km.
- **Stitching:** shortest paths fill in where consecutive predicted arcs don't connect.
- **Tuning:** β ∈ {1,2,5,10,20} × w ∈ {0.5,1,2} on 20 train trips, using **out-of-fold** probabilities (in-sample probabilities are over-confident). Best: β=5, w=2. That is at the edge of the grid, so a wider grid could help.

**Student-test, mean over 10 trips**

| | Top-1 | Continuity | Route P | Route R | Route F1 |
|---|---|---|---|---|---|
| classifier only | 0.910 | 0.799 | 0.621 | 0.994 | 0.752 |
| HMM distance only | 0.917 | 0.909 | 0.926 | 0.992 | 0.957 |
| **classifier + HMM** | **0.940** | **0.907** | **0.935** | **0.991** | **0.961** |

- Classifier + HMM beat classifier-only on all 10 trips. On 00000040, the distance-only HMM (0.971) beat classifier + HMM (0.947), and 00000000 tied.
- 00000077 is the hardest trip (0.838). See `results/tuning_hmm/maps/student_test_00000077.html`.
- The classifier-only route precision is low (0.62) because the stitched argmax path jumps onto reverse twins and side streets.
- These Top-1 figures average per trip. The models table pools all fixes, which is why classifier-only shows 0.910 here and 0.896 there.

## 5. API and UI (`laptop1_model_api/mac1/api/`, `laptop2_ui/`)

- **Flow:** UI → `POST /predict {trip_id}` → `process_trip(with_labels=False)` → `model.predict_proba` → `viterbi_decode` → `score_prediction`. The last step is the only place a `.route` file is read, and only after prediction is finished. The response is JSON with the predicted route, GPS points and aggregate metrics, never the ground truth.
- `/trips` only lists real trip folders, and `/predict` rejects any other id, which prevents path tricks.
- Every request is logged on Laptop 1 with client IP, status, trip, Top-1 and time.
- The API reproduces the offline result exactly (mean Top-1 0.9400 across the 10 student-test trips), at under 2 s per trip.
