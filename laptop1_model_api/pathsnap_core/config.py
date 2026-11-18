"""Shared paths and constants for the whole PathSnap pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

TRAIN_DIR = DATA_RAW / "train"
STUDENT_TEST_DIR = DATA_RAW / "student_test"
# Evaluator trips are dropped in here on the API laptop on evaluation day.
EVALUATOR_DIR = DATA_RAW / "evaluator_test"

SEED = 42

# --- cleaning ---
MAX_SPEED_MPS = 85.0          # implied speeds above this (306 km/h) are treated as GPS glitches

# --- candidate generation ---
CAND_RADIUS_M = 60.0          # search radius around each GPS fix
CAND_MAX_K = 12               # keep at most this many nearest arcs per fix
DENSIFY_STEP_M = 10.0         # arcs are sampled every N metres for the KD-tree index

# --- features ---
CONTEXT_WINDOW = 3            # fixes on each side used for windowed distance features

# --- training ---
TRAIN_STRIDE = 5              # use every 5th fix per training trip (1 Hz data is highly autocorrelated)
CV_FOLDS = 5


def ensure_dirs():
    for d in (DATA_PROCESSED, ARTIFACTS, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
