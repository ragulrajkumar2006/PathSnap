"""Raw file loading.

File formats (no headers, whitespace separated):
  .track  lon lat t        GPS fixes, t in seconds
  .nodes  lon lat          row index = node id
  .arcs   from to          row index = arc id (directed)
  .route  arc_id           ground-truth arcs in travel order
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import TRAIN_DIR, STUDENT_TEST_DIR, EVALUATOR_DIR


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3) lon, lat, t
    nodes: np.ndarray            # (V, 2) lon, lat
    arcs: np.ndarray             # (E, 2) from, to
    route: np.ndarray | None     # (R,) arc ids, None when not supplied
    meta: dict = field(default_factory=dict)


def _load(path: Path, dtype=float, ndmin=2):
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_trip(trip_dir: Path, with_route: bool = True) -> Trip:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _load(trip_dir / f"{tid}.track")
    nodes = _load(trip_dir / f"{tid}.nodes")
    arcs = _load(trip_dir / f"{tid}.arcs", dtype=np.int64)
    route = None
    rpath = trip_dir / f"{tid}.route"
    if with_route and rpath.exists():
        route = _load(rpath, dtype=np.int64, ndmin=1)
    return Trip(tid, track, nodes, arcs, route)


def list_trips(partition: str) -> list[Path]:
    base = {"train": TRAIN_DIR, "student_test": STUDENT_TEST_DIR,
            "evaluator_test": EVALUATOR_DIR}[partition]
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())
