"""One call that turns a raw trip folder into a candidate feature table.

Used by dataset building, HMM experiments and the API, so
training and serving share exactly the same preprocessing.
"""
import numpy as np

from .candidates import build_candidates
from .config import CAND_RADIUS_M
from .features import make_features
from .io import load_trip
from .labels import align_to_route
from .preprocess import prepare


def process_trip(trip_dir, with_labels=True):
    """Return (PreparedTrip, candidate DataFrame, per-fix label dict or None)."""
    trip = load_trip(trip_dir, with_route=with_labels)
    pt = prepare(trip)
    cand = build_candidates(pt)
    df = make_features(pt, cand)
    df.insert(0, "trip_id", trip.trip_id)
    labels = None
    if with_labels and trip.route is not None:
        lab, ldist, pos = align_to_route(pt, trip.route)
        valid = ldist <= CAND_RADIUS_M
        df["label"] = (df["arc"].to_numpy() == lab[df["obs"].to_numpy()]).astype(int)
        df["label_valid"] = valid[df["obs"].to_numpy()].astype(int)
        labels = {"label_arc": lab, "label_dist": ldist, "route_pos": pos, "valid": valid}
    return pt, df, labels


def fix_table(pt, labels=None):
    """Per-fix summary (used for EDA, decoding and evaluation)."""
    import pandas as pd
    out = pd.DataFrame({"obs": np.arange(pt.n), "t": pt.t,
                        "lon": pt.trip.track[pt.keep, 0], "lat": pt.trip.track[pt.keep, 1]})
    out.insert(0, "trip_id", pt.trip.trip_id)
    if labels is not None:
        out["label_arc"] = labels["label_arc"]
        out["label_dist"] = labels["label_dist"]
        out["label_valid"] = labels["valid"].astype(int)
    return out
