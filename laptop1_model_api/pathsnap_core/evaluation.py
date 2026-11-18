"""Evaluation helpers that combine model scores with trip geometry."""
import numpy as np
import pandas as pd

from .io import list_trips, load_trip
from .metrics import candidate_metrics, route_continuity, top1_accuracy, top1_choice

_ARCS_CACHE: dict = {}


def trip_arcs(partition, trip_id):
    key = (partition, trip_id)
    if key not in _ARCS_CACHE:
        path = next(p for p in list_trips(partition) if p.name == trip_id)
        _ARCS_CACHE[key] = load_trip(path).arcs
    return _ARCS_CACHE[key]


def classifier_continuity(df, partition, score_col="p"):
    """Mean per-trip route continuity of classifier-only (argmax) predictions."""
    best = top1_choice(df, score_col)
    vals = []
    for tid, g in best.groupby("trip_id"):
        vals.append(route_continuity(g.sort_values("obs").pred_arc.to_numpy(), trip_arcs(partition, tid)))
    return float(np.mean(vals))


def full_eval(df, fixes, p, partition):
    """All observation- and candidate-level metrics for scores p on df."""
    d = df.assign(p=p)
    valid = d.label_valid == 1
    out = candidate_metrics(d.label[valid].to_numpy(), d.p[valid].to_numpy())
    out["top1_accuracy"] = top1_accuracy(d, fixes)
    out["route_continuity"] = classifier_continuity(d, partition)
    return out
