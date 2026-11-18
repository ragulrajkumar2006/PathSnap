"""End-to-end inference: raw trip folder -> predicted arc per fix + route.

Used by the API on Mac 1. The .route file is never read here; scoring against
the reference happens separately in `score_prediction`, server-side only.
"""
import json
import time

import joblib
import numpy as np

from .config import ARTIFACTS, CAND_RADIUS_M
from .features import FEATURES
from .hmm import HMMParams, RoadGraph, viterbi_decode
from .io import load_trip
from .labels import align_to_route
from .metrics import route_continuity, route_overlap
from .pipeline import process_trip
from .preprocess import prepare


class PathSnapPredictor:
    def __init__(self, artifacts=ARTIFACTS):
        self.model = joblib.load(artifacts / "final_model.joblib")
        self.model_info = json.load(open(artifacts / "final_model.json"))
        hp = json.load(open(artifacts / "hmm_params.json"))
        self.hmm_params = HMMParams(**hp["params"])

    def predict(self, trip_dir):
        timings = {}
        t0 = time.time()
        pt, df, _ = process_trip(trip_dir, with_labels=False)
        timings["features_s"] = time.time() - t0
        t0 = time.time()
        df["p"] = self.model.predict_proba(df[FEATURES].to_numpy(float))[:, 1] if len(df) else []
        timings["classifier_s"] = time.time() - t0
        t0 = time.time()
        g = RoadGraph(pt)
        res = viterbi_decode(pt, df, "p", self.hmm_params, graph=g)
        timings["viterbi_s"] = time.time() - t0
        best = df.loc[df.groupby("obs")["p"].idxmax()] if len(df) else df
        clf_arc = np.full(pt.n, -1)
        clf_arc[best.obs.to_numpy()] = best.arc.to_numpy()
        return pt, {"pred_arc": res["pred_arc"], "classifier_arc": clf_arc, "route": res["route"],
                    "n_breaks": res["n_breaks"], "timings": timings}


def score_prediction(trip_dir, pt, pred):
    """Server-side scoring against the reference route (never sent to the model)."""
    trip = load_trip(trip_dir, with_route=True)
    if trip.route is None:
        return None
    lab, ldist, _ = align_to_route(pt, trip.route)
    valid = ldist <= CAND_RADIUS_M
    arcs = trip.arcs
    out = {
        "fixes_scored": int(valid.sum()),
        "top1_accuracy_hmm": float(np.mean(pred["pred_arc"][valid] == lab[valid])),
        "top1_accuracy_classifier_only": float(np.mean(pred["classifier_arc"][valid] == lab[valid])),
        "route_continuity_hmm": route_continuity(pred["pred_arc"], arcs),
        "route_continuity_classifier_only": route_continuity(pred["classifier_arc"], arcs),
    }
    out.update(route_overlap(pred["route"], trip.route))
    out.update({f"{k}_length_weighted": v for k, v in route_overlap(pred["route"], trip.route, pt.arc_len).items()})
    return out
