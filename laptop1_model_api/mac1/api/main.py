"""Mac 1 / PathSnap FastAPI backend.

Start:  uvicorn mac1.api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health            model + decoder info
  GET  /trips             trip ids available for evaluation
  POST /predict           {"trip_id": "..."} -> predicted route + server-side metrics

Trip folder: env PATHSNAP_TRIPS_DIR, else data/raw/evaluator_test if present,
else data/raw/student_test. Reference routes are read only inside
score_prediction() after the prediction is finished, and are never returned.
"""
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap_core.config import EVALUATOR_DIR, STUDENT_TEST_DIR
from pathsnap_core.predictor import PathSnapPredictor, score_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")


def trips_dir() -> Path:
    env = os.environ.get("PATHSNAP_TRIPS_DIR")
    if env:
        return Path(env)
    if EVALUATOR_DIR.exists() and any(EVALUATOR_DIR.iterdir()):
        return EVALUATOR_DIR
    return STUDENT_TEST_DIR


TRIPS_DIR = trips_dir()
app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = PathSnapPredictor()
log.info("loaded model=%s decoder=%s trips_dir=%s", predictor.model_info["model"],
         predictor.hmm_params, TRIPS_DIR)


class PredictRequest(BaseModel):
    trip_id: str


def available_trips():
    return sorted(p.name for p in TRIPS_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    log.info("%s %s from %s -> %s in %.2fs", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code, time.time() - t0)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.model_info["model"], "model_params": predictor.model_info["params"],
            "hmm_params": predictor.hmm_params.__dict__, "trips_dir": TRIPS_DIR.name}


@app.get("/trips")
def trips():
    return {"partition": TRIPS_DIR.name, "trips": available_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in available_trips():
        raise HTTPException(404, f"unknown trip_id {req.trip_id}")
    trip_dir = TRIPS_DIR / req.trip_id
    t0 = time.time()
    pt, pred = predictor.predict(trip_dir)
    predict_s = time.time() - t0
    metrics = score_prediction(trip_dir, pt, pred)
    nodes, arcs = pt.trip.nodes, pt.trip.arcs

    def lines(arc_ids):
        return [[[nodes[a, 1], nodes[a, 0]], [nodes[b, 1], nodes[b, 0]]] for a, b in arcs[arc_ids]]

    track = pt.trip.track[pt.keep]
    step = max(1, len(track) // 3000)
    clf = [int(a) for a in pred["classifier_arc"] if a >= 0]
    clf = [a for i, a in enumerate(clf) if i == 0 or a != clf[i - 1]]
    log.info("trip %s: %d fixes, %d route arcs, top1=%s, %.2fs", req.trip_id, pt.n, len(pred["route"]),
             None if metrics is None else round(metrics["top1_accuracy_hmm"], 4), predict_s)
    return {
        "trip_id": req.trip_id,
        "n_fixes": int(pt.n),
        "n_dropped_fixes": int((~pt.keep).sum()),
        "gps": track[::step, [1, 0]].tolist(),
        "predicted_route": lines(pred["route"]),
        "classifier_only_route": lines(clf),
        "predicted_route_arc_ids": list(map(int, pred["route"])),
        "n_breaks": pred["n_breaks"],
        "metrics": metrics,
        "timings": {**pred["timings"], "total_s": predict_s},
    }
