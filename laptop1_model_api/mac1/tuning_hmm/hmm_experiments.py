"""Mac 1 / tuning + HMM stage - HMM/Viterbi decoder tuning and classifier-only vs classifier+HMM study.

Run:  python -m mac1.tuning_hmm.hmm_experiments

Steps
 1. Out-of-fold (OOF) probabilities for the 80 train trips: 5-fold GroupKFold,
    the final tuned configuration is refit on the other folds' trips. OOF scores
    behave like unseen-trip scores, so decoder tuning is not fooled by the
    over-confident in-sample probabilities.
 2. Grid search of (beta, emission_weight) on a fixed random subset of train
    trips, objective = mean per-fix Top-1 accuracy after decoding.
 3. Student-test comparison of three decoders:
      classifier only (argmax)  |  HMM with distance-only emission  |  classifier + HMM
Outputs: results/tuning_hmm/hmm_*.{csv,png}, predictions_student_test.parquet,
         artifacts/hmm_params.json
"""
import itertools
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap_core.config import ARTIFACTS, CV_FOLDS, RESULTS, SEED, TRAIN_STRIDE, ensure_dirs
from pathsnap_core.data import load_candidates, load_fixes
from pathsnap_core.features import FEATURES
from pathsnap_core.hmm import HMMParams, RoadGraph, stitch_route, viterbi_decode
from pathsnap_core.io import list_trips, load_trip
from pathsnap_core.metrics import route_continuity, route_overlap
from pathsnap_core.preprocess import prepare

OUT = RESULTS / "tuning_hmm"
BETAS = [1.0, 2.0, 5.0, 10.0, 20.0]
EMISSION_WEIGHTS = [0.5, 1.0, 2.0]
N_TUNE_TRIPS = 20
MAX_TUNE_FIXES = 4000


def oof_probabilities(model, cand):
    X = cand[FEATURES].to_numpy(float)
    groups = cand.trip_id.to_numpy()
    p = np.zeros(len(cand))
    train_mask = (cand.label_valid.to_numpy() == 1) & (cand.obs.to_numpy() % TRAIN_STRIDE == 0)
    for tr, va in GroupKFold(n_splits=CV_FOLDS).split(X, groups=groups):
        fit_idx = tr[train_mask[tr]]
        m = clone(model).fit(X[fit_idx], cand.label.to_numpy()[fit_idx])
        p[va] = m.predict_proba(X[va])[:, 1]
    return p


def decode_trip(partition, trip_id, cand_t, fixes_t, params, mode):
    path = next(p for p in list_trips(partition) if p.name == trip_id)
    trip = load_trip(path)
    pt = prepare(trip)
    t0 = time.time()
    if mode == "classifier_only":
        best = cand_t.loc[cand_t.groupby("obs")["p"].idxmax()]
        pred = np.full(pt.n, -1)
        pred[best.obs.to_numpy()] = best.arc.to_numpy()
        g = RoadGraph(pt)
        route, breaks = stitch_route(g, pred, params.max_search_m), 0
    else:
        res = viterbi_decode(pt, cand_t, "p", params, emission="distance" if mode == "hmm_distance" else "classifier")
        pred, route, breaks = res["pred_arc"], res["route"], res["n_breaks"]
    dec_s = time.time() - t0
    lab = fixes_t.label_arc.to_numpy()
    valid = fixes_t.label_valid.to_numpy() == 1
    raw_seq_overlap = route_overlap([a for a in pred if a >= 0], trip.route)
    return {
        "trip_id": trip_id, "mode": mode, "fixes": pt.n,
        "top1_accuracy": float(np.mean(pred[valid] == lab[valid])),
        "route_continuity": route_continuity(pred, trip.arcs),
        "n_breaks": breaks, "decode_s": dec_s,
        **route_overlap(route, trip.route),
        "route_f1_length_weighted": route_overlap(route, trip.route, pt.arc_len)["route_f1"],
        "matched_arcs_f1": raw_seq_overlap["route_f1"],
    }, pred, route


def tune(cand, fixes):
    rng = np.random.default_rng(SEED)
    sizes = fixes.groupby("trip_id").size()
    pool = sizes[sizes <= MAX_TUNE_FIXES].index.to_numpy()
    tune_trips = sorted(rng.choice(pool, min(N_TUNE_TRIPS, len(pool)), replace=False))
    grid = list(itertools.product(BETAS, EMISSION_WEIGHTS))
    by_c = dict(tuple(cand[cand.trip_id.isin(tune_trips)].groupby("trip_id")))
    by_f = dict(tuple(fixes[fixes.trip_id.isin(tune_trips)].groupby("trip_id")))

    def run(beta, w, tid):
        r, _, _ = decode_trip("train", tid, by_c[tid], by_f[tid], HMMParams(beta=beta, emission_weight=w), "hmm_classifier")
        return {"beta": beta, "emission_weight": w, **r}

    rows = Parallel(n_jobs=-1)(delayed(run)(b, w, t) for b, w in grid for t in tune_trips)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "hmm_tuning_per_trip.csv", index=False)
    agg = res.groupby(["beta", "emission_weight"])[["top1_accuracy", "route_f1", "route_continuity"]].mean().reset_index()
    agg.to_csv(OUT / "hmm_tuning_grid.csv", index=False)
    best = agg.sort_values("top1_accuracy", ascending=False).iloc[0]

    piv = agg.pivot(index="emission_weight", columns="beta", values="top1_accuracy")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    ax.set(xlabel="beta (m)", ylabel="emission weight",
           title=f"Decoder tuning on {len(tune_trips)} train trips (OOF probs): Top-1")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT / "hmm_tuning_grid.png", dpi=130); plt.close(fig)
    return HMMParams(beta=float(best.beta), emission_weight=float(best.emission_weight)), tune_trips


def main():
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    model = joblib.load(ARTIFACTS / "final_model.joblib")

    print("1/3 out-of-fold probabilities on train trips ...")
    tr_c = load_candidates("train")
    tr_c["p"] = oof_probabilities(model, tr_c)
    tr_f = load_fixes("train")

    print("2/3 decoder grid search ...")
    params, tune_trips = tune(tr_c, tr_f)
    json.dump({"params": params.__dict__, "tuned_on": "train trips (OOF probabilities)",
               "tune_trips": list(map(str, tune_trips)), "grid": {"beta": BETAS, "emission_weight": EMISSION_WEIGHTS}},
              open(ARTIFACTS / "hmm_params.json", "w"), indent=2)
    print("   best", params)

    print("3/3 student-test comparison ...")
    te_c = load_candidates("student_test")
    te_c["p"] = model.predict_proba(te_c[FEATURES].to_numpy(float))[:, 1]
    te_f = load_fixes("student_test")
    rows, preds = [], []
    for tid in sorted(te_f.trip_id.unique()):
        c, f = te_c[te_c.trip_id == tid], te_f[te_f.trip_id == tid]
        for mode in ("classifier_only", "hmm_distance", "hmm_classifier"):
            r, pred, route = decode_trip("student_test", tid, c, f, params, mode)
            rows.append(r)
            preds.append(pd.DataFrame({"trip_id": tid, "mode": mode, "obs": np.arange(len(pred)), "pred_arc": pred}))
            preds.append(pd.DataFrame({"trip_id": tid, "mode": f"{mode}_route", "obs": np.arange(len(route)), "pred_arc": route}))
        print(f"   {tid}: " + "  ".join(f"{r['mode']} {r['top1_accuracy']:.3f}" for r in rows[-3:]))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "hmm_student_test_per_trip.csv", index=False)
    pd.concat(preds).to_parquet(OUT / "predictions_student_test.parquet", index=False)
    summary = res.groupby("mode")[["top1_accuracy", "route_continuity", "route_precision", "route_recall",
                                   "route_f1", "route_f1_length_weighted", "n_breaks", "decode_s"]].mean()
    summary = summary.loc[["classifier_only", "hmm_distance", "hmm_classifier"]]
    summary.to_csv(OUT / "hmm_student_test_summary.csv")
    plots(res, summary)
    print(summary.round(4).to_string())


def plots(res, summary):
    modes = ["classifier_only", "hmm_distance", "hmm_classifier"]
    colors = ["tab:red", "tab:gray", "tab:blue"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, metric in zip(ax, ["top1_accuracy", "route_continuity", "route_f1"]):
        piv = res.pivot(index="trip_id", columns="mode", values=metric)[modes]
        xs = np.arange(len(piv))
        for k, (m, c) in enumerate(zip(modes, colors)):
            a.bar(xs + (k - 1) * .27, piv[m], .27, label=m, color=c)
        a.set_xticks(xs, piv.index, rotation=60, fontsize=7)
        a.set(title=f"Student-test {metric} per trip", ylim=(0, 1.02))
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "hmm_per_trip_comparison.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ["top1_accuracy", "route_continuity", "route_precision", "route_recall", "route_f1"]
    xs = np.arange(len(metrics))
    for k, (m, c) in enumerate(zip(modes, colors)):
        ax.bar(xs + (k - 1) * .27, summary.loc[m, metrics], .27, label=m, color=c)
    ax.set_xticks(xs, metrics, rotation=20)
    ax.set(ylim=(0, 1.02), title="Classifier-only vs HMM decoding (student-test mean over 10 trips)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "hmm_summary_comparison.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
