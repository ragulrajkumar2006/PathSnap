"""Mac 1 / models stage - Train, cross-validate and compare the candidate classifiers.

Run:  python -m mac1.models.train_models            (all models)
      python -m mac1.models.train_models knn mlp    (a subset)

Protocol
  * Training data: 80 train trips, every TRAIN_STRIDE-th fix, valid labels only.
  * GroupKFold (5 folds) with trip_id as group, so no trip is split across folds.
  * Student-test (10 trips, every fix) is used for development-stage comparison.
  * Primary selection metric: CV macro F1 (computed without touching student-test),
    tie-broken by student-test Top-1 accuracy.
Outputs: results/models/, artifacts/untuned_<model>.joblib, artifacts/model_selection.json
"""
import json
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from pathsnap_core.config import ARTIFACTS, CV_FOLDS, RESULTS, SEED, ensure_dirs
from pathsnap_core.data import eval_frame, training_frame
from pathsnap_core.evaluation import full_eval
from pathsnap_core.features import FEATURES
from mac1.models.model_zoo import MODELS, build

OUT = RESULTS / "models"


def subsample(X, y, n, rng):
    if n is None or len(y) <= n:
        return X, y
    idx = rng.choice(len(y), n, replace=False)
    return X[idx], y[idx]


def top1_from_rows(df_rows, p):
    d = df_rows.assign(p=p)
    best = d.loc[d.groupby(["trip_id", "obs"])["p"].idxmax()]
    return float(best.label.mean())


def cross_validate(name, df, X, y, groups):
    rng = np.random.default_rng(SEED)
    cap = MODELS[name][3]
    rows = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=CV_FOLDS).split(X, y, groups)):
        Xt, yt = subsample(X[tr], y[tr], cap, rng)
        model = build(name)
        t0 = time.time()
        model.fit(Xt, yt)
        fit_s = time.time() - t0
        pv = model.predict_proba(X[va])[:, 1]
        # training-fold score on a sample, for over/underfitting analysis
        ti = rng.choice(len(yt), min(len(yt), 50_000), replace=False)
        pt = model.predict_proba(Xt[ti])[:, 1]
        rows.append({
            "model": name, "fold": fold, "fit_s": fit_s,
            "train_macro_f1": f1_score(yt[ti], pt >= .5, average="macro"),
            "cv_macro_f1": f1_score(y[va], pv >= .5, average="macro"),
            "cv_log_loss": log_loss(y[va], np.clip(pv, 1e-7, 1 - 1e-7), labels=[0, 1]),
            "cv_top1": top1_from_rows(df.iloc[va], pv),
        })
        print(f"  fold {fold}: F1 {rows[-1]['cv_macro_f1']:.4f}  top1 {rows[-1]['cv_top1']:.4f}  ({fit_s:.1f}s)")
    return rows


def history(name, model):
    """Epoch-wise curves only for models that actually produce them."""
    est = model[-1] if hasattr(model, "steps") else model
    if name == "mlp":
        return {"train_loss": list(est.loss_curve_), "val_accuracy": list(est.validation_scores_)}
    if name == "hist_gradient_boosting":
        return {"train_score": list(est.train_score_), "val_score": list(est.validation_score_)}
    return None


def main(names):
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    df, X, y, groups = training_frame()
    te_df, te_fixes = eval_frame("student_test")
    Xte = te_df[FEATURES].to_numpy(float)
    print(f"train rows {len(y)} from {len(np.unique(groups))} trips, positives {y.mean():.3f}; "
          f"student-test rows {len(te_df)}")

    fold_rows, summary, histories, cms = [], [], {}, {}
    rng = np.random.default_rng(SEED)
    for name in names:
        family, _, params, cap = MODELS[name]
        print(f"\n== {name} ({family})")
        fr = cross_validate(name, df, X, y, groups)
        fold_rows += fr
        Xt, yt = subsample(X, y, cap, rng)
        model = build(name)
        t0 = time.time()
        model.fit(Xt, yt)
        train_s = time.time() - t0
        t0 = time.time()
        p = model.predict_proba(Xte)[:, 1]
        infer_s = time.time() - t0
        ev = full_eval(te_df, te_fixes, p, "student_test")
        valid = te_df.label_valid.to_numpy() == 1
        cms[name] = confusion_matrix(te_df.label.to_numpy()[valid], (p[valid] >= .5).astype(int))
        h = history(name, model)
        if h:
            histories[name] = h
        f = pd.DataFrame(fr)
        summary.append({
            "model": name, "family": family,
            "config": json.dumps({k: v for k, v in params.items() if k not in ("n_jobs",)}, default=str),
            "cv_macro_f1": f.cv_macro_f1.mean(), "cv_macro_f1_std": f.cv_macro_f1.std(),
            "cv_top1": f.cv_top1.mean(), "cv_log_loss": f.cv_log_loss.mean(),
            "train_macro_f1": f.train_macro_f1.mean(),
            **{f"test_{k}": v for k, v in ev.items()},
            "train_time_s": train_s, "inference_time_s": infer_s,
            "inference_ms_per_1k_rows": 1000 * infer_s / len(te_df) * 1000,
        })
        print(f"  student-test: top1 {ev['top1_accuracy']:.4f}  F1 {ev['macro_f1']:.4f}  "
              f"logloss {ev['log_loss']:.4f}  continuity {ev['route_continuity']:.3f}")
        joblib.dump(model, ARTIFACTS / f"untuned_{name}.joblib", compress=3)

    s = pd.DataFrame(summary)
    s.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "cv_folds.csv", index=False)
    json.dump(histories, open(OUT / "training_histories.json", "w"))
    json.dump({k: v.tolist() for k, v in cms.items()}, open(OUT / "confusion_matrices.json", "w"))

    best = s.sort_values(["cv_macro_f1", "test_top1_accuracy"], ascending=False).iloc[0]
    sel = {"selected_model": best.model, "rule": "highest CV macro F1, tie-break student-test Top-1",
           "cv_macro_f1": best.cv_macro_f1, "test_top1_accuracy": best.test_top1_accuracy}
    json.dump(sel, open(ARTIFACTS / "model_selection.json", "w"), indent=2)
    plots(s, pd.DataFrame(fold_rows), histories, cms)
    cols = ["model", "cv_macro_f1", "test_top1_accuracy", "test_macro_f1", "test_log_loss",
            "test_route_continuity", "inference_time_s"]
    print("\n", s[cols].round(4).to_string(index=False))
    print(f"\nSelected model: {best.model}")


def plots(s, folds, histories, cms):
    s = s.sort_values("cv_macro_f1")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(s.model, s.cv_macro_f1, xerr=s.cv_macro_f1_std, color="tab:blue")
    ax.set(xlabel="CV macro F1 (5-fold GroupKFold, train trips)", title="Cross-validation macro F1 by model",
           xlim=(max(0, s.cv_macro_f1.min() - .05), 1))
    fig.tight_layout(); fig.savefig(OUT / "cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(s))
    ax.bar(xs - .2, s.test_top1_accuracy, .4, label="Top-1 accuracy")
    ax.bar(xs + .2, s.test_macro_f1, .4, label="Macro F1")
    ax.set_xticks(xs, s.model, rotation=30, ha="right")
    ax.set(title="Student-test performance", ylim=(0.5, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "student_test_top1_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].barh(s.model, s.test_log_loss, color="tab:red")
    ax[0].set(title="Student-test log loss (lower is better)")
    ax[1].barh(s.model, s.inference_ms_per_1k_rows, color="tab:gray")
    ax[1].set(title="Inference time (ms per 1k candidate rows)", xscale="log")
    fig.tight_layout(); fig.savefig(OUT / "log_loss_and_inference.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(s.train_macro_f1, s.cv_macro_f1)
    for _, r in s.iterrows():
        ax.annotate(r.model, (r.train_macro_f1, r.cv_macro_f1), fontsize=8)
    lo = min(s.train_macro_f1.min(), s.cv_macro_f1.min()) - .02
    ax.plot([lo, 1], [lo, 1], "k--", lw=.8)
    ax.set(xlabel="training-fold macro F1", ylabel="CV macro F1",
           title="Over/underfitting check (gap from diagonal = overfitting)")
    fig.tight_layout(); fig.savefig(OUT / "train_vs_cv.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for m, g in folds.groupby("model"):
        ax.plot(g.fold, g.cv_macro_f1, marker="o", label=m)
    ax.set(xlabel="fold", ylabel="macro F1", title="Per-fold CV macro F1 (stability)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "cv_fold_stability.png", dpi=130); plt.close(fig)

    n = len(cms)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    for a, (m, cm) in zip(np.ravel(axes), cms.items()):
        ConfusionMatrixDisplay(cm, display_labels=["not match", "match"]).plot(ax=a, colorbar=False, values_format="d")
        a.set_title(m, fontsize=9)
    for a in np.ravel(axes)[n:]:
        a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=110); plt.close(fig)

    for m, h in histories.items():
        fig, ax1 = plt.subplots(figsize=(7, 4))
        k1, k2 = list(h)
        ax1.plot(h[k1], label=k1, color="tab:blue")
        ax1.set(xlabel="epoch" if m == "mlp" else "boosting iteration", ylabel=k1)
        ax2 = ax1.twinx()
        ax2.plot(h[k2], label=k2, color="tab:orange")
        ax2.set_ylabel(k2)
        fig.legend(loc="center right")
        ax1.set_title(f"{m}: training history (from fitted model)")
        fig.tight_layout(); fig.savefig(OUT / f"history_{m}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
