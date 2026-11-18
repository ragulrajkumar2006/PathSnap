"""Mac 1 / tuning + HMM stage - Bayesian hyperparameter optimisation of the selected model.

Run:  python -m mac1.tuning_hmm.bayes_opt [n_trials]      (default 30)

* Sampler: Optuna TPE (Tree-structured Parzen Estimator), a Bayesian method that
  models p(params | good score) vs p(params | bad score) from earlier trials.
* Objective: mean macro F1 over a 3-fold GroupKFold on the 80 train trips
  (groups = trip_id). 3 folds keep each trial affordable; the median pruner stops
  trials whose early folds are clearly worse than the median.
* Trial 0 is the untuned default configuration, so untuned vs tuned is compared
  under exactly the same folds.
* Student-test data is only used AFTER the search, to report tuned vs untuned.
Outputs: results/tuning_hmm/bayes_*.{csv,png}, artifacts/final_model.joblib, artifacts/final_model.json
"""
import json
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from pathsnap_core.config import ARTIFACTS, RESULTS, SEED, ensure_dirs
from pathsnap_core.data import eval_frame, training_frame
from pathsnap_core.evaluation import full_eval
from pathsnap_core.features import FEATURES
from mac1.models.model_zoo import MODELS, build

OUT = RESULTS / "tuning_hmm"
FOLDS = 3


def search_space(name, trial):
    if name == "hist_gradient_boosting":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
                "max_features": trial.suggest_float("max_features", 0.4, 1.0)}
    if name in ("random_forest", "extra_trees"):
        # leaf >= 2 and <= 300 trees keep the saved model under GitHub's 100 MB file limit
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 8, 40),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30, log=True),
                "max_features": trial.suggest_float("max_features", 0.2, 1.0)}
    if name == "mlp":
        width = trial.suggest_categorical("width", [32, 64, 128])
        depth = trial.suggest_int("depth", 1, 3)
        return {"hidden_layer_sizes": tuple([width] * depth),
                "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)}
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 100, log=True)}
    if name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 5, 75),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"])}
    if name == "decision_tree":
        return {"max_depth": trial.suggest_int("max_depth", 4, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 200, log=True)}
    if name == "gaussian_nb":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-3, log=True)}
    raise ValueError(name)


def default_trial_params(name):
    """The untuned configuration expressed in search-space coordinates (for enqueue)."""
    d = MODELS[name][2]
    if name == "mlp":
        h = d["hidden_layer_sizes"]
        return {"width": 64, "depth": len(h), "alpha": d["alpha"], "learning_rate_init": d["learning_rate_init"]}
    if name == "hist_gradient_boosting":
        return {"learning_rate": d["learning_rate"], "max_leaf_nodes": d["max_leaf_nodes"],
                "min_samples_leaf": 20, "l2_regularization": 1e-6, "max_features": 1.0}
    if name in ("random_forest", "extra_trees"):
        return {"n_estimators": d["n_estimators"], "max_depth": 40,
                "min_samples_leaf": d["min_samples_leaf"],
                "max_features": 1.0 if name == "extra_trees" else 0.22}
    if name == "knn":
        return {"n_neighbors": d["n_neighbors"], "weights": d["weights"]}
    return {k: d[k] for k in ("C", "max_depth", "min_samples_leaf", "var_smoothing") if k in d}


def main(n_trials):
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    name = json.load(open(ARTIFACTS / "model_selection.json"))["selected_model"]
    cap = MODELS[name][3]
    df, X, y, groups = training_frame()
    rng = np.random.default_rng(SEED)
    splits = list(GroupKFold(n_splits=FOLDS).split(X, y, groups))
    print(f"Tuning {name} with TPE, {n_trials} trials, {FOLDS}-fold GroupKFold")

    def objective(trial):
        params = search_space(name, trial)
        scores = []
        for k, (tr, va) in enumerate(splits):
            idx = tr if cap is None or len(tr) <= cap else rng.choice(tr, cap, replace=False)
            m = build(name, params)
            m.fit(X[idx], y[idx])
            scores.append(f1_score(y[va], m.predict_proba(X[va])[:, 1] >= .5, average="macro"))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name=f"pathsnap_{name}",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study.enqueue_trial(default_trial_params(name))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    search_s = time.time() - t0

    trials = study.trials_dataframe()
    trials.to_csv(OUT / "bayes_trials.csv", index=False)
    untuned_cv = study.trials[0].value
    best_params = search_space(name, optuna.trial.FixedTrial(study.best_params))

    # refit untuned and tuned on all training rows, evaluate both on student-test
    te_df, te_fixes = eval_frame("student_test")
    Xte = te_df[FEATURES].to_numpy(float)
    comp = []
    for label, params, cv in (("untuned", None, untuned_cv), ("tuned", best_params, study.best_value)):
        idx = np.arange(len(y)) if cap is None or len(y) <= cap else rng.choice(len(y), cap, replace=False)
        m = build(name, params)
        t1 = time.time(); m.fit(X[idx], y[idx]); fit_s = time.time() - t1
        t1 = time.time(); p = m.predict_proba(Xte)[:, 1]; inf_s = time.time() - t1
        ev = full_eval(te_df, te_fixes, p, "student_test")
        comp.append({"model": f"{name} ({label})", "cv_macro_f1_3fold": cv, "test_top1_accuracy": ev["top1_accuracy"],
                     "test_macro_f1": ev["macro_f1"], "test_log_loss": ev["log_loss"],
                     "test_route_continuity": ev["route_continuity"],
                     "train_time_s": fit_s, "inference_time_s": inf_s})
        if label == "tuned":
            joblib.dump(m, ARTIFACTS / "final_model.joblib", compress=3)
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "bayes_tuned_vs_untuned.csv", index=False)
    json.dump({"model": name, "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()},
               "features": FEATURES, "search_seconds": search_s, "n_trials": len(study.trials),
               "n_pruned": int(sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)),
               "best_cv_macro_f1": study.best_value, "untuned_cv_macro_f1": untuned_cv},
              open(ARTIFACTS / "final_model.json", "w"), indent=2, default=str)

    done = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(done.number, done.value, s=18, label="trial")
    ax.plot(done.number, done.value.cummax(), color="tab:red", label="best so far")
    ax.axhline(untuned_cv, color="gray", ls="--", label="untuned (trial 0)")
    ax.set(xlabel="trial", ylabel="CV macro F1 (3-fold)", title=f"Bayesian optimisation history: {name}")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "bayes_history.png", dpi=130); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1])
        ax.set(title="Hyperparameter importance (fANOVA)")
        fig.tight_layout(); fig.savefig(OUT / "bayes_param_importance.png", dpi=130); plt.close(fig)
    except Exception as e:  # needs enough completed trials
        print("importance skipped:", e)

    print(f"\nsearch time {search_s:.0f}s, best params {best_params}")
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
