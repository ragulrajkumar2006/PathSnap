"""Mac 1 / models stage - Model definitions (untuned baseline configurations).

Every model outputs a probability that a (fix, candidate arc) pair is the true
match, which is what the HMM emission needs.
"""
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pathsnap_core.config import SEED

# name -> (family, factory(params) -> estimator, default params, max training rows)
# max training rows: KNN is subsampled because prediction cost grows with training size.
MODELS = {
    "logistic_regression": ("Linear / probabilistic",
                            lambda p: make_pipeline(StandardScaler(), LogisticRegression(**p)),
                            {"C": 1.0, "max_iter": 1000}, None),
    "gaussian_nb": ("Linear / probabilistic", lambda p: GaussianNB(**p), {"var_smoothing": 1e-9}, None),
    "knn": ("Instance-based",
            lambda p: make_pipeline(StandardScaler(), KNeighborsClassifier(**p)),
            {"n_neighbors": 25, "weights": "distance", "n_jobs": -1}, 60_000),
    "decision_tree": ("Tree-based", lambda p: DecisionTreeClassifier(**p),
                      {"max_depth": 12, "min_samples_leaf": 20, "random_state": SEED}, None),
    "random_forest": ("Ensemble (bagging)", lambda p: RandomForestClassifier(**p),
                      {"n_estimators": 200, "min_samples_leaf": 5, "n_jobs": -1, "random_state": SEED}, None),
    "extra_trees": ("Ensemble (bagging)", lambda p: ExtraTreesClassifier(**p),
                    {"n_estimators": 200, "min_samples_leaf": 5, "n_jobs": -1, "random_state": SEED}, None),
    "hist_gradient_boosting": ("Boosting", lambda p: HistGradientBoostingClassifier(**p),
                               {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31,
                                "early_stopping": True, "validation_fraction": 0.1,
                                "random_state": SEED}, None),
    "mlp": ("Neural network",
            lambda p: make_pipeline(StandardScaler(), MLPClassifier(**p)),
            {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3,
             "max_iter": 60, "early_stopping": True, "validation_fraction": 0.1,
             "n_iter_no_change": 8, "random_state": SEED}, None),
}


def build(name, params=None):
    family, factory, default, _ = MODELS[name]
    p = dict(default)
    if params:
        p.update(params)
    return factory(p)
