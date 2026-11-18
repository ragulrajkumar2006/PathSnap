"""Shared evaluation metrics, so the report, the HMM study and the API all use
identical definitions.

Definitions
-----------
candidate-level      binary classification of every (fix, candidate) row:
                     accuracy, precision, recall, macro F1, log loss
Top-1 accuracy       per fix: the candidate with the highest score equals the
                     reference arc (fixes whose reference arc is not among the
                     candidates count as wrong)
candidate miss rate  share of fixes whose reference arc is not a candidate
route continuity     share of consecutive changes of predicted arc where the new
                     arc starts at the node where the previous one ends
route P / R / F1     set overlap between predicted route arcs and .route arcs
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score)


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, average="macro", zero_division=0),
        "recall": recall_score(y, yhat, average="macro", zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def top1_choice(df: pd.DataFrame, score_col="p"):
    """Return per-(trip, fix) chosen arc and whether it is correct."""
    idx = df.groupby(["trip_id", "obs"], sort=False)[score_col].idxmax()
    best = df.loc[idx, ["trip_id", "obs", "arc", "label"]]
    return best.rename(columns={"arc": "pred_arc"})


def top1_accuracy(df: pd.DataFrame, fixes: pd.DataFrame, score_col="p"):
    """Top-1 over all valid fixes; fixes with no candidates count as wrong."""
    best = top1_choice(df, score_col)
    m = fixes.merge(best[["trip_id", "obs", "label"]], on=["trip_id", "obs"], how="left")
    m = m[m.label_valid == 1]
    return float(m["label"].fillna(0).mean())


def candidate_miss_rate(df: pd.DataFrame, fixes: pd.DataFrame):
    hit = df[df.label == 1][["trip_id", "obs"]].drop_duplicates().assign(hit=1)
    m = fixes.merge(hit, on=["trip_id", "obs"], how="left")
    m = m[m.label_valid == 1]
    return float(1 - m["hit"].fillna(0).mean())


def route_continuity(arc_seq, arcs):
    """arc_seq: predicted arc per fix (−1 = none). Fraction of arc changes that
    are physically connected (to-node of previous == from-node of next)."""
    s = np.asarray([a for a in arc_seq if a >= 0])
    if len(s) < 2:
        return 1.0
    keep = np.r_[True, s[1:] != s[:-1]]
    s = s[keep]
    if len(s) < 2:
        return 1.0
    return float(np.mean(arcs[s[:-1], 1] == arcs[s[1:], 0]))


def route_overlap(pred_route, true_route, arc_len=None):
    """Set precision / recall / F1 of arcs, optionally length-weighted."""
    P, T = set(map(int, pred_route)), set(map(int, true_route))
    inter = P & T
    if arc_len is None:
        w = lambda s: float(len(s))
    else:
        w = lambda s: float(sum(arc_len[a] for a in s))
    prec = w(inter) / w(P) if P else 0.0
    rec = w(inter) / w(T) if T else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
