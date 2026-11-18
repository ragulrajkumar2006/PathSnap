"""Candidate road-segment generation.

Arcs are densified into sample points every DENSIFY_STEP_M metres and stored in
a KD-tree. For each GPS fix we query the tree within radius + step/2, collect the
arcs that were hit, compute the exact point-to-segment distance and keep the
CAND_MAX_K nearest arcs within CAND_RADIUS_M.
"""
import numpy as np
from scipy.spatial import cKDTree

from .config import CAND_RADIUS_M, CAND_MAX_K, DENSIFY_STEP_M
from .geo import point_segment
from .preprocess import PreparedTrip


def build_candidates(pt: PreparedTrip, radius=CAND_RADIUS_M, k=CAND_MAX_K):
    """Return dict of flat arrays, one row per (fix, candidate arc)."""
    # only arcs near the track's bounding box are indexed
    pad = radius + DENSIFY_STEP_M
    xmin, xmax = pt.px.min() - pad, pt.px.max() + pad
    ymin, ymax = pt.py.min() - pad, pt.py.max() + pad
    near = ~((np.maximum(pt.ax, pt.bx) < xmin) | (np.minimum(pt.ax, pt.bx) > xmax) |
             (np.maximum(pt.ay, pt.by) < ymin) | (np.minimum(pt.ay, pt.by) > ymax))
    arc_ids = np.flatnonzero(near)

    nseg = np.maximum(1, np.ceil(pt.arc_len[arc_ids] / DENSIFY_STEP_M).astype(int))
    owner = np.repeat(arc_ids, nseg + 1)
    frac = np.concatenate([np.linspace(0, 1, s + 1) for s in nseg]) if len(arc_ids) else np.empty(0)
    sx = pt.ax[owner] + frac * (pt.bx[owner] - pt.ax[owner])
    sy = pt.ay[owner] + frac * (pt.by[owner] - pt.ay[owner])
    tree = cKDTree(np.column_stack([sx, sy]))

    hits = tree.query_ball_point(np.column_stack([pt.px, pt.py]), r=radius + DENSIFY_STEP_M / 2)
    obs, arc = [], []
    for i, h in enumerate(hits):
        if h:
            u = np.unique(owner[h])
            obs.append(np.full(len(u), i))
            arc.append(u)
    if not obs:
        return {k_: np.empty(0) for k_ in ("obs", "arc", "dist", "frac")}
    obs = np.concatenate(obs)
    arc = np.concatenate(arc)
    dist, t = point_segment(pt.px[obs], pt.py[obs], pt.ax[arc], pt.ay[arc], pt.bx[arc], pt.by[arc])

    m = dist <= radius
    obs, arc, dist, t = obs[m], arc[m], dist[m], t[m]
    order = np.lexsort((dist, obs))
    obs, arc, dist, t = obs[order], arc[order], dist[order], t[order]
    # rank within each observation, keep top-k
    start = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    rank = np.arange(len(obs)) - np.repeat(start, np.diff(np.r_[start, len(obs)]))
    m = rank < k
    return {"obs": obs[m], "arc": arc[m], "dist": dist[m], "frac": t[m], "rank": rank[m]}
