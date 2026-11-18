"""Feature engineering for (GPS fix, candidate arc) pairs.

Every feature is computed from GPS geometry and the road network only. Nothing
here touches the .route file, so the same code runs unchanged inside the API.
"""
import numpy as np
import pandas as pd

from .config import CONTEXT_WINDOW
from .geo import angle_diff, bearing, point_segment
from .preprocess import PreparedTrip

FEATURES = [
    "dist", "rank", "dist_minus_min", "dist_ratio", "n_cand",
    "frac", "at_endpoint", "arc_len",
    "heading_diff", "heading_cos", "moving", "speed", "dt_prev",
    "win_dist_mean", "win_dist_max", "win_heading_diff",
    "link_prev", "link_next", "has_twin",
    "deg_out", "deg_in",
]


def _gps_heading(px, py, w):
    n = len(px)
    lo = np.clip(np.arange(n) - w, 0, n - 1)
    hi = np.clip(np.arange(n) + w, 0, n - 1)
    dx, dy = px[hi] - px[lo], py[hi] - py[lo]
    return bearing(dx, dy), np.hypot(dx, dy)


def make_features(pt: PreparedTrip, cand: dict) -> pd.DataFrame:
    obs, arc, dist = cand["obs"], cand["arc"], cand["dist"]
    n = pt.n
    arcs = pt.trip.arcs
    W = CONTEXT_WINDOW

    # per-fix quantities
    head, disp = _gps_heading(pt.px, pt.py, 2)
    step = np.r_[0.0, np.hypot(np.diff(pt.px), np.diff(pt.py))]
    dt = np.r_[1.0, np.diff(pt.t)]
    speed = step / np.maximum(dt, 1e-6)

    start = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    counts = np.diff(np.r_[start, len(obs)])
    min_d = np.full(n, np.nan)
    min_d[obs[start]] = dist[start]
    n_cand = np.zeros(n, dtype=int)
    n_cand[obs[start]] = counts
    nearest = np.full(n, -1)
    nearest[obs[start]] = arc[start]

    abear = bearing(pt.bx - pt.ax, pt.by - pt.ay)
    hdiff = angle_diff(head[obs], abear[arc])
    moving = disp[obs] > 3.0
    hdiff = np.where(moving, hdiff, 90.0)

    # windowed geometry: how well do neighbouring fixes fit the same arc?
    wd = []
    wh = []
    for o in range(-W, W + 1):
        j = np.clip(obs + o, 0, n - 1)
        d, _ = point_segment(pt.px[j], pt.py[j], pt.ax[arc], pt.ay[arc], pt.bx[arc], pt.by[arc])
        wd.append(d)
        if o != 0:
            wh.append(np.where(disp[j] > 3.0, angle_diff(head[j], abear[arc]), 90.0))
    wd = np.vstack(wd)
    wh = np.vstack(wh)

    # connectivity with neighbours' nearest arc
    np_ = nearest[np.clip(obs - 1, 0, n - 1)]
    nn_ = nearest[np.clip(obs + 1, 0, n - 1)]
    link_prev = (np_ == arc) | ((np_ >= 0) & (arcs[np.maximum(np_, 0), 1] == arcs[arc, 0]))
    link_next = (nn_ == arc) | ((nn_ >= 0) & (arcs[arc, 1] == arcs[np.maximum(nn_, 0), 0]))

    # reverse twin (u->v and v->u both candidates of the same fix)
    key = pd.Series(list(zip(obs, arcs[arc, 0], arcs[arc, 1])))
    twin_key = set(zip(obs, arcs[arc, 1], arcs[arc, 0]))
    has_twin = key.map(lambda k: k in twin_key).to_numpy()

    deg_out = np.bincount(arcs[:, 0], minlength=len(pt.nx))
    deg_in = np.bincount(arcs[:, 1], minlength=len(pt.nx))

    return pd.DataFrame({
        "obs": obs, "arc": arc,
        "dist": dist, "rank": cand["rank"], "dist_minus_min": dist - min_d[obs],
        "dist_ratio": dist / (min_d[obs] + 1.0), "n_cand": n_cand[obs],
        "frac": cand["frac"], "at_endpoint": ((cand["frac"] <= 0) | (cand["frac"] >= 1)).astype(int),
        "arc_len": pt.arc_len[arc],
        "heading_diff": hdiff, "heading_cos": np.cos(np.radians(hdiff)), "moving": moving.astype(int),
        "speed": speed[obs], "dt_prev": dt[obs],
        "win_dist_mean": wd.mean(0), "win_dist_max": wd.max(0), "win_heading_diff": wh.mean(0),
        "link_prev": link_prev.astype(int), "link_next": link_next.astype(int),
        "has_twin": has_twin.astype(int),
        "deg_out": deg_out[arcs[arc, 1]], "deg_in": deg_in[arcs[arc, 0]],
    })
