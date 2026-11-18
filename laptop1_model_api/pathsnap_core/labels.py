"""Per-fix ground-truth labels from the .route file.

The .route file lists arcs in travel order but does not say which GPS fix lies on
which arc. We align fixes to route positions with a monotone dynamic programme:

    D[i, j] = cost(i, j) + min_{j' <= j} D[i-1, j']

where cost(i, j) is the distance from fix i to route arc j. The vehicle can only
move forward along the route (or stay on the same arc), and it may skip several
short arcs between two fixes. Back-tracking gives one route arc per fix.
"""
import numpy as np

from .geo import point_segment
from .preprocess import PreparedTrip


def align_to_route(pt: PreparedTrip, route: np.ndarray):
    """Return (label_arc, label_dist, route_pos) for every kept fix."""
    n, m = pt.n, len(route)
    ax, ay, bx, by = pt.ax[route], pt.ay[route], pt.bx[route], pt.by[route]
    back = np.empty((n, m), dtype=np.int32)
    prev = None
    for i in range(n):
        c, _ = point_segment(pt.px[i], pt.py[i], ax, ay, bx, by)
        if prev is None:
            cur = c
            back[i] = np.arange(m)
        else:
            # running prefix minimum of the previous row and where it came from
            pm_idx = np.zeros(m, dtype=np.int64)
            better = np.r_[True, prev[1:] < np.minimum.accumulate(prev)[:-1]]
            pm_idx = np.maximum.accumulate(np.where(better, np.arange(m), 0))
            cur = c + prev[pm_idx]
            back[i] = pm_idx
        prev = cur
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    lab = route[pos]
    d, _ = point_segment(pt.px, pt.py, pt.ax[lab], pt.ay[lab], pt.bx[lab], pt.by[lab])
    return lab, d, pos
