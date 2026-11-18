"""HMM + Viterbi route reconstruction.

Hidden states  : the candidate arcs of each GPS fix (state = (fix i, arc a)).
Emission       : log P(fix | arc) = w_e * log(classifier probability), or a
                 Gaussian on GPS-to-arc distance for the classifier-free baseline.
Transition     : Newson & Krumm (2009) style. With d_route the network distance
                 from the projected position on arc a to the projected position
                 on arc b, and d_gc the straight-line distance between the fixes,
                     log P(b | a) = -|d_route - d_gc| / (beta * sqrt(dt))
                 Unreachable pairs get -inf.
Breaks         : when no transition into fix i is reachable (disconnected
                 candidates, very large gap), the chain restarts at fix i from
                 the emissions alone and the break is counted.
Missing cands  : fixes without candidates are skipped and get arc -1.
"""
import heapq
from dataclasses import dataclass

import numpy as np

from .preprocess import PreparedTrip

NEG = -1e18


@dataclass
class HMMParams:
    beta: float = 5.0             # transition scale (m)
    emission_weight: float = 1.0  # weight on log classifier probability
    sigma: float = 10.0           # used only for the distance-only emission
    max_search_m: float = 5000.0  # Dijkstra bound
    back_tol_m: float = 15.0      # tolerated backward motion on the same arc (GPS noise)
    eps: float = 1e-6


class RoadGraph:
    """Directed road graph with bounded, cached Dijkstra from a node."""

    def __init__(self, pt: PreparedTrip):
        arcs = pt.trip.arcs
        self.arcs = arcs
        self.arc_len = pt.arc_len
        order = np.argsort(arcs[:, 0], kind="stable")
        self.adj_arc = order
        self.adj_ptr = np.searchsorted(arcs[order, 0], np.arange(len(pt.nx) + 1))
        self._cache: dict[int, tuple[float, dict, dict]] = {}

    def dists_from(self, src: int, bound: float):
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= bound:
            return hit[1]
        dist = {src: 0.0}
        pred = {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > bound:
                continue
            for k in range(self.adj_ptr[u], self.adj_ptr[u + 1]):
                e = self.adj_arc[k]
                v = self.arcs[e, 1]
                nd = d + self.arc_len[e]
                if nd < dist.get(v, np.inf) and nd <= bound:
                    dist[v] = nd
                    pred[v] = e
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (bound, dist, pred)
        return dist

    def path_arcs(self, src: int, dst: int, bound: float):
        """Arc ids on the shortest path src -> dst, or None if unreachable."""
        self.dists_from(src, bound)
        _, dist, pred = self._cache[src]
        if dst not in dist:
            return None
        out = []
        v = dst
        while v != src:
            e = pred[v]
            out.append(int(e))
            v = self.arcs[e, 0]
        return out[::-1]


def _route_dist(g: RoadGraph, a, sa, b, sb, bound, tol):
    if a == b and sb >= sa - tol:
        return max(0.0, sb - sa)
    d = g.dists_from(int(g.arcs[a, 1]), bound).get(int(g.arcs[b, 0]))
    if d is None:
        return None
    return (g.arc_len[a] - sa) + d + sb


def viterbi_decode(pt: PreparedTrip, df, score_col="p", params: HMMParams = HMMParams(),
                   emission="classifier", graph: RoadGraph | None = None):
    """Decode one trip. df holds the trip's candidate rows (obs, arc, frac, dist, score).

    Returns dict with pred_arc (per fix, -1 if no candidate), route (stitched arc
    list) and n_breaks.
    """
    g = graph or RoadGraph(pt)
    df = df.sort_values(["obs", score_col], ascending=[True, False])
    groups = [(int(o), grp) for o, grp in df.groupby("obs", sort=True)]
    pred = np.full(pt.n, -1, dtype=np.int64)
    if not groups:
        return {"pred_arc": pred, "route": [], "n_breaks": 0}

    def emis(grp):
        if emission == "classifier":
            return params.emission_weight * np.log(grp[score_col].to_numpy() + params.eps)
        return -0.5 * (grp["dist"].to_numpy() / params.sigma) ** 2

    obs_list, arcs_list, back_list, delta_list = [], [], [], []
    prev_obs, prev_arcs, prev_s, prev_delta = None, None, None, None
    n_breaks = 0
    for o, grp in groups:
        a_cur = grp["arc"].to_numpy()
        s_cur = grp["frac"].to_numpy() * pt.arc_len[a_cur]
        e = emis(grp)
        if prev_obs is None:
            delta = e
            back = np.full(len(a_cur), -1)
        else:
            gc = float(np.hypot(pt.px[o] - pt.px[prev_obs], pt.py[o] - pt.py[prev_obs]))
            dt = max(float(pt.t[o] - pt.t[prev_obs]), 1.0)
            bound = min(params.max_search_m, max(3.0 * gc, gc + 300.0))
            scale = params.beta * np.sqrt(dt)
            trans = np.full((len(prev_arcs), len(a_cur)), NEG)
            for i, (a, sa) in enumerate(zip(prev_arcs, prev_s)):
                for j, (b, sb) in enumerate(zip(a_cur, s_cur)):
                    d = _route_dist(g, a, sa, b, sb, bound, params.back_tol_m)
                    if d is not None:
                        trans[i, j] = -abs(d - gc) / scale
            score = prev_delta[:, None] + trans
            back = score.argmax(0)
            best = score[back, np.arange(len(a_cur))]
            if np.all(best <= NEG / 2):
                n_breaks += 1
                delta = e
                back = np.full(len(a_cur), -1)
            else:
                delta = np.where(best <= NEG / 2, NEG, best) + e
        obs_list.append(o)
        arcs_list.append(a_cur)
        back_list.append(back)
        delta_list.append(delta)
        prev_obs, prev_arcs, prev_s, prev_delta = o, a_cur, s_cur, delta

    # backtrack, restarting at every chain break
    j = int(np.argmax(prev_delta))
    for k in range(len(obs_list) - 1, -1, -1):
        pred[obs_list[k]] = arcs_list[k][j]
        bj = back_list[k][j]
        if bj < 0 and k > 0:
            # chain break: the previous segment ends at its own best state
            j = int(np.argmax(delta_list[k - 1]))
        else:
            j = int(bj)
    return {"pred_arc": pred, "route": stitch_route(g, pred, params.max_search_m),
            "n_breaks": n_breaks}


def stitch_route(g: RoadGraph, pred_arc, bound):
    """Collapse per-fix arcs into a travel sequence and fill connection gaps
    with shortest paths so the output route is drivable where possible."""
    seq = [int(a) for a in pred_arc if a >= 0]
    route = []
    for a in seq:
        if route and route[-1] == a:
            continue
        if route:
            prev = route[-1]
            if g.arcs[prev, 1] != g.arcs[a, 0]:
                fill = g.path_arcs(int(g.arcs[prev, 1]), int(g.arcs[a, 0]), bound)
                if fill:
                    route.extend(fill)
        route.append(a)
    return route
