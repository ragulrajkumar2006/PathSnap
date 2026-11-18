"""Cleaning and projection of a raw trip."""
from dataclasses import dataclass

import numpy as np

from .config import MAX_SPEED_MPS
from .geo import LocalProjection
from .io import Trip


@dataclass
class PreparedTrip:
    trip: Trip
    proj: LocalProjection
    keep: np.ndarray        # boolean mask over raw track rows that survived cleaning
    px: np.ndarray          # projected x of kept fixes (m)
    py: np.ndarray
    t: np.ndarray           # timestamps of kept fixes (s)
    nx: np.ndarray          # projected node coordinates
    ny: np.ndarray
    ax: np.ndarray          # arc start / end coordinates
    ay: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    arc_len: np.ndarray

    @property
    def n(self):
        return len(self.px)


def clean_track(track: np.ndarray, proj: LocalProjection) -> np.ndarray:
    """Return a keep-mask. Drops non-finite rows, out-of-range coordinates,
    non-increasing timestamps and isolated speed spikes."""
    lon, lat, t = track[:, 0], track[:, 1], track[:, 2]
    keep = np.isfinite(track).all(axis=1) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
    x, y = proj.to_xy(lon, lat)
    last = None
    for i in range(len(track)):
        if not keep[i]:
            continue
        if last is not None:
            dt = t[i] - t[last]
            if dt <= 0:
                keep[i] = False
                continue
            if np.hypot(x[i] - x[last], y[i] - y[last]) / dt > MAX_SPEED_MPS:
                keep[i] = False
                continue
        last = i
    return keep


def prepare(trip: Trip) -> PreparedTrip:
    finite = np.isfinite(trip.track).all(axis=1)
    proj = LocalProjection.for_track(trip.track[finite])
    keep = clean_track(trip.track, proj)
    tr = trip.track[keep]
    px, py = proj.to_xy(tr[:, 0], tr[:, 1])
    nx, ny = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
    a, b = trip.arcs[:, 0], trip.arcs[:, 1]
    ax, ay, bx, by = nx[a], ny[a], nx[b], ny[b]
    return PreparedTrip(trip, proj, keep, px, py, tr[:, 2], nx, ny, ax, ay, bx, by,
                        np.hypot(bx - ax, by - ay))
