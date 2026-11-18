"""Mac 1 / data stage - Interactive Folium maps of raw GPS, road network and reference route.

Run:  python -m mac1.data.folium_maps
Writes results/data/maps/<partition>_<trip>.html
"""
import folium
import numpy as np
from scipy.spatial import cKDTree

from pathsnap_core.config import RESULTS
from pathsnap_core.geo import LocalProjection
from pathsnap_core.io import list_trips, load_trip

OUT = RESULTS / "data" / "maps"
TRIPS = [("train", 0), ("train", 1), ("student_test", 0), ("student_test", 3)]


def arc_lines(trip, arc_ids):
    n = trip.nodes
    return [[(n[a, 1], n[a, 0]), (n[b, 1], n[b, 0])] for a, b in trip.arcs[arc_ids]]


def arcs_near_track(trip, radius_m=150.0):
    """Arc ids whose midpoint lies within radius_m of any GPS fix."""
    proj = LocalProjection.for_track(trip.track)
    tx, ty = proj.to_xy(trip.track[:, 0], trip.track[:, 1])
    nx, ny = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
    a, b = trip.arcs[:, 0], trip.arcs[:, 1]
    mid = np.column_stack([(nx[a] + nx[b]) / 2, (ny[a] + ny[b]) / 2])
    d, _ = cKDTree(np.column_stack([tx, ty])).query(mid, distance_upper_bound=radius_m)
    return np.flatnonzero(np.isfinite(d))


def make_map(trip_dir, partition):
    trip = load_trip(trip_dir)
    tr = trip.track
    m = folium.Map(tiles="OpenStreetMap")
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])

    near = arcs_near_track(trip)
    net = folium.FeatureGroup(name=f"Road network ({len(near)} arcs near track)")
    folium.PolyLine(arc_lines(trip, near), color="#9e9e9e", weight=1, opacity=.6).add_to(net)
    net.add_to(m)

    ref = folium.FeatureGroup(name="Reference route (.route)")
    folium.PolyLine(arc_lines(trip, trip.route), color="#2e7d32", weight=5, opacity=.8).add_to(ref)
    ref.add_to(m)

    gps = folium.FeatureGroup(name=f"Raw GPS fixes ({len(tr)})")
    step = max(1, len(tr) // 3000)
    for lon, lat, t in tr[::step]:
        folium.CircleMarker((lat, lon), radius=2, color="#1565c0", fill=True, fill_opacity=.8,
                            popup=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker((tr[0, 1], tr[0, 0]), tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((tr[-1, 1], tr[-1, 0]), tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ('<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;'
              'border:1px solid #888;font-size:13px">'
              f'<b>{partition} trip {trip.trip_id}</b><br>'
              '<span style="color:#1565c0">&#9679;</span> raw GPS<br>'
              '<span style="color:#2e7d32">&#9644;</span> reference route<br>'
              '<span style="color:#9e9e9e">&#9644;</span> road network</div>')
    m.get_root().html.add_child(folium.Element(legend))
    path = OUT / f"{partition}_{trip.trip_id}.html"
    m.save(path)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for part, idx in TRIPS:
        trips = list_trips(part)
        if idx < len(trips):
            print("wrote", make_map(trips[idx], part))


if __name__ == "__main__":
    main()
