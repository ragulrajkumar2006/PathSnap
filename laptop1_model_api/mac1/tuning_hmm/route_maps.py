"""Mac 1 / tuning + HMM stage - Folium maps comparing GPS, reference route and predicted routes.

Run (after hmm_experiments):  python -m mac1.tuning_hmm.route_maps
Writes results/tuning_hmm/maps/student_test_<trip>.html for all 10 student-test trips.
"""
import folium
import pandas as pd

from pathsnap_core.config import RESULTS
from pathsnap_core.io import list_trips, load_trip
from pathsnap_core.preprocess import prepare
from mac1.data.folium_maps import arc_lines, arcs_near_track

OUT = RESULTS / "tuning_hmm" / "maps"
LAYERS = [  # (mode in predictions file, label, colour, weight, dash)
    ("classifier_only", "Classifier-only (argmax per fix)", "#e53935", 3, "6,6"),
    ("hmm_classifier_route", "Classifier + HMM/Viterbi route", "#1e88e5", 5, None),
]


def make(trip_dir, preds, metrics):
    trip = load_trip(trip_dir)
    pt = prepare(trip)
    tr = trip.track[pt.keep]
    m = folium.Map(tiles="OpenStreetMap")
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])

    net = folium.FeatureGroup(name="Road network (near track)", show=False)
    folium.PolyLine(arc_lines(trip, arcs_near_track(trip)), color="#9e9e9e", weight=1, opacity=.6).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup(name="Reference route")
    folium.PolyLine(arc_lines(trip, trip.route), color="#43a047", weight=8, opacity=.5).add_to(ref)
    ref.add_to(m)
    for mode, label, color, w, dash in LAYERS:
        arcs = preds[preds["mode"] == mode].sort_values("obs").pred_arc.to_numpy()
        arcs = [a for a in arcs if a >= 0]
        if mode == "classifier_only":   # collapse repeats, keep order
            arcs = [a for i, a in enumerate(arcs) if i == 0 or a != arcs[i - 1]]
        fg = folium.FeatureGroup(name=label)
        folium.PolyLine(arc_lines(trip, arcs), color=color, weight=w, opacity=.9, dash_array=dash).add_to(fg)
        fg.add_to(m)
    gps = folium.FeatureGroup(name="GPS fixes")
    step = max(1, len(tr) // 2000)
    for lon, lat, t in tr[::step]:
        folium.CircleMarker((lat, lon), radius=2, color="#212121", fill=True, popup=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    rows = "".join(f"<tr><td>{r['mode']}</td><td>{r['top1_accuracy']:.3f}</td><td>{r['route_continuity']:.3f}</td>"
                   f"<td>{r['route_f1']:.3f}</td></tr>" for _, r in metrics.iterrows())
    legend = ('<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;'
              'border:1px solid #888;font-size:12px">'
              f'<b>student_test trip {trip.trip_id}</b><br>'
              '<span style="color:#212121">&#9679;</span> GPS fixes &nbsp;'
              '<span style="color:#43a047">&#9644;</span> reference &nbsp;'
              '<span style="color:#e53935">- -</span> classifier only &nbsp;'
              '<span style="color:#1e88e5">&#9644;</span> classifier + HMM'
              '<table style="margin-top:4px"><tr><th>mode</th><th>Top-1</th><th>continuity</th><th>route F1</th></tr>'
              f'{rows}</table></div>')
    m.get_root().html.add_child(folium.Element(legend))
    path = OUT / f"student_test_{trip.trip_id}.html"
    m.save(path)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(RESULTS / "tuning_hmm" / "predictions_student_test.parquet")
    metrics = pd.read_csv(RESULTS / "tuning_hmm" / "hmm_student_test_per_trip.csv", dtype={"trip_id": str})
    for p in list_trips("student_test"):
        print("wrote", make(p, preds[preds.trip_id == p.name], metrics[metrics.trip_id == p.name]))


if __name__ == "__main__":
    main()
