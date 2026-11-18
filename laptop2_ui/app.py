"""Laptop 2 / PathSnap evaluator UI.

Start:  streamlit run laptop2_ui/app.py -- --api http://<LAPTOP1_IP>:8000
(the API URL can also be changed in the sidebar)

The UI only talks to the API: it never loads the model or any reference route.
"""
import argparse
import math
import sys

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

parser = argparse.ArgumentParser()
parser.add_argument("--api", default="http://localhost:8000")
args, _ = parser.parse_known_args(sys.argv[1:])

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: map-matching and route reconstruction")

api = st.sidebar.text_input("API URL", value=st.session_state.get("api", args.api)).rstrip("/")
st.session_state["api"] = api
show_clf = st.sidebar.checkbox("Also show classifier-only path", value=False)

try:
    health = requests.get(f"{api}/health", timeout=5).json()
    trips = requests.get(f"{api}/trips", timeout=5).json()
    st.sidebar.success(f"Connected: model {health['model']}")
    st.sidebar.caption(f"HMM: beta={health['hmm_params']['beta']}, "
                       f"emission weight={health['hmm_params']['emission_weight']}")
except Exception as e:
    st.error(f"Cannot reach API at {api}: {e}")
    st.stop()

st.caption(f"{len(trips['trips'])} trips available ({trips['partition']})")
trip_id = st.selectbox("Select a trip", trips["trips"])

if st.button("Reconstruct route", type="primary"):
    with st.spinner(f"Sending trip {trip_id} to the API..."):
        r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
    else:
        st.session_state["result"] = r.json()

res = st.session_state.get("result")
if res:
    m = res["metrics"]
    st.subheader(f"Trip {res['trip_id']}")
    if m:
        c = st.columns(5)
        c[0].metric("Top-1 accuracy (HMM)", f"{m['top1_accuracy_hmm']:.1%}",
                    f"{m['top1_accuracy_hmm'] - m['top1_accuracy_classifier_only']:+.1%} vs classifier only")
        c[1].metric("Route continuity", f"{m['route_continuity_hmm']:.1%}")
        c[2].metric("Route F1", f"{m['route_f1']:.1%}")
        c[3].metric("Route precision / recall", f"{m['route_precision']:.0%} / {m['route_recall']:.0%}")
        c[4].metric("API processing time", f"{res['timings']['total_s']:.1f} s")
    else:
        st.info("No reference route on the server for this trip; showing prediction only.")
    st.caption(f"{res['n_fixes']} GPS fixes used ({res['n_dropped_fixes']} dropped as glitches), "
               f"{len(res['predicted_route'])} predicted route arcs, {res['n_breaks']} decoder chain breaks")

    gps = res["gps"]
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    center = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]
    # st_folium ignores fit_bounds, so derive a Web-Mercator zoom that fits the
    # GPS extent into a ~1200x600 px map (latitude span is stretched by 1/cos(lat)).
    dlon = max(max(lons) - min(lons), 1e-4)
    dlat = max(max(lats) - min(lats), 1e-4) / math.cos(math.radians(center[0]))
    zoom = int(min(math.log2(1200 * 360 / (256 * dlon)), math.log2(600 * 360 / (256 * dlat))))
    zoom = max(3, min(zoom, 17))
    fmap = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
    g = folium.FeatureGroup(name="GPS fixes")
    for lat, lon in gps:
        folium.CircleMarker((lat, lon), radius=2, color="#212121", fill=True).add_to(g)
    g.add_to(fmap)
    if show_clf and res["classifier_only_route"]:
        c = folium.FeatureGroup(name="Classifier-only path")
        folium.PolyLine(res["classifier_only_route"], color="#e53935", weight=3, dash_array="6,6").add_to(c)
        c.add_to(fmap)
    p = folium.FeatureGroup(name="Predicted route (classifier + HMM)")
    if res["predicted_route"]:
        folium.PolyLine(res["predicted_route"], color="#1e88e5", weight=5, opacity=.9).add_to(p)
    p.add_to(fmap)
    folium.Marker(gps[0], tooltip="start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="end", icon=folium.Icon(color="red")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    st_folium(fmap, center=center, zoom=zoom, height=600, use_container_width=True, returned_objects=[])

    with st.expander("Timings and full metrics"):
        st.json({"timings": res["timings"], "metrics": m})
