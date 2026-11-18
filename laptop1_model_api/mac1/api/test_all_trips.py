"""Mac 1 / Call the running API for every available trip and record the results.

Run (API must be running):  python -m mac1.api.test_all_trips [http://host:8000]
Writes results/api/api_test_results.csv
"""
import sys
import time

import pandas as pd
import requests

from pathsnap_core.config import RESULTS


def main(api):
    trips = requests.get(f"{api}/trips", timeout=10).json()
    rows = []
    for tid in trips["trips"]:
        t0 = time.time()
        r = requests.post(f"{api}/predict", json={"trip_id": tid}, timeout=600)
        r.raise_for_status()
        j = r.json()
        rows.append({"trip_id": tid, "fixes": j["n_fixes"], "route_arcs": len(j["predicted_route_arc_ids"]),
                     "round_trip_s": time.time() - t0, **(j["metrics"] or {})})
        print(f"{tid}: top1 {rows[-1].get('top1_accuracy_hmm', float('nan')):.4f} "
              f"route F1 {rows[-1].get('route_f1', float('nan')):.4f} ({rows[-1]['round_trip_s']:.1f}s)")
    df = pd.DataFrame(rows)
    out = RESULTS / "api"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"api_test_results_{trips['partition']}.csv", index=False)
    print(df.drop(columns=["trip_id"]).mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
