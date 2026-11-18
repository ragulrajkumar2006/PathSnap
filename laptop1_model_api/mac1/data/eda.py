"""Mac 1 / data stage - Dataset inspection and exploratory data analysis.

Run:  python -m mac1.data.eda
Outputs go to results/data/.
Partitions analysed: train (80 trips) and student_test (10 trips).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap_core.config import RESULTS, ensure_dirs
from pathsnap_core.io import list_trips, load_trip
from pathsnap_core.labels import align_to_route
from pathsnap_core.preprocess import prepare

OUT = RESULTS / "data"


def inspect_trip(path, partition):
    trip = load_trip(path)
    tr = trip.track
    files = {ext: (path / f"{trip.trip_id}.{ext}").stat().st_size for ext in ("track", "nodes", "arcs", "route")
             if (path / f"{trip.trip_id}.{ext}").exists()}
    pt = prepare(trip)
    x, y = pt.proj.to_xy(tr[:, 0], tr[:, 1])
    dt = np.diff(tr[:, 2])
    step = np.hypot(np.diff(x), np.diff(y))
    speed = step / np.where(dt > 0, dt, np.nan)

    V, E = len(trip.nodes), len(trip.arcs)
    A = coo_matrix((np.ones(E), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(V, V))
    n_weak, _ = connected_components(A, directed=True, connection="weak")
    n_strong, lab_strong = connected_components(A, directed=True, connection="strong")
    deg = np.bincount(trip.arcs.ravel(), minlength=V)
    largest_scc = np.bincount(lab_strong).max() / V

    route = trip.route
    r_conn = float(np.mean(trip.arcs[route[:-1], 1] == trip.arcs[route[1:], 0])) if len(route) > 1 else 1.0
    lab, ldist, pos = align_to_route(pt, route)

    return {
        "partition": partition, "trip_id": trip.trip_id,
        **{f"bytes_{k}": v for k, v in files.items()},
        "n_fixes": len(tr), "n_nan": int(np.isnan(tr).sum()),
        "n_dropped_clean": int((~pt.keep).sum()),
        "duration_s": float(tr[-1, 2] - tr[0, 2]), "t_start": float(tr[0, 2]),
        "dt_median": float(np.median(dt)), "dt_max": float(dt.max()),
        "pct_dt_1s": float(np.mean(dt == 1)), "n_gaps_gt10s": int(np.sum(dt > 10)),
        "n_gaps_gt60s": int(np.sum(dt > 60)), "n_nonincreasing_t": int(np.sum(dt <= 0)),
        "speed_p50": float(np.nanmedian(speed)), "speed_max": float(np.nanmax(speed)),
        "n_speed_gt85": int(np.nansum(speed > 85)),
        "length_km": float(np.nansum(step) / 1000),
        "lon_min": tr[:, 0].min(), "lon_max": tr[:, 0].max(),
        "lat_min": tr[:, 1].min(), "lat_max": tr[:, 1].max(),
        "n_nodes": V, "n_arcs": E, "n_isolated_nodes": int(np.sum(deg == 0)),
        "n_weak_components": int(n_weak), "n_strong_components": int(n_strong),
        "largest_scc_frac": float(largest_scc),
        "route_len": len(route), "route_unique_frac": len(set(route)) / len(route),
        "route_connected_frac": r_conn,
        "route_coverage_by_fixes": len(np.unique(pos)) / len(route),
        "label_dist_median": float(np.median(ldist)), "label_dist_p95": float(np.percentile(ldist, 95)),
        "pct_fixes_far_from_route": float(np.mean(ldist > 60)),
        "_dt": dt, "_speed": speed[np.isfinite(speed)], "_ldist": ldist,
    }


def plots(df, extra):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for p, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        d = df[df.partition == p]
        ax[0, 0].hist(d.n_fixes, bins=30, alpha=.6, label=p, color=c)
        ax[0, 1].hist(d.length_km, bins=30, alpha=.6, label=p, color=c)
        ax[0, 2].hist(np.log10(d.n_arcs), bins=30, alpha=.6, label=p, color=c)
    ax[0, 0].set(title="GPS fixes per trip", xlabel="fixes")
    ax[0, 1].set(title="Trip length", xlabel="km")
    ax[0, 2].set(title="Road-network size", xlabel="log10(arcs)")
    dt = np.concatenate(extra["dt"])
    ax[1, 0].hist(np.clip(dt, 0, 15), bins=np.arange(0, 16), color="gray")
    ax[1, 0].set(title="Sampling interval (train+student_test, clipped at 15 s)", xlabel="seconds", yscale="log")
    sp = np.concatenate(extra["speed"])
    ax[1, 1].hist(np.clip(sp, 0, 120), bins=60, color="gray")
    ax[1, 1].axvline(85, color="red", ls="--", label="glitch threshold")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set(title="Implied speed between fixes", xlabel="m/s", yscale="log")
    ld = np.concatenate(extra["ldist"])
    ax[1, 2].hist(np.clip(ld, 0, 100), bins=50, color="gray")
    ax[1, 2].axvline(60, color="red", ls="--", label="label validity radius")
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].set(title="Distance of fix to its aligned reference arc", xlabel="m", yscale="log")
    for a in ax[0]:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "eda_distributions.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    cx = (df.lon_min + df.lon_max) / 2
    cy = (df.lat_min + df.lat_max) / 2
    for p, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        m = df.partition == p
        ax.scatter(cx[m], cy[m], s=25, label=p, color=c)
    ax.set(title="Trip locations (centre of GPS extent)", xlabel="longitude", ylabel="latitude",
           xlim=(-180, 180), ylim=(-90, 90))
    ax.grid(alpha=.3)
    ax.legend()
    fig.savefig(OUT / "eda_trip_locations.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(df.n_nodes, df.n_arcs, c=(df.partition == "train").map({True: "tab:blue", False: "tab:orange"}))
    ax[0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Network size per trip")
    ax[1].hist(df.largest_scc_frac, bins=20, color="tab:green")
    ax[1].set(title="Share of nodes in largest strongly connected component", xlabel="fraction")
    fig.tight_layout()
    fig.savefig(OUT / "eda_network_connectivity.png", dpi=130)
    plt.close(fig)


def plot_trajectory(trip_dir, fname):
    trip = load_trip(trip_dir)
    tr, nodes, arcs = trip.track, trip.nodes, trip.arcs
    pad = 0.003
    m = ((nodes[arcs[:, 0], 0] > tr[:, 0].min() - pad) & (nodes[arcs[:, 0], 0] < tr[:, 0].max() + pad) &
         (nodes[arcs[:, 0], 1] > tr[:, 1].min() - pad) & (nodes[arcs[:, 0], 1] < tr[:, 1].max() + pad))
    fig, ax = plt.subplots(figsize=(8, 8))
    for a, b in arcs[m]:
        ax.plot(nodes[[a, b], 0], nodes[[a, b], 1], color="lightgray", lw=.6, zorder=1)
    for e in trip.route:
        a, b = arcs[e]
        ax.plot(nodes[[a, b], 0], nodes[[a, b], 1], color="green", lw=2.5, zorder=2)
    ax.scatter(tr[:, 0], tr[:, 1], s=3, c=tr[:, 2], cmap="viridis", zorder=3)
    ax.set(title=f"Trip {trip.trip_id}: network (grey), reference route (green), GPS (coloured by time)",
           xlabel="longitude", ylabel="latitude")
    ax.set_aspect(1 / np.cos(np.radians(tr[:, 1].mean())))
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def main():
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(p, part) for part in ("train", "student_test") for p in list_trips(part)]
    rows = Parallel(n_jobs=-1)(delayed(inspect_trip)(p, part) for p, part in jobs)
    extra = {k: [r.pop(f"_{k}") for r in rows] for k in ("dt", "speed", "ldist")}
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dataset_inventory.csv", index=False)

    summary = df.groupby("partition").agg(
        trips=("trip_id", "count"), fixes=("n_fixes", "sum"), fixes_mean=("n_fixes", "mean"),
        nan_values=("n_nan", "sum"), dropped_by_cleaning=("n_dropped_clean", "sum"),
        gaps_gt60s=("n_gaps_gt60s", "sum"), max_gap_s=("dt_max", "max"),
        pct_dt_1s=("pct_dt_1s", "mean"), arcs_mean=("n_arcs", "mean"),
        isolated_nodes=("n_isolated_nodes", "sum"),
        route_connected=("route_connected_frac", "mean"),
        far_from_route=("pct_fixes_far_from_route", "mean")).round(4)
    summary.to_csv(OUT / "partition_summary.csv")
    missing = pd.DataFrame({
        "check": ["NaN values in .track", "non-increasing timestamps", "speed glitches > 85 m/s",
                  "gaps > 60 s", "isolated nodes", "fixes > 60 m from reference route"],
        "count": [df.n_nan.sum(), df.n_nonincreasing_t.sum(), df.n_speed_gt85.sum(),
                  df.n_gaps_gt60s.sum(), df.n_isolated_nodes.sum(),
                  int((df.pct_fixes_far_from_route * df.n_fixes).sum())]})
    missing.to_csv(OUT / "data_quality_checks.csv", index=False)

    plots(df, extra)
    plot_trajectory(list_trips("train")[0], "trajectory_train_00000001.png")
    plot_trajectory(list_trips("student_test")[0], "trajectory_student_test_00000000.png")

    anomalies = df[(df.dt_max > 300) | (df.n_speed_gt85 > 0) | (df.pct_fixes_far_from_route > 0.05)]
    anomalies[["partition", "trip_id", "n_fixes", "dt_max", "n_gaps_gt60s", "n_speed_gt85",
               "pct_fixes_far_from_route", "route_coverage_by_fixes"]].to_csv(OUT / "anomalous_trips.csv", index=False)

    print(summary.T)
    print("\nData quality checks:\n", missing.to_string(index=False))
    print(f"\nAnomalous trips: {len(anomalies)} (see anomalous_trips.csv)")
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
