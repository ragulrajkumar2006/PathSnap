"""Mac 1 / data stage - Build the candidate feature datasets used by every later stage.

Run:  python -m mac1.data.build_dataset
Writes
  data/processed/candidates_{train,student_test}.parquet   one row per (fix, candidate arc)
  data/processed/fixes_{train,student_test}.parquet        one row per cleaned GPS fix
  results/data/candidate_stats.csv                      candidate miss rate etc. per trip
"""
import time

import pandas as pd
from joblib import Parallel, delayed

from pathsnap_core.config import DATA_PROCESSED, RESULTS, ensure_dirs
from pathsnap_core.io import list_trips
from pathsnap_core.pipeline import fix_table, process_trip


def one(path, partition):
    t0 = time.time()
    pt, cand, labels = process_trip(path)
    fixes = fix_table(pt, labels)
    hit = cand[cand.label == 1].obs.unique()
    valid = labels["valid"]
    hit_mask = pd.Series(False, index=range(pt.n))
    hit_mask[hit] = True
    stats = {
        "partition": partition, "trip_id": pt.trip.trip_id, "fixes": pt.n,
        "fixes_valid_label": int(valid.sum()),
        "fixes_without_candidates": int(pt.n - cand.obs.nunique()),
        "candidates_per_fix": len(cand) / pt.n,
        "candidate_miss_rate": float(1 - hit_mask[valid].mean()),
        "seconds": time.time() - t0,
    }
    return cand.assign(partition=partition), fixes.assign(partition=partition), stats


def main():
    ensure_dirs()
    out_stats = RESULTS / "data"
    out_stats.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for part in ("train", "student_test"):
        res = Parallel(n_jobs=-1)(delayed(one)(p, part) for p in list_trips(part))
        cand = pd.concat([r[0] for r in res], ignore_index=True)
        fixes = pd.concat([r[1] for r in res], ignore_index=True)
        cand.to_parquet(DATA_PROCESSED / f"candidates_{part}.parquet", index=False)
        fixes.to_parquet(DATA_PROCESSED / f"fixes_{part}.parquet", index=False)
        all_stats += [r[2] for r in res]
        print(f"{part}: {fixes.trip_id.nunique()} trips, {len(fixes)} fixes, {len(cand)} candidate rows, "
              f"positive rate {cand.label.mean():.3f}")
    st = pd.DataFrame(all_stats)
    st.to_csv(out_stats / "candidate_stats.csv", index=False)
    print(st.groupby("partition")[["candidates_per_fix", "candidate_miss_rate", "fixes_without_candidates"]].mean())


if __name__ == "__main__":
    main()
