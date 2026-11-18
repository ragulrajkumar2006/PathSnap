# Raw trip data (not committed)

The course dataset is not included in this repository. Copy it here before running the pipeline or the API:

```
data/raw/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          # 80 training trips
├── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   # 10 student-test trips
└── evaluator_test/<trip_id>/...                                # evaluation day only (never used in development)
```

The API serves `evaluator_test/` when it exists, otherwise `student_test/`. Override with `PATHSNAP_TRIPS_DIR=/path`.
