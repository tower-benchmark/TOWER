# Offline scoring

Run the scorer on a complete saved recording, writing to a new output file:

```sh
tower score /path/to/run --task lateral-insertion-2 --ruleset bench-2026.11 --input-path files --output /path/to/new-score.json
```

Keep the input recording unchanged. The file loader validates and hashes its
recording inputs; reports identify the ruleset, input path, evaluator version
and canonical task id. Missing or incomplete evidence must not be reported as
successful evaluation. See [scoring reference](../reference/scoring.md).

Each run needs `episode_spec.json`, `status.json`, `episode/states.hdf5`,
`finger-link-poses.npz`, `finger-feedback.npz` and `finger_geometry.json`,
including each finger’s `distal_link_corners_m`. Timestamps must align and gaps
cannot exceed 100 ms. Inputs are opened read-only. Existing automatic events
and human ratings are never detector inputs.
