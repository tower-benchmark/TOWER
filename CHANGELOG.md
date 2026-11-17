# Changelog

## Unreleased

### Breaking API changes

- Isaac policy transport now requires XPolicyLab WebSockets. The compatibility
  adapter and endpoint timeout options are removed. `policy_url` retains its name
  and requires `ws://` or `wss://`; request schema remains `tower-public-benchmark-v2`.
- Active environments use sequential atomic INFER/audit calls on one connection.
  Metrics cover that exchange; full-horizon response evidence remains exact.
  INFER and get_prediction each have a 300-second request timeout. get_identity
  retains the old health timeout of 30 seconds. A failed exchange permits one
  reconnect attempt with a 180-second connect budget.
- MME-VLA authorization is unavailable pending an XPolicyLab history adapter.
  Benchmark, data, scene, coordinate, camera and hardware contracts are unchanged.

- `EvaluationConfig.controller` is removed; configure the explicit controller interface.
- `GeneratorKinematics` is renamed to `SimulatorKinematics`.
- `GeneratorSimulator` is removed; use the public benchmark runtime.

### Changes

- Completed runs exit zero after a teardown crash only when all evidence checks
  pass; incomplete runs still fail.
- Camera delivery rejects any invalid or all-zero view and any frame with four
  quality-failed views, including in the independent reference render.
- Formal runs write a run-level `result.json` with evidence completeness and
  `worker_exit` details, separating crash phase from evidence validity.
- Public smoke `result.json` is now written by the parent after checking the
  worker's durable completion receipt, preserving teardown crash details.
- Public benchmark revision 2026.11.1 with neutral scene identifiers and a complete
  physical equivalence map for 600 fixed evaluation scenes.
- Evaluation-only `tower-sim`, `tower-eval`, and `nero-kinematics` packages.
- Bound release wheels, calibration data, dependency manifests and runtime evidence.
- Joint/EEF controls, task scoring, camera verification and offline evaluation.
- Includes only fixed-scene evaluation code.
- Ordinary grasp and current approach/carry review requirements remain in force.

### Public benchmark runs

Added full-task and subset benchmark runs, sequential batching and shard
aggregation without mandatory pilot review. Automatic recorded TCP posture
flags are informational. Maintainer qualification remains fail-closed; benchmark
revision, scoring rules, data hashes and package versions are unchanged.
