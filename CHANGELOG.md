# Changelog

## Unreleased

### Breaking API changes

- Isaac policy transport now requires XPolicyLab WebSockets. The compatibility
  adapter and endpoint timeout options are removed. `policy_url` retains its name
  and requires `ws://` or `wss://`; request schema remains `tower-public-benchmark-v2`.
- Active environments use sequential atomic INFER/audit calls on one connection.
  Metrics cover that exchange; full-horizon response evidence remains exact.
  The effective prediction timeout remains 300 seconds, matching the previous transport.
- MME-VLA authorization is unavailable pending an XPolicyLab history adapter.
  Benchmark, data, scene, coordinate, camera and hardware contracts are unchanged.

- `EvaluationConfig.controller` is removed; configure the explicit controller interface.
- `GeneratorKinematics` is renamed to `SimulatorKinematics`.
- `GeneratorSimulator` is removed; use the public benchmark runtime.

### Changes

- Public benchmark revision 2026.11.1 with neutral scene identifiers and a complete
  physical equivalence map for 600 fixed evaluation scenes.
- Evaluation-only `tower-sim`, `tower-eval`, and `nero-kinematics` packages.
- Bound release wheels, calibration data, dependency manifests and runtime evidence.
- Joint/EEF controls, task scoring, camera verification and offline evaluation.
- Includes only fixed-scene evaluation code.
- Ordinary grasp and current approach/carry review requirements remain in force.
