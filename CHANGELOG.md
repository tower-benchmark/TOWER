# Changelog

## Unreleased

### Breaking API changes

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
