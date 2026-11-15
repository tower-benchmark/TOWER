# Configuration

The evaluator accepts TOML and dotted `--set key=value` overrides. Validate with
`tower config-check /path/to/config.toml` before running. Simulator
configuration is documented in [simulator configuration](simulator-config.md); use the released
configuration without changing physics or command bounds.

Use `tower_sim.SimConfig` for evaluation. `from_v1()` accepts known legacy
JSON keys and rejects unknown keys; `to_dict()` emits simulation fields only.
Public construction passes verified resource paths explicitly, including
calibration inputs. Installed package data supplies default paths;
`TOWER_SOURCE_ROOT` does not override them.
