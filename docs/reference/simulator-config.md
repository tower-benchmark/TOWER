# Simulator configuration and DTOs

`from tower_sim import SimConfig` provides the evaluation configuration. Its
fields preserve the original simulation defaults and types. `SimConfig.from_v1`
reads historical JSON, retains known legacy values for the original v1
validation, and rejects unknown keys. `to_dict()` emits simulation fields only.

`tower_eval.simulator.spec` is the pure simulator DTO layer: `Block`, `Transfer`,
`EpisodeSpec`, `State`, and simulator-facing protocols. These types retain the v1
JSON field names and are not exported by `tower_eval` at top level.

These simulator DTOs are distinct from the public evaluation data types in
`tower_eval.model` (including its `Block`) and the task registry's
`tower_eval.tasks.Task`. Use the full module path when referring to the
simulator protocols or scene specifications. Historical JSON is not renamed.
