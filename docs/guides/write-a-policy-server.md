# Policy adapters

Policies implement `tower_eval.PolicyAdapter`:

```python
def reset(*, seed=None, metadata=None): ...
def act(observation, info=None): ...
def close(): ...
```

`act` may return one action or `tower_eval.ActionChunk(actions, dt_s, metadata)`.
The evaluator executes the supplied chunk up to the episode limit and checks its
timestep. TOWER owns this lightweight type; no serving package is required.

`LocalPolicyAdapter` wraps callable/`act` policies and XPolicyLab-style models
with `update_obs`, `get_action` and argument-free `reset`. It uses the same
encoder observation mapping, action decoder and exact anchor validation as the
WebSocket client. Supply all camera/state fields; raw CPU `qpos` is insufficient.
Model-style joint
actions use standard XPolicyLab dictionaries and are decoded into joint16
encoder endpoints; specify `action_mode="eef_delta"` for tagged TOWER encoder
EEF targets. `prepare_case` receives the episode seed when the model supports it.
See the [random CPU model](../../src/tower-eval/examples/random_policy/README.md).

Remote models run through `XPolicyLabAdapter`:

```toml
[policy]
adapter = "xpolicylab"
server = "ws://localhost:19000"
timeout_s = 120
```

The [TOWER ↔ XPolicyLab contract](../reference/policy-interface.md) defines required
modalities, frames, action modes, batching and deployment. TOWER contains no
policy server. `LegacyPredictAdapter` supports the HTTP
`/health` and `/predict` protocol for compatible servers, with independent
30 s / 300 s defaults. Health authorizes the package digest and prediction pins
the complete health identity. See the reference above for configuration and
error classification.


See the [policy interface](../reference/policy-interface.md) for units, frames and lifecycle.
