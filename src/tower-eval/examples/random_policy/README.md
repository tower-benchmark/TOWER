# Random CPU model

`Model` uses XPolicyLab's `update_obs` / `get_action` / `reset` lifecycle and
standard dual-arm joint dictionaries. `LocalPolicyAdapter(Model())` requires
complete TOWER encoder observations and produces tagged encoder-frame actions.
The generic CPU backend cannot execute those commands.

`CpuSmokePolicy` is an explicit test fixture: it supplies synthetic camera/encoder
observations and labels this model's small random joint values as physical CPU
commands. It is not an executable robot policy or benchmark result.

```python
from tower_eval.examples.random_policy.model import CpuSmokePolicy
from tower_eval.envs import TowerEnv
from tower_eval.evaluation import evaluate
from tower_eval.simulator import CpuReferenceBackend

policy = CpuSmokePolicy()
try:
    result = evaluate(TowerEnv(CpuReferenceBackend(), max_steps=3), policy, seed=42)
finally:
    policy.close()
```

For a CPU `tower evaluate` configuration, the local entrypoint is
`tower_eval.examples.random_policy.model:CpuSmokePolicy`. Installed wheels include
this example. For the external-server loopback, run
`tower smoke --policy xpolicylab://demo_policy` from a submodule checkout.
