# Architecture

TOWER owns evaluation, scoring, simulation, robot control and result records.
XPolicyLab owns policy loading and serving in a separate environment.

`tower-eval` supplies configuration, task identity, policy adapters and scoring.
`tower-sim` supplies the Isaac runtime. `nero-kinematics` supplies IK/FK and
explicit control profiles. Simulator imports remain lazy so CPU installation
and protocol checks do not require the simulator SDK.

The CPU reference backend has no tower contacts and cannot establish physical
task success. Benchmark results require the Isaac runtime and pinned assets.
See [installation](../getting-started/install.md) and
[observations and actions](observation-and-action.md).
