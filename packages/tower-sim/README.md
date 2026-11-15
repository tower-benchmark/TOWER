# tower-sim

Evaluation control, kinematics, camera capture, and evidence for the TOWER
benchmark. See [public runtime](../../docs/reference/bench-public.md),
[configuration](../../docs/reference/simulator-config.md), and
[camera contracts](../../docs/reference/bench-cameras.md).

`SimConfig.from_v1` reads historical configuration data.
Shared replay helpers preserve the recorded-command and calibration contracts.
Installed package data is byte-bound to the release. Native simulator imports
remain lazy; the independent IK service starts only when explicitly requested.

The command limits, J2 conversion, observation tolerance and hardware protections
are distinct. Reject reverse grasps without fallback, preserve task contact
geometry, and inspect planned and recorded approach/carry before accepting a benchmark run.
See the [control and review contract](https://github.com/tower-benchmark/TOWER/blob/main/docs/reference/bench-public.md#control-and-review-contract).
