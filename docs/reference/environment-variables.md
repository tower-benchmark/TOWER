# Environment variables

The evaluator reads TOML configuration and `--set key=value` overrides.
Environment variables do not expand command bounds or observation tolerance.

`TOWER_REQUIRE_NATIVE_IK=1` requires native IK in tests instead of skipping it;
`TOWER_NERO_URDF` supplies the test model. Production IK takes explicit model
arguments. Native numerical libraries use `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` at their launch boundary.

Simulator license consent is explicit. See [Isaac setup](../isaac-sim.md) for
its environment and [configuration](config.md) for supported runtime settings.


## Simulator and launch boundaries

| Variable | Meaning |
| --- | --- |
| `OMNI_KIT_ACCEPT_EULA` | Accepts `YES`, `Y` or `1`, case-insensitive. Set only after accepting NVIDIA's terms. |
| `TOWER_ISAAC_PYTHON` | `tools/run_eef_validation.py` default for `--isaac-python`; explicitly provisioned interpreter. |
| `TOWER_IK_PREFIX` | Same tool's default for `--native-prefix`; explicit native IK environment. |
| `TOWER_ASSET_ROOT` | Asset location at legacy launch boundaries, not a `tower run` setting. Public construction passes verified resource paths explicitly. |
| `TOWER_SOURCE_ROOT` | Does not override installed package data or default paths. |
| `PYTHONDONTWRITEBYTECODE` | Set to `1` before wheel builds to avoid bytecode in bound source trees. Benchmark entry points also set `sys.dont_write_bytecode = True`. |
| `PYTHONPYCACHEPREFIX` | If bytecode is enabled elsewhere, direct its cache outside bound source trees. |
| `TOWER_SDK_ROOT`, `TOWER_IK_PYTHON` | SDK prefix and native IK interpreter supplied by the benchmark launcher. |
| `CUDA_VISIBLE_DEVICES` | Native launch boundaries manage device visibility; benchmark bootstrap removes it inside its scoped process environment. Not a CPU runner setting. |
