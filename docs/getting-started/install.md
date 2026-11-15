# Install

Use Python 3.11 or 3.12 for CPU evaluation tooling. Isaac Sim 6.1 requires a
separate Python 3.12 environment and compatible NVIDIA GPU and drivers.
From the repository root:

```sh
uv sync --locked
uv run --package tower-eval tower smoke
```

A successful smoke run checks package imports and the CPU contract only.
Install the simulator separately following [Isaac setup](../isaac-sim.md).
Obtain assets through the [public benchmark instructions](../guides/evaluate-a-policy.md#download-and-verify-assets)
and verify their locked identity before a physics run. Accept NVIDIA's license
explicitly; installing TOWER does not grant simulator or weight licenses.

`tower-sim` includes Pillow, h5py and PyYAML as runtime dependencies. Its
`evidence` extra is an empty compatibility alias; no extra is needed for
image or HDF5 recording. Isaac and native IK still need separate environments.
