# Isaac Sim GPU setup

The supported benchmark runtime is **Isaac Sim 6.1 / benchmark 2026.11**.
Use Linux x86_64, an NVIDIA RTX GPU, Python 3.12 for Isaac, and the separately
provisioned IK environment. The Python 3.11/3.12 CPU workspace supports CI and
API development; its reference backend does not model tower contacts or task
success.

## Install the benchmark runtime

Follow the [November 2026 guide](reference/bench-public.md) for the
pinned Isaac dependencies, environment checks, independent IK solver, public
asset download, request preparation and evaluation commands. Keep Isaac and
model dependencies out of the lightweight CPU workspace. The versioned entry
point is `tools/simulator/benchmark.py`.

The default runtime is described in [Public benchmark runtime](reference/bench-public.md).
It requires a release manifest and clean tracked files. The guide's in-checkout
`.venv-isaac61`, `.venv-benchmark-ik`, `benchmark-assets` and `benchmark-smoke/`
layout is supported. Untracked or ignored files are allowed outside `src/`,
`packages/`, `simulator/`, `tools/simulator/` and `scripts/`; they are rejected
inside those bound trees. Set `PYTHONDONTWRITEBYTECODE=1` or direct Python's
bytecode cache outside those trees. Interpreter roots must not overlap a bound
tree. Startup checks already loaded imports before GPU work: recorded SDK roots
may be inside the checkout, but unrelated checkout imports remain forbidden.
See the public runtime page for the interpreter and dependency trust boundary.

Review NVIDIA's license before installation and set `OMNI_KIT_ACCEPT_EULA=YES`
yourself only after accepting it. The launcher requires explicit consent before
starting workers. Record the GPU model, runtime and asset identities, policy
checkpoint identity, seed and posture-review evidence with each benchmark run.

## CPU installation check

From the repository root:

```bash
uv sync --locked
uv run --package tower-eval tower smoke
uv run --package tower-eval tower evaluate \
  --config src/tower-eval/configs/smoke.toml
```

An installed wheel can run `tower smoke` without checkout-relative configs.
Use explicit asset paths in your own configs for other CPU integration tests.

## Headless and remote machines

Use Isaac Sim's headless mode on remote RTX hosts and keep policy services
bound to localhost or an authenticated TLS proxy. Model serving lives in
XPolicyLab; see the [policy interface](reference/policy-interface.md).
