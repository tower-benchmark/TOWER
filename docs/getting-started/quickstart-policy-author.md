# Policy author quickstart

Install [TOWER](install.md), then initialize the pinned serving integration:

```sh
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init third_party/XPolicyLab
uv sync --locked --package tower-eval --extra xpolicylab
uv run --package tower-eval tower smoke --policy xpolicylab://demo_policy
```

This CPU loopback starts and stops the demo server and checks request/response
handling. Install your model dependencies in the separate policy environment.
Implement the [observation and action contract](../reference/policy-interface.md),
then follow [write a policy server](../guides/write-a-policy-server.md).
Declare action mode, units and frames explicitly; never infer them from shapes.
