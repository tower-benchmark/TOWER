# Command line

Use `tower --help`, `tower smoke --help`, `tower run --help` and
`tower score --help` in the installed evaluator environment. `tower smoke` is
asset-free; `tower run` accepts a TOML configuration. Benchmark physics commands
are documented in the [public runtime guide](bench-public.md).

## Multi-scene benchmark

`tools/simulator/benchmark_run.py run` evaluates a task with the Isaac interpreter.
Supply `--task`, `--assets`, `--package`, `--policy-url`, `--ik-python`, `--output`
and `--accept-eula`; `--data-root`, `--gpu` and shader-cache flags match the
portable runner. `--scenes` defaults to `0-49` and accepts comma-separated ids
and ranges. `--batch-size` defaults to 1 and is limited to 10. `--keep-going`
continues after failed batches without retrying. `--pilot-review` is optional
maintainer qualification.

`benchmark_run.py aggregate --runs DIR... --output DIR` combines disjoint runs
without Isaac. Duplicate scenes and mixed identities or qualifications are errors.
