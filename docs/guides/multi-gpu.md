# Multiple GPUs

Run independent evaluations with separate output directories and explicitly
assigned devices. Do not share a mutable policy session between episodes.
Keep the same runtime, asset and checkpoint identities across workers and
aggregate only complete results. Use `--gpu` per request; device allocation is the caller’s responsibility.

## Benchmark shards

Run `benchmark_run.py run` separately on each GPU with disjoint `--scenes`
selections, `--gpu` and fresh output directories. Batches within each process
are sequential. Use `benchmark_run.py aggregate --runs DIR... --output DIR`
to combine shards; duplicate scenes or mixed qualifications are rejected.
See [run the benchmark](evaluate-a-policy.md#run-the-benchmark).

Keep complete run bundles. Runtime, release, data, package-file and checkpoint
identities must match. Duplicate scenes remain errors even when one attempt was
excluded. Report each failed/retried attempt separately; replacement is not supported.
