# Multiple GPUs

Run independent evaluations with separate output directories and explicitly
assigned devices. Do not share a mutable policy session between episodes.
Keep the same runtime, asset and checkpoint identities across workers and
aggregate only complete results. Use `--gpu` per request; device allocation is the caller’s responsibility.
