"""Public shader-cache options, validated before starting a worker."""

from pathlib import Path


def add_arguments(parser):
    parser.add_argument(
        "--shader-cache-seed", type=Path, help="Local GPU/driver cache seed directory"
    )
    parser.add_argument("--shader-cache-digest", type=Path, help="Seed digest_tree file")
    parser.add_argument(
        "--shader-cache-warmup", type=Path, help="Create a cache with one serial start"
    )


def child_arguments(options):
    if options is None:
        return []
    seed = getattr(options, "shader_cache_seed", None)
    digest = getattr(options, "shader_cache_digest", None)
    warmup = getattr(options, "shader_cache_warmup", None)
    if bool(seed) != bool(digest) or (warmup and seed):
        raise ValueError("Supply both shader-cache seed and digest, or warmup alone")
    result = []
    for name, path in (("seed", seed), ("digest", digest), ("warmup", warmup)):
        if path is not None:
            result.extend(["--shader-cache-" + name, str(Path(path).resolve())])
    return result


def configuration(options, output):
    child_arguments(options)
    warmup = getattr(options, "shader_cache_warmup", None)
    seed = getattr(options, "shader_cache_seed", None)
    if warmup:
        path = Path(warmup).resolve()
        return dict(seed=None, seed_sha=None, private_root=path.parent, name=path.name, warmup=True)
    if seed:
        return dict(
            seed=Path(seed).resolve(),
            seed_sha=Path(options.shader_cache_digest).resolve(),
            private_root=Path(output).parent,
            name=Path(output).name + "-shader-cache",
        )
    return None
