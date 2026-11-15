"""Public shader-cache options, validated before starting a worker."""

import json
import shutil
from pathlib import Path
from typing import Any


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


def configuration(options, output) -> dict[str, Any] | None:
    child_arguments(options)
    output = Path(output).resolve()
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


def finalize(options, output, *, passed):
    """Finalize after worker exit: native fast shutdown can bypass Python finally."""
    config = configuration(options, output)
    if config is None:
        return
    receipt_path = Path(output) / "simulation/private-shader-cache.json"
    if not receipt_path.is_file():
        return  # No proof that this worker created the directory.
    receipt = json.loads(receipt_path.read_text())
    directory = Path(config["private_root"]) / config["name"]
    target = directory / "nv_shadercache"
    if receipt.get("private_cache") != str(target):
        raise ValueError("Shader cache receipt differs from worker configuration")
    if config.get("warmup"):
        if passed:
            from .app import digest_tree

            (directory / "digest.sha256").write_text(digest_tree(target))
    else:
        shutil.rmtree(directory, ignore_errors=True)
