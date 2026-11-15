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
        "--shader-cache-warmup",
        type=Path,
        help="Export compiled cache files and the serial start status",
    )


def child_arguments(options):
    if options is None:
        return []
    seed = getattr(options, "shader_cache_seed", None)
    digest = getattr(options, "shader_cache_digest", None)
    warmup = getattr(options, "shader_cache_warmup", None)
    if bool(seed) != bool(digest):
        raise ValueError("Supply both shader-cache seed and digest")
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
    if not warmup and not seed:
        return None
    config = dict(
        seed=Path(seed).resolve() if seed else None,
        seed_sha=Path(options.shader_cache_digest).resolve() if seed else None,
        private_root=output.parent,
        name=output.name + "-kit",
        portable=True,
    )
    if warmup:
        config.update(warmup=True, warmup_destination=Path(warmup).resolve())
    return config


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
    target = directory / "cache"
    if receipt.get("private_cache") != str(target):
        raise ValueError("Shader cache receipt differs from worker configuration")
    if config.get("warmup"):
        from .app import digest_tree

        # Compilation output is useful to a second warm-up pass even when the
        # scene failed. Preserve that failure; cache bytes never qualify a run.
        destination = Path(config["warmup_destination"])
        destination.mkdir(parents=True, exist_ok=False)
        exported = destination / "cache"
        shutil.copytree(target, exported)
        (destination / "digest.sha256").write_text(digest_tree(exported))
        (destination / "warmup.json").write_text(
            json.dumps(dict(source_run_passed=bool(passed)), indent=2) + "\n"
        )
    else:
        shutil.rmtree(target, ignore_errors=True)
