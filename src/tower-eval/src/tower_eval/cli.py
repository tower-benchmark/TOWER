# SPDX-License-Identifier: Apache-2.0
"""Small public CLI for discovery and smoke tests."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import sys
from pathlib import Path

from .constants import _SCHEMAS, ACTION_DT
from .tasks import resolve

# Written before the first episode so --overwrite can also clear an interrupted run.
RUN_MARKER = ".tower-run.json"
RUN_MARKER_SCHEMA = _SCHEMAS["run"]

# Current dataset and legacy local manifests, including the private asset archive.
ASSET_REPOSITORIES = frozenset(
    {
        "tower-benchmark/TOWER-Assets",
        "tower-benchmark/TOWER-Assets-archive",
        "anonymous" + "-review-assets",
    }
)


def _validate_task_assets(config, config_path: Path) -> None:
    """Validate that a configured task has a local HF asset manifest entry."""
    task = str(config.environment.get("task", "tower_transfer_2"))
    manifest_value = config.environment.get("asset_manifest")
    if not manifest_value:
        return
    manifest_path = Path(str(manifest_value))
    if not manifest_path.is_absolute():
        manifest_path = (config_path.resolve().parent / manifest_path).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"asset manifest not found: {manifest_path}")
    document = json.loads(manifest_path.read_text())
    if document.get("repository") not in ASSET_REPOSITORIES:
        raise ValueError(f"asset manifest must target one of {sorted(ASSET_REPOSITORIES)}")
    paths = {str(item.get("path")) for item in document.get("assets", [])}
    required = {"station/station.usdz", f"tasks/{task}.usda"}
    missing = sorted(required - paths)
    if missing:
        raise ValueError(f"asset manifest is missing task assets: {missing}")
    for relative in required:
        local_path = manifest_path.parent / relative
        if not local_path.is_file():
            raise ValueError(f"local task asset is not staged: {local_path}")


def _has_schema(path: Path, schema: str) -> bool:
    if path.is_symlink():
        return False
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(document, dict) and document.get("schema") == schema


def _remove_output_safely(output: Path) -> None:
    """Remove only recognized run outputs, independently of install layout."""
    if output.is_symlink():
        raise ValueError(f"refusing to overwrite a symlink: {output}")
    output = output.resolve()
    cwd = Path.cwd().resolve()
    if output in {cwd, Path.home().resolve(), Path(output.anchor)} or output in cwd.parents:
        raise ValueError("refusing to overwrite cwd, home, a filesystem root or an ancestor of cwd")
    if output.is_dir():
        if not any(output.iterdir()):
            output.rmdir()
            return
        markers = {"summary.json": _SCHEMAS["report"], RUN_MARKER: RUN_MARKER_SCHEMA}
        if not any(_has_schema(output / name, schema) for name, schema in markers.items()):
            raise ValueError(
                f"refusing to overwrite a directory that is not a TOWER run output: {output}; "
                "remove it manually or choose another output path"
            )
        shutil.rmtree(output)
    elif output.exists():
        raise ValueError(f"output path is not a directory: {output}")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tower", description="TOWER robotics benchmark")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("version", help="print the installed evaluator version")
    smoke_parser = subparsers.add_parser("smoke", help="run a CPU-only package smoke test")
    smoke_parser.add_argument("--policy", default="reference")
    config_parser = subparsers.add_parser("config-check", help="validate a TOWER TOML config")
    config_parser.add_argument("path")
    config_parser.add_argument("--set", action="append", default=[], dest="overrides")
    evaluate_parser = subparsers.add_parser(
        "run", aliases=["evaluate"], help="run an evaluation from TOML"
    )
    evaluate_parser.add_argument("--config", required=True, type=Path)
    evaluate_parser.add_argument("--set", action="append", default=[], dest="overrides")
    evaluate_parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output directory"
    )
    from .scoring import cli as scoring_cli

    score_parser = subparsers.add_parser("score", help="score a recording with versioned rules")
    scoring_cli.add_arguments(score_parser)
    args = parser.parse_args(argv)
    if args.command == "score":
        return _score(args, parser)
    handlers = {
        "version": _version,
        "smoke": _smoke,
        "config-check": _config_check,
        "run": _run,
        "evaluate": _run,
    }
    if args.command in handlers:
        return handlers[args.command](args)
    parser.print_help()
    return 0


def _score(args, parser):
    from .scoring import cli as scoring_cli

    return scoring_cli.run(args, parser)


def _version(args):
    from . import __version__

    print(__version__)
    return 0


def _smoke(args):
    if args.policy == "xpolicylab://demo_policy":
        from .policies.smoke import run_xpolicylab_smoke

        print(json.dumps(run_xpolicylab_smoke()))
        return 0
    if args.policy != "reference":
        raise ValueError("smoke --policy supports reference or xpolicylab://demo_policy")
    from .envs import TowerEnv
    from .policies import ReferenceAdapter
    from .runner.episode import run_episode
    from .simulator import CpuReferenceBackend

    policy = ReferenceAdapter([0.0] * 16)
    try:
        result = run_episode(
            TowerEnv(CpuReferenceBackend(), max_steps=3),
            policy,
            seed=0,
            max_steps=3,
        )
    finally:
        policy.close()
    print(json.dumps({"status": "ok", **result}, indent=2))
    return 0


def _config_check(args):
    from .config import load_config

    print(json.dumps(load_config(args.path, args.overrides).to_dict(), default=str, indent=2))
    return 0


def _run(args):
    from . import __version__
    from .config import load_config
    from .envs import TowerEnv
    from .runner.episode import run_episode
    from .simulator import CpuReferenceBackend

    config = load_config(args.config, args.overrides)
    _validate_task_assets(config, args.config)
    backend_name = str(config.environment.get("backend", "cpu"))
    if backend_name not in {"cpu", "cpu-reference", "cpu-smoke", "reference"}:
        raise SystemExit(
            f"backend {backend_name!r} is not included in public v1; use 'cpu'. "
            "Isaac Sim/simulator integration is an optional backend and must be installed separately."
        )
    policy = _load_policy(config.policy)
    task = str(config.environment.get("task", "tower_transfer_2"))
    episodes = []
    max_steps = max(1, math.ceil(config.duration_seconds / ACTION_DT))
    try:
        output = config.output
        if output.exists():
            if not args.overwrite:
                raise SystemExit(
                    f"output already exists: {output} (use --overwrite or choose another output path)"
                )
            _remove_output_safely(output)
        output.mkdir(parents=True)
        from .artifacts import write_json

        write_json(output / RUN_MARKER, {"schema": RUN_MARKER_SCHEMA})
        for index in range(config.episodes):
            episode = run_episode(
                TowerEnv(
                    CpuReferenceBackend(task=task),
                    max_steps=max_steps,
                ),
                policy,
                seed=config.seed + index,
                max_steps=max_steps,
                output=output / "episodes" / f"episode_{index:04d}.json",
                config=config.to_dict(),
            )
            episodes.append(episode)
    except BaseException:
        # Keep the evaluation error; a secondary close failure must not replace it.
        with contextlib.suppress(Exception):
            policy.close()
        raise
    policy.close()
    result = {
        "schema": _SCHEMAS["report"],
        "canonical_task_id": resolve(task).id,
        "episodes": {
            "requested": config.episodes,
            "completed": len(episodes),
            "valid": 0,
            "invalid": len(episodes),
        },
        "metrics": {
            name: None if name == "disturbed_block_count" else 0.0
            for name in (
                "task_success_rate",
                "grasp_success_rate",
                "place_success_rate",
                "long_horizon_progress_rate",
                "disturbed_block_count",
            )
        },
        "episode_reports": [f"episodes/episode_{i:04d}.json" for i in range(len(episodes))],
        "config": config.to_dict(),
        "provenance": {
            "evaluator_version": __version__,
            "backend": "cpu-reference",
            "invalid_reason": "backend does not provide physical task events",
        },
    }
    write_json(output / "summary.json", result)
    print(json.dumps(result, default=str, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"tower: error: {exc}", file=sys.stderr)
        return 2


def _load_policy(configuration):
    from .policies import ReferenceAdapter

    adapter_name = str(configuration.get("adapter", "reference")).lower()
    if adapter_name == "reference":
        policy = ReferenceAdapter([0.0] * 16)
    elif adapter_name == "local":
        import importlib

        from .policies import LocalPolicyAdapter

        entrypoint = str(configuration.get("entrypoint", ""))
        if ":" not in entrypoint:
            raise ValueError("policy.entrypoint must use 'module:attribute' for a local adapter")
        module_name, attribute = entrypoint.split(":", 1)
        candidate = getattr(importlib.import_module(module_name), attribute)
        policy = LocalPolicyAdapter(candidate() if isinstance(candidate, type) else candidate)
    elif adapter_name == "xpolicylab":
        from .policies import XPolicyLabAdapter

        options = configuration.get("config", {})
        policy = XPolicyLabAdapter(
            str(configuration.get("server", configuration.get("url", ""))),
            mode=options.get("action_mode", "joint_absolute"),
            root=options.get("client_root"),
            max_connect_attempts=options.get("max_connect_attempts", 200),
            connect_timeout_s=options.get("connect_timeout_s", 10.0),
            max_connect_seconds=options.get("max_connect_seconds", 180.0),
            timeout_s=float(configuration.get("timeout_s", 120.0)),
        )
    else:
        raise SystemExit("unsupported policy.adapter; choose reference, local, or xpolicylab")
    return policy


if __name__ == "__main__":
    raise SystemExit(main())
