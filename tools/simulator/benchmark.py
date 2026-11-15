# SPDX-License-Identifier: Apache-2.0
"""Inspect and use the versioned November 2026 twelve-task simulator/scorer."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
import tempfile
from contextlib import contextmanager
from pathlib import Path

if "--reference" in sys.argv[1:] or importlib.util.find_spec("tower_eval") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/tower-eval/src"))
from tower_eval.tasks import benchmark_legacy_id
from tower_eval.tasks import resolve as resolve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def task_argument(alias: str) -> str:
    """Keep registry suggestions visible in argparse errors."""
    try:
        return benchmark_legacy_id(alias)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


@contextmanager
def runtime_request(path, request):
    """Pass aliases to the frozen file-reading worker without rewriting input evidence."""
    path = Path(path).resolve()
    if read(path)["task"] == request["task"]:
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="tower-request-") as directory:
        normalized = Path(directory) / "request.json"
        normalized.write_text(json.dumps(request, indent=2) + "\n")
        yield normalized


def require_eula(accept_eula=False, environment=None):
    """Isaac workers start only after the user accepts NVIDIA's Omniverse license."""
    from tower_sim.replay.eula import require_eula as require

    require(accept_eula, environment=os.environ if environment is None else environment)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def refuse_reference():
    raise RuntimeError(
        "The frozen reference runtime is not part of this distribution; "
        "maintainers reproduce it from the private archive."
    )


def posture_evidence(review, task, package_digest):
    """A historical pass or a symmetric flag cannot qualify a new public pilot."""
    posture = review.get("posture_review", {})
    if (
        posture.get("schema") != "tower-no-reversed-grasp-review-v1"
        or posture.get("task") != task
        or posture.get("package_digest") != package_digest
    ):
        raise ValueError("A task/package-bound current posture review is required")
    for field in (
        "planned_approach_carry_passed",
        "recorded_approach_carry_passed",
        "reverse_180_grasp_absent",
        "task_contact_geometry_preserved",
    ):
        if posture.get(field) is not True:
            raise ValueError(f"Posture review has not passed {field}")
    paths = []
    for field in ("planned_trajectory", "recorded_approach_carry"):
        evidence = posture.get(field, {})
        path = Path(evidence.get("path", ""))
        if not path.is_absolute() or not path.is_file() or sha256(path) != evidence.get("sha256"):
            raise ValueError(f"Missing or changed posture evidence: {field}")
        paths.append(path)
    return paths


def check_environment(python):
    """Check SDK metadata and native material plugins without starting Kit."""
    result = subprocess.run(
        [str(python), str(Path(__file__).with_name("isaac_environment_probe.py"))],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Isaac environment check failed:\n{result.stderr.strip()}")
    identity = json.loads(result.stdout)
    if identity["isaacsim"] != "6.1.0.0" or not identity["python"].startswith("3.12."):
        raise ValueError(f"Requires Isaac 6.1.0.0 / Python 3.12; found {identity}")
    # Version metadata is necessary, not sufficient: the request's runtime
    # manifest must also bind the exact installed SDK/source files.
    return identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser(
        "verify", help="CPU-only source and scene inventory verification"
    )
    inventory.add_argument("--assets", type=Path)
    inventory.add_argument("--data-root", type=Path)
    commands.add_parser("describe", help="Print the explicit benchmark settings")
    environment = commands.add_parser(
        "check-environment",
        help="Check Isaac metadata and load native material plugins without GPU use",
    )
    environment.add_argument("--isaac-python", type=Path, required=True)
    score = commands.add_parser("score", help="Score a recording into a new file")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--task", required=True, type=task_argument)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--input-path", choices=("campaign", "files"), default="files")
    score.add_argument("--ruleset", choices=("paper-v1", "bench-2026.11"))
    for name in ("check-request", "run"):
        command = commands.add_parser(name)
        command.add_argument("--request", type=Path, required=True)
        if name == "run":
            command.add_argument("--isaac-python", type=Path, required=True)
            from tower_sim.bench.shader_cache import add_arguments

            add_arguments(command)
            command.add_argument(
                "--accept-eula",
                action="store_true",
                help="Accept the NVIDIA Omniverse License Agreement",
            )
    for command in commands.choices.values():
        command.allow_abbrev = False
        command.add_argument("--allow-skip", action="store_true", help=argparse.SUPPRESS)
        command.add_argument("--benchmark", choices=("2026.11.1",), default="2026.11.1")
        command.add_argument("--v1-root", type=Path, help="Explicit v1 archive export")
        command.add_argument(
            "--reference",
            action="store_true",
            help="Unavailable: the frozen reference runtime is maintained in the private archive",
        )
    args = parser.parse_args(argv)
    if args.command == "score" and (
        args.output.resolve().is_relative_to(args.run.resolve()) or args.output.exists()
    ):
        parser.error("Output must be a new file outside the input recording")
    if args.reference:
        refuse_reference()
    import benchmark_candidate

    from scripts.maintainer.archive_scene import MissingV1Data

    try:
        return benchmark_candidate.command(args)
    except MissingV1Data as error:
        if args.v1_root is not None:
            parser.error(str(error))
        print(json.dumps(dict(passed=False, skipped=True, reason=str(error))))
        return 0 if args.allow_skip else 77


# Maintainer comparison helpers resolve archives only when explicitly called.
def verify_bundle(*args, **kwargs):
    from scripts.maintainer.archived_benchmark import verify_bundle as verify

    return verify(*args, **kwargs)


def load_scorer():
    from scripts.maintainer.archived_benchmark import load_scorer as load

    return load()


def score_run(run, task):
    from scripts.maintainer.archived_benchmark import score_run as score

    return score(run, task)


def preflight_request(path):
    from scripts.maintainer.archived_benchmark import preflight_request as preflight

    return preflight(path)


if __name__ == "__main__":
    raise SystemExit(main())
