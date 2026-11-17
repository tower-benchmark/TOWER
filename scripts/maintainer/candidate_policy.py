"""Explicit opt-in learned-policy entry. Never starts an inference server.

The frozen public entry is unchanged. Use `run --request candidate.json` under a
detached supervisor; --diagnostic produces non-formal evidence only. The worker
uses the same construction/provenance and physics lifecycle as command replay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

__tower_entry_sha256__ = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tower_sim.bench import policy_entry
from tower_sim.bench.fast_observed_joints import observed_joint_validator
from tower_sim.bench.ik_client import IKServiceOptions
from tower_sim.bench.j2comp import J2Compensation
from tower_sim.bench.metrics import Metrics, TrialRequestID
from tower_sim.bench.options import ControlOptions
from tower_sim.bench.policy_ik import PolicyIK
from tower_sim.bench.resources import Resources, save, save_exclusive
from tower_sim.bench.tasks import BenchTask

from scripts.maintainer.candidate_replay import (
    CandidateRunner,
    JsonSink,
    build_identity,
    data_identity,
    runtime_profile,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/simulator"))
benchmark = importlib.import_module("benchmark")

ROOT = Path(__file__).resolve().parents[2]


def worker(args, *, verification=None, runner=None):
    from tower_sim.bench.verification import ReleaseVerification

    if verification is not None and type(verification) is not ReleaseVerification:
        raise ValueError("Formal verification must be a public ReleaseVerification instance")
    if args.private_kit_root:
        from scripts.maintainer.kit_safety import configure

        configure(args.private_kit_root)
    benchmark_assets = importlib.import_module("benchmark_assets")

    request = json.loads(args.request.read_text())
    if request.get("command_replay", False):
        raise ValueError("Policy entrypoint cannot execute reference-replay requests")
    if verification is None:
        benchmark.verify_bundle()
    else:
        verification.verify()
    output = Path(request["output"]).resolve()
    if output.exists() and (not args.diagnostic or set(output.iterdir()) != {output / "inputs"}):
        raise ValueError("Worker output already contains evidence")
    output.mkdir(parents=True, exist_ok=args.diagnostic)
    resources = Resources.from_roots(checkout=ROOT, assets=Path(request["assets"]))
    task = BenchTask.from_resources(request["task"], resources)
    if args.diagnostic:
        duration = request["duration"]
        if not 0 < duration <= task.deadline_s or round(duration * 400) != duration * 400:
            raise ValueError("Diagnostic duration must be a positive bounded physics tick")
        task = replace(task, deadline_s=duration)
    rows = benchmark_assets.load_rows(Path(request["manifest"]))
    policy_entry.validate_entry(request, rows, task)
    if verification is None:
        commit, wheels, dependencies, imports = build_identity(ROOT)
        identity = dict(
            commit=commit,
            wheel_sha256=wheels,
            dependency_sha256=dependencies,
            package_import_paths=imports,
            immutable_data=data_identity(ROOT, Path(request["assets"])),
            runtime=runtime_profile(request["gpu"]),
            diagnostic_only=args.diagnostic,
        )
    else:
        identity = dict(verification.identity, diagnostic_only=args.diagnostic)
    save(output / "implementation-identity.json", identity)
    sink = JsonSink(
        output / "metrics.jsonl"
        if args.diagnostic
        else output.with_name(output.name + ".metrics.jsonl")
    )
    metrics = Metrics(sink, TrialRequestID(request["trial_id"]))
    solver = PolicyIK(IKServiceOptions(args.ik_python))
    prepared = session = None
    completed = False
    try:
        session, evaluation = policy_entry.prepare(
            rows,
            task,
            request,
            metrics=metrics,
            solver=solver,
            diagnostic=args.diagnostic,
            benchmark_protocol=resources.read("protocol"),
            verification=verification,
            client_root=ROOT / "third_party/XPolicyLab",
        )
        setup = SimpleNamespace(
            task=request["task"],
            assets=Path(request["assets"]),
            output=output,
            gpu=request["gpu"],
            frozen_components=(),
        )
        prepared = (runner or CandidateRunner()).setup(setup, rows=rows, task=task)
        if args.private_kit_root:
            from scripts.maintainer.kit_safety import verify

            verify(args.private_kit_root)
        context = prepared.context
        if verification is not None:
            context.public_provenance = verification.provenance
        context.compensation = J2Compensation.from_cache(
            resources.path("scene_cache"),
            resources.path("scene"),
        )
        validator = observed_joint_validator(
            context.compensation, options=ControlOptions(observed_joints="fast")
        )
        code = policy_entry.execute(
            context,
            rows,
            task,
            worker=session,
            evaluation=evaluation,
            metrics=metrics,
            source_collector=prepared.collect_sources,
            observed_validator=validator,
        )
        if code:
            raise RuntimeError(f"Candidate policy worker failed: {code}")
        save(
            output / "fast-observed-joints.json",
            dict(
                certificate=getattr(validator, "certificate", None),
                stats=getattr(validator, "stats", None),
            ),
        )
        completed = True
    except BaseException:
        save(output / "error.json", dict(error=traceback.format_exc()))
        raise
    finally:
        if session is not None:
            session.close()
        solver.close()
        metrics.close()
        sink.close()
        try:
            if completed:
                # Publish the durable boundary only after all evidence writers close.
                # A native crash before this point must remain an execution failure.
                save_exclusive(
                    output / "worker-complete.json",
                    dict(completed=True, phase="teardown", gain_checks=context._drive_checks),
                )
        finally:
            if prepared is not None:
                prepared.close()


def completion_state(request):
    """The durable formal completion contract, independent of teardown exit code."""
    output = Path(request["output"])

    def read(path):
        return json.loads(path.read_text()) if path.is_file() else {}

    completion = read(output / "worker-complete.json")
    boundary = completion.get("completed") is True and completion.get("phase") == "teardown"
    completed = (
        boundary
        and read(output / "simulation/batch-policy-results.json").get("completed")
        and bool(request["seeds"])
    )
    for seed in request["seeds"]:
        status = read(output / "simulation" / f"seed-{seed}" / "status.json")
        completed = completed and status.get("valid_trial") and status.get("evidence_complete")
    return boundary, completed


def finish(request, returncode, *, diagnostic):
    """Run outside Kit: exit zero during exception cleanup is not completion."""
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    if diagnostic:
        task = BenchTask.from_resources(
            request["task"], Resources.from_roots(checkout=ROOT, assets=Path(request["assets"]))
        )
        completed = []
        results = []
        for seed in request["seeds"]:
            result = policy_entry.collect_diagnostic(
                output,
                request["task"],
                seed,
                returncode,
                bench_task=task,
                duration=request.get("duration"),
            )
            results.append(result)
            completed.append(result["evidence_complete"])
        save_exclusive(
            output / "diagnostic-result.json",
            results[0] if len(results) == 1 else dict(results=results),
        )
        return 0 if completed and all(completed) else 1

    boundary, completed = completion_state(request)
    save_exclusive(
        output / "result.json",
        dict(
            evidence_complete=bool(completed),
            worker_returncode=returncode,
            worker_exit=policy_entry.worker_exit(returncode, boundary=boundary),
        ),
    )
    return 0 if completed else (returncode or 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "_worker"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--ik-python", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--allow-desktop-gpu", action="store_true")
    parser.add_argument("--private-kit-root", type=Path)
    args = parser.parse_args(argv)
    if not args.diagnostic:
        raise ValueError(
            "Formal runs require a ReleaseVerification through the public entrypoint; use --diagnostic here."
        )
    if args.operation == "_worker":
        worker(args)
        return 0
    import fcntl

    from scripts.maintainer.gpu_baseline import gpu_idle

    request = json.loads(args.request.read_text())
    if request.get("command_replay", False):
        raise ValueError("Policy entrypoint cannot execute reference-replay requests")
    if Path(request["output"]).exists():
        raise ValueError("Diagnostic output must be a new directory")
    gpu = request["gpu"]
    lock = Path(
        os.environ.get("TOWER_GPU_LOCK", "/run/lock/tower-v3-gpu{gpu}.lock").format(gpu=gpu)
    )
    descriptor = os.open(lock, os.O_RDONLY | os.O_CREAT, 0o666)
    with os.fdopen(descriptor) as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.allow_desktop_gpu:
            from scripts.maintainer.gpu_admission import admit_desktop

            admit_desktop(gpu, args.request.with_suffix(".admission.json"))
        else:
            gpu_idle(gpu, False)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_worker",
            "--request",
            str(args.request.resolve()),
            "--ik-python",
            str(args.ik_python),
        ]
        if args.diagnostic:
            command.append("--diagnostic")
        if args.private_kit_root:
            command.extend(["--private-kit-root", str(args.private_kit_root.resolve())])
        result = subprocess.run(command, check=False)
    return finish(request, result.returncode, diagnostic=args.diagnostic)


if __name__ == "__main__":
    raise SystemExit(main())
