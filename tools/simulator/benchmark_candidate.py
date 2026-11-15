"""Public benchmark entry with pinned scene revisions and release identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
from pathlib import Path
from types import SimpleNamespace

from tower_sim.bench.ik_identity import shared_spec_identity
from tower_sim.bench.resources import Resources

__tower_entry_sha256__ = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

import benchmark
import benchmark_assets
from tower_sim.bench.data_release import manifest as data_manifest
from tower_sim.bench.resources import save_exclusive, sha
from tower_sim.bench.scene_format import DEFAULT_BENCHMARK, safe_path
from tower_sim.bench.verification import (
    ReleaseVerification,
    envelope,
    package_identity,
    verify_data,
    verify_release,
    verify_startup,
)
from tower_sim.bench.verification import (
    read as read_bound,
)

ROOT = benchmark.ROOT
sys.path.insert(0, str(ROOT))
METRICS_ENV = (
    "TOWER_BASELINE_METRICS",
    "TOWER_BASELINE_TRIAL",
    "TOWER_BASELINE_RELEASE",
    "TOWER_BASELINE_TRACE",
)


def protocol_path(checkout, benchmark=DEFAULT_BENCHMARK, *, v1_root=None):
    if benchmark == "2026.11.0":
        from scripts.maintainer.archive_scene import protocol_path as archive_protocol

        return archive_protocol(checkout, v1_root=v1_root)
    return safe_path(checkout, data_manifest()["protocol"])


def child_environment():
    return {key: value for key, value in os.environ.items() if key not in METRICS_ENV}


def create_runner(*, benchmark_version=DEFAULT_BENCHMARK, data_root=None, v1_root=None):
    """Shared public construction factory; replay callers still require the harness."""
    from scripts.maintainer.candidate_replay import CandidateRunner

    return CandidateRunner(
        public=True, benchmark_version=benchmark_version, data_root=data_root, v1_root=v1_root
    )


def verify(*, benchmark_version=DEFAULT_BENCHMARK, assets=None, data_root=None, v1_root=None):
    result = dict(
        verify_data(
            ROOT, benchmark=benchmark_version, assets=assets, data_root=data_root, v1_root=v1_root
        ),
        packages=package_identity(),
    )
    if assets is not None:
        result["assets"] = benchmark_assets.verify(assets)
    result.pop("files")
    return result


def preflight(path, *, diagnostic=False):
    request = benchmark.read(path)
    if request.get("command_replay"):
        raise ValueError("Expected a learned-policy request for a frozen task")
    request["task"] = benchmark.benchmark_legacy_id(request.get("task"))
    benchmark_version = request.get("public", {}).get("benchmark", DEFAULT_BENCHMARK)
    v1_root = request.get("v1_root")
    verify(
        benchmark_version=benchmark_version,
        assets=request["assets"],
        data_root=request.get("data_root"),
        v1_root=v1_root,
    )
    release = verify_release(
        request["release_manifest"], checkout=ROOT, expected=request["release_manifest_sha256"]
    )
    verify_startup(release)
    if request.get("public") != envelope(release, benchmark=benchmark_version):
        raise ValueError("Request is not a public v2 request for this release; prepare it again")
    task = benchmark.read(protocol_path(ROOT, benchmark_version, v1_root=v1_root))["tasks"][
        request["task"]
    ]
    if request["duration"] != task["duration_s"] or request["success_rule"] != task["success_rule"]:
        raise ValueError("Request does not use the frozen deadline/scorer")
    if request["prompt"] != task["prompt"]:
        raise ValueError("Request prompt differs from benchmark")
    output = Path(request["output"])
    if not output.is_absolute() or (
        output.exists() and not (diagnostic and set(output.iterdir()) == {output / "inputs"})
    ):
        raise ValueError(
            "A new absolute output directory is required; never reuse an archived trial"
        )
    seeds = request["seeds"]
    if not 1 <= len(seeds) <= 16 or len(set(seeds)) != len(seeds):
        raise ValueError("Expected 1..16 unique seeds")
    for field in ("release_manifest", "dependency_manifest", "ik_dependency_manifest", "ik_bounds"):
        if sha(request[field]) != request[field + "_sha256"]:
            raise ValueError(f"Changed request binding: {field}")
    spec = Resources.from_roots(checkout=benchmark.ROOT, assets=benchmark.ROOT).path(
        "shared_ik_spec"
    )
    spec_identity = shared_spec_identity()
    if sha(request["ik_bounds"]) != sha(spec):
        raise ValueError("Request changes the shared IK bounds")
    if not Path(request["ik_python"]).is_file():
        raise ValueError("Missing independent IK Python environment")
    body = benchmark.read(request["manifest"])
    if "benchmark" not in body:
        raise ValueError("Scene manifest must declare its benchmark revision")
    if Path(body["assets"]).resolve() != Path(request["assets"]).resolve():
        raise ValueError("Scene manifest uses a different asset root")
    if (
        body["benchmark"] != benchmark_version
        or body.get("data_root") != request.get("data_root")
        or body.get("v1_root") != v1_root
    ):
        raise ValueError("Scene manifest differs from request revision/data root")
    rows = benchmark_assets.load_rows(request["manifest"])
    if [(r["manifest_entry"]["task"], r["candidate"]["seed"]) for r in rows] != [
        (request["task"], seed) for seed in seeds
    ]:
        raise ValueError("Portable scene selection differs from request")
    if not diagnostic:
        settings = read_bound(request["protocol"], request["protocol_sha256"])
        if settings["ik"]["shared_spec_sha256"] != spec_identity:
            raise ValueError("Protocol changes the canonical shared IK identity")
        policy = benchmark.read(protocol_path(ROOT, benchmark_version, v1_root=v1_root))["policy"]
        for key, value in policy.items():
            if settings["policy"].get("hz" if key == "control_hz" else key) != value:
                raise ValueError(f"Policy contract differs: {key}")
        review = settings["pilot_review"]
        posture = benchmark.posture_evidence(
            read_bound(review["path"], review["sha256"]),
            request["task"],
            benchmark.read(request["package"])["package_digest"],
        )
        files = read_bound(request["runtime_manifest"], request["runtime_manifest_sha256"])["files"]
        for name in (request["manifest"], request["package"], review["path"], *posture):
            if files.get(str(Path(name).resolve())) != sha(name):
                raise ValueError("Unbound evaluation input or posture evidence")
        for name, digest in files.items():
            if sha(name) != digest:
                raise ValueError(f"Changed evaluation input: {name}")
    return request


def prepare(args, *, diagnostic=False):
    """Write inputs only: never starts Kit, inference or a policy server."""
    benchmark_version = getattr(args, "benchmark", DEFAULT_BENCHMARK)
    data_root = getattr(args, "data_root", None)
    v1_root = getattr(args, "v1_root", None)
    verify(
        benchmark_version=benchmark_version,
        assets=args.assets,
        data_root=data_root,
        v1_root=v1_root,
    )
    task_id = benchmark.benchmark_legacy_id(args.task)
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    scenes = benchmark_assets.prepare_scenes(
        task_id,
        args.seeds,
        args.assets,
        work / "scenes",
        reference=False,
        benchmark_version=benchmark_version,
        data_root=data_root,
        v1_root=v1_root,
    )
    contract = benchmark.read(protocol_path(ROOT, benchmark_version, v1_root=v1_root))
    task = contract["tasks"][task_id]
    spec_identity = shared_spec_identity()
    spec = Resources.from_roots(checkout=benchmark.ROOT, assets=benchmark.ROOT).path(
        "shared_ik_spec"
    )
    release = ROOT / "dist/release-manifest.json"
    release_identity = verify_release(release, checkout=ROOT)
    dependency = work / "dependencies.json"
    probe = subprocess.run(
        [str(args.isaac_python), "-m", "tower_sim.bench.verification"],
        env=child_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    save_exclusive(dependency, json.loads(probe.stdout))
    ik_dependency = work / "ik-dependencies.json"
    ik_probe = subprocess.run(
        [
            str(args.ik_python),
            "-m",
            "tower_sim.bench.verification",
            "--ik",
            "--release",
            str(release),
            "--release-sha256",
            release_identity["manifest_sha256"],
            "--checkout",
            str(ROOT),
        ],
        env=child_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    save_exclusive(ik_dependency, json.loads(ik_probe.stdout))
    request = dict(
        public=envelope(release_identity, benchmark=benchmark_version),
        task=task_id,
        canonical_task_id=benchmark.resolve(task_id).id,
        seeds=args.seeds,
        duration=task["duration_s"],
        success_rule=task["success_rule"],
        prompt=task["prompt"],
        manifest=str(scenes),
        output=str(args.run_output.resolve()),
        package=str(args.package.resolve()),
        policy_url=args.policy_url,
        assets=str(args.assets.resolve()),
        data_root=str(data_root.resolve()) if data_root is not None else None,
        v1_root=str(v1_root.resolve()) if v1_root is not None else None,
        portable_assets=str(args.assets.resolve()),
        ik_python=str(args.ik_python.absolute()),
        gpu=args.gpu,
        trial_id=args.run_output.name,
        release_manifest=str(release),
        release_manifest_sha256=sha(release),
        dependency_manifest=str(dependency),
        dependency_manifest_sha256=sha(dependency),
        ik_dependency_manifest=str(ik_dependency),
        ik_dependency_manifest_sha256=sha(ik_dependency),
        ik_bounds=str(spec),
        ik_bounds_sha256=sha(spec),
    )
    if not diagnostic:
        package, review = benchmark.read(args.package), benchmark.read(args.pilot_review)
        for field in (
            "passed",
            "native_inference_qualified",
            "scene_reference_review_passed",
            "shared_ik_parity_passed",
        ):
            if review.get(field) is not True:
                raise ValueError(f"Pilot review has not passed {field}")
        if (
            review.get("package_digest") != package["package_digest"]
            or review.get("shared_spec_sha256") != spec_identity
        ):
            raise ValueError("Pilot review belongs to different policy/IK inputs")
        posture = benchmark.posture_evidence(review, task_id, package["package_digest"])
        policy = dict(contract["policy"])
        policy["hz"] = policy.pop("control_hz")
        protocol = work / "protocol.json"
        save_exclusive(
            protocol,
            dict(
                schema="tower-public-benchmark-protocol-v1",
                public=envelope(release_identity, benchmark=benchmark_version),
                task=task_id,
                seeds=args.seeds,
                scene_version="tower_v4",
                duration_s=task["duration_s"],
                success_rule=task["success_rule"],
                package_digest=package["package_digest"],
                policy=policy,
                authorization={"scene_selection": "Current TOWER v4 scene"},
                pilot_review=dict(
                    path=str(args.pilot_review.resolve()), sha256=sha(args.pilot_review)
                ),
                ik=dict(shared_spec_sha256=spec_identity),
            ),
        )
        inputs = [scenes, protocol, args.package, args.pilot_review, *posture]
        inputs += [p for p in (work / "scenes").rglob("*") if p.is_file()]
        manifest = work / "input-manifest.json"
        save_exclusive(
            manifest,
            dict(
                schema="tower-evaluation-inputs-v2",
                files={str(p.resolve()): sha(p) for p in inputs},
            ),
        )
        request.update(
            protocol=str(protocol),
            protocol_sha256=sha(protocol),
            runtime_manifest=str(manifest),
            runtime_manifest_sha256=sha(manifest),
        )
    path = work / "request.json"
    save_exclusive(path, request)
    preflight(path, diagnostic=diagnostic)
    return path


def launch(path, python, *, diagnostic=False):
    request = preflight(path, diagnostic=diagnostic)
    benchmark.check_environment(python)
    command = [str(python), "-u", str(Path(__file__).resolve()), "--request", str(path)]
    if diagnostic:
        command.append("--diagnostic")
    with benchmark.runtime_request(path, request) as normalized:
        command[command.index("--request") + 1] = str(normalized)
        result = subprocess.run(command, env=child_environment(), check=False)
    from scripts.maintainer.candidate_policy import finish

    code = finish(request, result.returncode, diagnostic=diagnostic)
    if not diagnostic:
        # Preserve the historical public seed paths without moving hash-bound
        # evidence or changing the candidate's internal recording layout.
        output = Path(request["output"])
        for seed in request["seeds"]:
            (output / f"seed-{seed}").symlink_to(
                Path("simulation") / f"seed-{seed}", target_is_directory=True
            )
    return code


def command(args):
    benchmark_version = getattr(args, "benchmark", DEFAULT_BENCHMARK)
    v1_root = getattr(args, "v1_root", None)
    if args.command == "verify":
        print(
            json.dumps(
                verify(
                    benchmark_version=benchmark_version,
                    assets=getattr(args, "assets", None),
                    data_root=getattr(args, "data_root", None),
                    v1_root=v1_root,
                ),
                indent=2,
            )
        )
    elif args.command == "describe":
        verify_data(ROOT, benchmark=benchmark_version, v1_root=v1_root)
        print(
            json.dumps(
                dict(
                    benchmark.read(protocol_path(ROOT, benchmark_version, v1_root=v1_root)),
                    public=envelope(benchmark=benchmark_version),
                ),
                indent=2,
            )
        )
    elif args.command == "check-environment":
        print(json.dumps(benchmark.check_environment(args.isaac_python), indent=2))
    elif args.command == "score":
        if args.output.resolve().is_relative_to(args.run.resolve()) or args.output.exists():
            raise ValueError("Output must be a new file outside the input recording")
        from tower_eval.scoring.api import score_run

        verify_data(ROOT, benchmark=benchmark_version, v1_root=v1_root)
        report = score_run(args.run, args.task, ruleset=args.ruleset, input_path=args.input_path)
        save_exclusive(args.output, dict(report, public=envelope(benchmark=benchmark_version)))
        return int(not report.get("valid"))
    else:
        request = preflight(args.request)
        if args.command == "check-request":
            print(
                json.dumps(
                    dict(
                        passed=True,
                        task=request["task"],
                        canonical_task_id=benchmark.resolve(request["task"]).id,
                        seeds=request["seeds"],
                        public=request["public"],
                    )
                )
            )
        else:
            benchmark.require_eula(args.accept_eula)
            return launch(args.request, args.isaac_python)
    return 0


def portable(args):
    if args.command == "prepare-request":
        print(prepare(args))
        return 0
    benchmark.require_eula(args.accept_eula)
    if args.command == "run":
        return launch(args.request, sys.executable)
    return smoke(args)


def diagnostic(args):
    benchmark.require_eula(args.accept_eula)
    if args.output.exists():
        raise ValueError("Diagnostic output must be a new directory")
    inputs = SimpleNamespace(**vars(args))
    inputs.output = args.output / "inputs"
    inputs.run_output, inputs.seeds, inputs.isaac_python = (
        args.output,
        [args.seed],
        Path(sys.executable),
    )
    path = prepare(inputs, diagnostic=True)
    code = launch(path, sys.executable, diagnostic=True)
    result = benchmark.read(args.output / "diagnostic-result.json")
    result.pop("prefix_only", None)
    result.pop("benchmark_deadline_s", None)
    if "score" in result:
        result["score"].pop("prefix_only", None)
    result.update(
        canonical_task_id=benchmark.resolve(args.task).id,
        public=benchmark.read(args.output / "implementation-identity.json")["public"],
    )
    save_exclusive(args.output / "result.json", result)
    if "score" in result:
        save_exclusive(args.output / "score.json", result["score"])
    print(json.dumps(result, indent=2))
    return code


def smoke(args):
    from scripts.maintainer.public_smoke import run

    return run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    request = preflight(args.request, diagnostic=args.diagnostic)
    sys.path.insert(0, str(ROOT))
    from scripts.maintainer.candidate_policy import worker

    verification = ReleaseVerification(
        checkout=ROOT, request=request, verify_assets=benchmark_assets.verify
    )
    args.ik_python = Path(request["ik_python"])
    args.private_kit_root = Path(request["output"]).with_name(Path(request["output"]).name + "-kit")
    with benchmark.runtime_request(args.request, request) as normalized:
        args.request = normalized
        worker(
            args,
            verification=verification,
            runner=create_runner(
                benchmark_version=request["public"]["benchmark"],
                data_root=request.get("data_root"),
                v1_root=request.get("v1_root"),
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
