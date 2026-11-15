"""Replay frozen expert commands on Isaac 6.1 and compare physical evidence.

Inputs are a local expert MANIFEST.json (task, seed, file, sha256 per trial)
and the pinned public assets. No policy server or hardware is used. Expert
commands remain external evidence, never generated or repaired by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
import time
from pathlib import Path
from typing import Any, cast

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
__tower_entry_sha256__ = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
from PIL import Image
from tower_sim.bench.artifact_io import read, write
from tower_sim.bench.resources import sha

from scripts.maintainer.archive_root import ArchivedPath

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ArchivedPath("")
sys.path.insert(0, str(ROOT / "tools/simulator"))
benchmark = importlib.import_module("benchmark")
from scripts.maintainer.gpu_host_profile import (
    candidate_summary,
    comparison_identity,
    frozen_summary,
    read_summary,
    reference_contract,
    verify_runtime,
)
from scripts.maintainer.gpu_identity import (
    COMPONENTS,
    benchmark_version,
    public_trial,
    require_equal,
)
from scripts.maintainer.gpu_runner import CommandTrajectory, ReplayRequest, resolve_candidate

TOLERANCES = dict(
    qpos_rad=1e-4, block_position_m=1e-4, block_orientation_deg=0.05, image_psnr_db=35.0
)


def gpu_idle(gpu: int, desktop_exception: bool) -> None:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    memory, utilization = (int(value.strip()) for value in output.strip().split(","))
    if utilization != 0:
        raise RuntimeError("Selected GPU is busy")
    if memory < 512:
        return
    # Explicit admission exception for the idle desktop service only. Query
    # graphics clients as well as compute clients; do not rely on compute-apps.
    import xml.etree.ElementTree as ET

    tree = ET.fromstring(
        subprocess.check_output(["nvidia-smi", f"--id={gpu}", "-q", "-x"], text=True)
    )
    names = [p.findtext("process_name", "") for p in tree.findall(".//process_info")]
    if (
        not desktop_exception
        or not names
        or any(Path(name).name != "gnome-remote-desktop-daemon" for name in names)
    ):
        raise RuntimeError("Selected GPU exceeds the 512 MiB idle memory limit")


def worker(args: Any) -> None:
    args.private_kit_root = getattr(args, "private_kit_root", None)
    if args.private_kit_root:
        from scripts.maintainer.kit_safety import configure

        configure(args.private_kit_root)
    if args.implementation in ("tower-sim", "public"):
        candidate_worker(args)
        return
    benchmark.check_environment(sys.executable)
    assets = importlib.import_module("benchmark_assets")
    portable = importlib.import_module("benchmark_portable")
    task = benchmark.benchmark_legacy_id(args.task)
    entry = next(
        row for row in read(args.expert_root / "MANIFEST.json")["trials"] if row["task"] == task
    )
    source = (args.expert_root / entry["file"]).resolve()
    if not source.is_relative_to(args.expert_root.resolve()) or sha(source) != entry["sha256"]:
        raise ValueError("Expert command identity differs")
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    scenes = assets.prepare_scenes(
        task, [entry["seed"]], args.assets, work / "scenes", reference=True
    )
    configured = portable.configure(args.assets, work, args.gpu)
    configured.output = work / "simulation"
    configured.task = task
    protocol = read(BUNDLE / "protocol.json")["tasks"][task]
    configured.duration = protocol["duration_s"]
    configured.success_rule = protocol["success_rule"]
    configured.command_replay = True
    configured.policy_url = "reference-replay://disabled"  # No connection in command_replay.
    configured.prompt = protocol["prompt"]
    configured.package = work / "replay-package.json"
    write(
        configured.package,
        {
            "package_digest": entry["sha256"],
            "checkpoint_sha256": entry["sha256"],
            "kind": "scripted_reference_command_replay",
        },
    )
    os.environ["TOWER_SDK_ROOT"] = str(Path(sys.prefix).resolve())
    os.environ.update(TOWER_BASELINE_METRICS=str(work / "metrics.jsonl"), TOWER_BASELINE_TRIAL=task)
    policy = importlib.import_module("batch_policy_worker")
    screen = policy.load_screen()
    probe = screen.load_probe()
    runtime = probe.load_worker()
    # The archived runner's audited, exact optimization paths. These shipped
    # modules compare cached results/decisions against the original functions.
    early_success = runtime.load_local("formal_batch_job").formal.success_scorer()
    importlib.import_module("fast_early_stop").install(early_success)
    importlib.import_module("fast_camera_quality").install(runtime.runtime)
    rows = runtime.load_manifest(scenes, allow_single_environment=True)
    row = rows[0]
    with np.load(source, allow_pickle=False) as data:
        times, actions = data["time_s"].copy(), data["action_qpos"].copy()
    if (
        times.ndim != 1
        or actions.shape != (len(times), 16)
        or len(times) < 2
        or times[0] != 0
        or not np.all(np.diff(times) > 0)
        or not np.isfinite(actions).all()
    ):
        raise ValueError("Invalid expert command arrays")
    np.testing.assert_allclose(actions[0], row["initial"]["qpos"], atol=2e-6, rtol=0)
    # Purely bind archived commands to the already verified frozen scene; this
    # does not change the scene loader, production bounds, or any source file.
    row.update(times=times, actions=actions, receipt=dict(source_numeric_sha256=entry["sha256"]))
    configured.loaded_configuration_hashes = policy.configuration_snapshot(configured, runtime.sha)
    original = runtime.adapt_task12_config
    runtime.adapt_task12_config = lambda config, cls: original(
        dict(config, render=True, camera_batch_mode="individual"), cls
    )
    gains_module = importlib.import_module("v4_policy_pilot")
    install = importlib.import_module("exact_j2_cache").install

    app = None
    try:
        app, env, replay = runtime.bootstrap(configured, rows)
        if args.private_kit_root:
            from scripts.maintainer.kit_safety import verify

            verify(args.private_kit_root)
        install(replay.j2comp)
        loader = runtime.load_local

        def load_local(name: str) -> Any:
            module = loader(name)
            if name == "control":
                receipt = importlib.import_module("fast_observed_joints").install(
                    module, replay.j2comp, mode="fast"
                )
                write(work / "fast-observed-joints.json", receipt)
            return module

        runtime.load_local = load_local
        gains = gains_module.apply_v4(env, configured.output, runtime)
        source_reader = runtime.runtime.imported_sources

        def checked_sources() -> Any:
            gains_module.verify_v4(env, gains)
            return source_reader()

        runtime.runtime.imported_sources = checked_sources
        code = policy.execute(configured, runtime, probe, screen, rows, app, env, replay)
        if code != 0:
            raise RuntimeError(f"Frozen command_replay failed: {code}")
        # Kit's fast shutdown can terminate the interpreter inside close().
        # Persist the derived artifacts while the finalized files are available.
        collect(work, task, entry)
        if args.allow_desktop_gpu:
            write(work / "summary.json", frozen_summary(read(work / "summary.json"), args.gpu))
    finally:
        if app is not None:
            app.close()


def candidate_preflight(args: Any) -> tuple[Any, dict[str, Any]]:
    from tower_sim.bench.asset_identity import asset_manifest_identity

    if args.implementation == "public":
        runner = importlib.import_module("benchmark_candidate").create_runner(
            benchmark_version=getattr(args, "benchmark", "2026.11.1"),
            data_root=getattr(args, "data_root", None),
            v1_root=getattr(args, "v1_root", None),
        )
    else:
        if getattr(args, "benchmark", "2026.11.1") != "2026.11.1":
            raise ValueError("tower-sim requires --benchmark 2026.11.1")
        runner = resolve_candidate()
    components = tuple(sorted(args.frozen_component))
    if len(set(components)) != len(components):
        raise ValueError("Duplicate frozen component substitution")
    if components and args.reference_root is None:
        raise ValueError("Frozen component substitution requires --reference-root")
    binding = runner.identity(args.assets, args.gpu, components, args.reference_root)
    candidate_summary(binding["candidate_identity"], binding["contract"], [])
    # The contract is runner-reported; check the asset release ourselves.
    manifest = args.assets / "manifest.json"
    expected = binding["contract"]["immutable_data"]["assets_manifest_sha256"]
    if not manifest.is_file() or asset_manifest_identity(sha(manifest)) != expected:
        raise ValueError("Candidate assets differ: manifest.json")
    require_equal(
        list(components), binding["candidate_identity"]["frozen_components"], "frozen_components"
    )
    return runner, binding


def candidate_worker(args: Any) -> None:
    runner, binding = candidate_preflight(args)
    task = benchmark.benchmark_legacy_id(args.task)
    entry = next(
        row for row in read(args.expert_root / "MANIFEST.json")["trials"] if row["task"] == task
    )
    entry = public_trial(entry)
    source = (args.expert_root / entry["file"]).resolve()
    if not source.is_relative_to(args.expert_root.resolve()) or sha(source) != entry["sha256"]:
        raise ValueError("Expert command identity differs")
    require_equal(
        reference_contract()["reference_identity"]["experts"][task],
        dict(scene_id=entry["scene_id"], sha256=entry["sha256"]),
        f"{task}/expert",
    )
    with np.load(source, allow_pickle=False) as data:
        times, actions = data["time_s"].copy(), data["action_qpos"].copy()
    if (
        times.ndim != 1
        or len(times) < 2
        or times[0] != 0
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
        or actions.shape != (len(times), 16)
        or not np.isfinite(actions).all()
    ):
        raise ValueError("Invalid expert command arrays")
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    request = ReplayRequest(
        task,
        int(entry["scene_id"].removeprefix("scene-")),
        entry["sha256"],
        args.assets.resolve(),
        work,
        args.gpu,
        binding["contract"],
        tuple(sorted(args.frozen_component)),
        args.reference_root,
    )
    prepared = runner.setup(request)
    try:
        if args.private_kit_root:
            from scripts.maintainer.kit_safety import verify

            verify(args.private_kit_root)
        evidence = prepared.replay(CommandTrajectory(times, actions))
        with np.load(work / "trace.npz", allow_pickle=False) as data:
            stamps = data["time"]
            row = dict(
                task=task,
                scene_id=entry["scene_id"],
                expert_sha256=entry["sha256"],
                trace_sha256=sha(work / "trace.npz"),
                samples=len(stamps),
                duration_s=float(stamps[-1]),
                cameras=evidence.cameras,
                replay_completed=evidence.replay_completed,
                task_success=evidence.task_success,
                policy_action_failure=evidence.policy_action_failure,
            )
            validate_trace(data, row)
        validate_artifacts(work, row)
        write(
            work / "summary.json",
            candidate_summary(binding["candidate_identity"], binding["contract"], [row]),
        )
    finally:
        prepared.close()


def collect(work: Path, task: str, entry: dict[str, Any]) -> None:
    trial = work / "simulation" / f"seed-{entry['seed']}"
    result = read(work / "simulation/batch-policy-results.json")
    status = read(trial / "status.json")
    if (
        not result["completed"]
        or not status["evidence_complete"]
        or status["seed"] != entry["seed"]
        or status["checkpoint_package_digest"] != entry["sha256"]
    ):
        raise ValueError("Cannot collect incomplete or differently bound replay evidence")
    with h5py.File(trial / "episode/states.hdf5", "r") as stream:
        arrays = {key: cast(h5py.Dataset, stream[key])[:] for key in ("time", "qpos", "blocks")}
    np.savez_compressed(work / "trace.npz", **arrays)
    cameras: dict[str, str] = {}
    with h5py.File(trial / "camera-frames.hdf5", "r") as stream:
        for name in stream:
            assert isinstance(name, str)
            if name == "time":
                continue
            image = Image.open(io.BytesIO(bytes(cast(h5py.Dataset, stream[name])[0]))).convert(
                "RGB"
            )
            filename = f"first-{name}.png"
            image.save(work / filename)
            cameras[name] = filename
    if len(cameras) != 4:
        raise ValueError("Expected all four first-frame cameras")
    write(
        work / "summary.json",
        dict(
            task=task,
            seed=entry["seed"],
            expert_sha256=entry["sha256"],
            trace_sha256=sha(work / "trace.npz"),
            samples=len(arrays["time"]),
            duration_s=float(arrays["time"][-1]),
            cameras=cameras,
            replay_completed=result["completed"],
            task_success=status["task_success"],
            policy_action_failure=status["policy_action_failure"],
            source_manifest_sha256=sha(BUNDLE / "SOURCE_MANIFEST.json"),
        ),
    )


def run(args: Any) -> None:
    import fcntl

    # Resolve before creating output, acquiring a GPU lock, or admitting a job.
    binding = (
        candidate_preflight(args)[1] if args.implementation in ("tower-sim", "public") else None
    )
    if args.allow_desktop_gpu:
        verify_runtime(args.gpu)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    entries = read(args.expert_root / "MANIFEST.json")["trials"]
    tasks = (
        list(reference_contract()["contract"]["task_order"])
        if binding
        else list(read(BUNDLE / "protocol.json")["tasks"])
    )
    if len(entries) != 12 or {row["task"] for row in entries} != set(tasks):
        raise ValueError("Expert manifest must contain exactly one frozen seed per task")
    if args.task:
        tasks = [benchmark.benchmark_legacy_id(args.task)]
    # Hosts that share GPUs with other schedulers set their own lock-path template.
    lock = Path(
        os.environ.get("TOWER_GPU_LOCK", "/run/lock/tower-gpu{gpu}.lock").format(gpu=args.gpu)
    )
    descriptor = os.open(lock, os.O_RDONLY | os.O_CREAT, 0o666)
    with os.fdopen(descriptor) as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task in tasks:
            entry = next(row for row in entries if row["task"] == task)
            summary_path = args.output / task / "summary.json"
            if args.resume and summary_path.is_file():
                summary = resume_summary(summary_path, binding)
                if (
                    summary["task"] != task
                    or (
                        summary["scene_id"] != public_trial(entry)["scene_id"]
                        if binding
                        else summary["seed"] != entry["seed"]
                    )
                    or summary["expert_sha256"] != entry["sha256"]
                    or not summary["replay_completed"]
                    or summary["trace_sha256"] != sha(summary_path.parent / "trace.npz")
                    or not all(
                        (summary_path.parent / file).is_file()
                        for file in summary["cameras"].values()
                    )
                ):
                    raise ValueError("Cannot resume changed or incomplete baseline artifacts")
                continue
            # GPU context cleanup and utilization sampling can outlast child
            # exit. Wait up to a minute; admission limits remain unchanged.
            for attempt in range(60):
                try:
                    if args.allow_desktop_gpu:
                        from scripts.maintainer.gpu_admission import admit_desktop

                        admit_desktop(args.gpu, args.output / f"{task}-admission.json")
                    else:
                        gpu_idle(args.gpu, args.allow_desktop_service)
                    break
                except RuntimeError:
                    if attempt == 59:
                        raise
                    time.sleep(1)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_worker",
                "--assets",
                str(args.assets.resolve()),
                "--expert-root",
                str(args.expert_root.resolve()),
                "--output",
                str((args.output / task).resolve()),
                "--task",
                task,
                "--gpu",
                str(args.gpu),
            ]
            if binding:
                command.extend(
                    ["--implementation", args.implementation, "--benchmark", args.benchmark]
                )
                if args.data_root is not None:
                    command.extend(["--data-root", str(args.data_root.resolve())])
                if args.v1_root is not None:
                    command.extend(["--v1-root", str(args.v1_root.resolve())])
                for component in args.frozen_component:
                    command.extend(["--frozen-component", component])
                if args.reference_root:
                    command.extend(["--reference-root", str(args.reference_root.resolve())])
            if args.allow_desktop_gpu:
                command.append("--allow-desktop-gpu")
            if args.private_kit_root:
                command.extend(["--private-kit-root", str(args.private_kit_root.resolve() / task)])
            with (args.output / (task + ".log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"Replay complete: {task}", flush=True)
    if binding:
        trials = [resume_summary(args.output / task / "summary.json", binding) for task in tasks]
        summary = candidate_summary(binding["candidate_identity"], binding["contract"], trials)
    else:
        summary = dict(
            schema="tower-gpu-baseline-v1",
            trials=[read(args.output / task / "summary.json") for task in tasks],
        )
        if args.allow_desktop_gpu:
            summary = frozen_summary(summary, args.gpu)
    write(args.output / "summary.json", summary)


def resume_summary(path: Path, binding: dict[str, Any] | None) -> dict[str, Any]:
    document = read(path)
    if binding is None:
        if document.get("source_manifest_sha256") != sha(BUNDLE / "SOURCE_MANIFEST.json"):
            raise ValueError("Cannot resume changed frozen source_manifest_sha256")
        return document
    if document.get("schema") != "tower-gpu-baseline-v2":
        raise ValueError("Cannot resume a different implementation: expected candidate v2")
    normalized = read_summary(document)
    for field in ("candidate_identity", "contract"):
        require_equal(binding[field], normalized[field], f"resume/{field}")
    if len(normalized["trials"]) != 1:
        raise ValueError("Cannot resume a candidate task with multiple trial summaries")
    row = normalized["trials"][0]
    validate_artifacts(path.parent, row)
    with np.load(path.parent / "trace.npz", allow_pickle=False) as data:
        validate_trace(data, row)
    return row


def artifact_path(root: Path, filename: str) -> Path:
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe replay artifact: {filename}")
    return path


def validate_artifacts(root: Path, summary: dict[str, Any]) -> None:
    if summary["replay_completed"] is not True:
        raise ValueError("Cannot compare incomplete replay evidence")
    for field in ("task_success", "policy_action_failure"):
        if type(summary[field]) is not bool:
            raise ValueError(f"Invalid terminal identity: {summary['task']}/{field}")
    if sha(artifact_path(root, "trace.npz")) != summary["trace_sha256"]:
        raise ValueError("Trace digest differs from summary")
    if len(summary["cameras"]) != 4:
        raise ValueError("Expected identical four-camera coverage")
    for filename in summary["cameras"].values():
        artifact_path(root, filename)


def validate_trace(stream: Any, summary: dict[str, Any]) -> None:
    task = summary["task"]
    if set(stream.files) != {"time", "qpos", "blocks"}:
        raise ValueError(f"Invalid trace arrays: {task}")
    stamps, qpos, blocks = (stream[key] for key in ("time", "qpos", "blocks"))
    if (
        stamps.ndim != 1
        or len(stamps) < 2
        or not np.isfinite(stamps).all()
        or stamps[0] != 0
        or not np.all(np.diff(stamps) > 0)
    ):
        raise ValueError(f"Invalid trace time: {task}")
    if type(summary["samples"]) is not int or len(stamps) != summary["samples"]:
        raise ValueError(f"Trace sample count differs from summary: {task}/samples")
    if float(stamps[-1]) != summary["duration_s"]:
        raise ValueError(f"Trace duration differs from summary: {task}/duration_s")
    if qpos.shape != (len(stamps), 16) or not np.isfinite(qpos).all():
        raise ValueError(f"Invalid trace shape or values: {task}/qpos")
    if (
        blocks.ndim != 3
        or blocks.shape[0] != len(stamps)
        or blocks.shape[1] < 1
        or blocks.shape[2] != 7
        or not np.isfinite(blocks).all()
    ):
        raise ValueError(f"Invalid trace shape or values: {task}/blocks")
    if np.any(np.linalg.norm(blocks[..., 3:], axis=-1) == 0):
        raise ValueError(f"Invalid zero quaternion: {task}/blocks")


def compare(
    baseline: Path,
    candidate: Path,
    *,
    required_tasks: list[str] | None = None,
    informational_cross_gpu: bool = False,
) -> dict[str, Any]:
    rows = []
    left_run = read_summary(read(baseline / "summary.json"))
    right_run = read_summary(read(candidate / "summary.json"))
    # A root acceptance/substitution identity must describe every actual worker.
    for root, run_summary in ((baseline, left_run), (candidate, right_run)):
        if run_summary["candidate_identity"]["implementation"] == "tower-sim":
            for trial in run_summary["trials"]:
                worker = read_summary(read(artifact_path(root, trial["task"] + "/summary.json")))
                for field in ("candidate_identity", "reference_identity", "contract"):
                    require_equal(run_summary[field], worker[field], f"{trial['task']}/{field}")
                require_equal([trial], worker["trials"], f"{trial['task']}/trials")
    comparison_identity(left_run, right_run, informational=informational_cross_gpu)
    reference, other = left_run["trials"], right_run["trials"]
    if required_tasks is None:
        required_tasks = list(reference_contract()["contract"]["task_order"])
    for trials in (reference, other):
        if sorted(row["task"] for row in trials) != sorted(required_tasks):
            raise ValueError("Baseline must cover every protocol task exactly once")
    if [row["task"] for row in reference] != [row["task"] for row in other]:
        raise ValueError("Baseline task order differs")
    for left, right in zip(reference, other, strict=True):
        task = left["task"]
        if not left["replay_completed"] or not right["replay_completed"]:
            raise ValueError("Cannot compare incomplete replay evidence")
        for key in (
            "scene_id",
            "expert_sha256",
            "samples",
            "duration_s",
            "replay_completed",
            "task_success",
            "policy_action_failure",
        ):
            if type(left[key]) is not type(right[key]) or left[key] != right[key]:
                raise ValueError(f"Replay identity differs: {task}/{key}")
        roots = [baseline / task, candidate / task]
        for root, summary in zip(roots, (left, right), strict=True):
            validate_artifacts(root, summary)
        with (
            np.load(roots[0] / "trace.npz", allow_pickle=False) as a,
            np.load(roots[1] / "trace.npz", allow_pickle=False) as b,
        ):
            validate_trace(a, left)
            validate_trace(b, right)
            if not np.array_equal(a["time"], b["time"]):
                raise ValueError(f"Replay time differs: {task}/time")
            for key in ("qpos", "blocks"):
                if a[key].shape != b[key].shape or not all(
                    np.isfinite(stream[key]).all() for stream in (a, b)
                ):
                    raise ValueError(f"Invalid or differently shaped trace: {task}/{key}")
            qpos = float(np.max(np.abs(a["qpos"] - b["qpos"])))
            position = float(
                np.max(np.linalg.norm(a["blocks"][..., :3] - b["blocks"][..., :3], axis=-1))
            )
            qa = np.asarray(a["blocks"][..., 3:], dtype=np.float64)
            qb = np.asarray(b["blocks"][..., 3:], dtype=np.float64)
            qa, qb = (
                qa / np.linalg.norm(qa, axis=-1, keepdims=True),
                qb / np.linalg.norm(qb, axis=-1, keepdims=True),
            )
            # Align equivalent quaternion signs. The chord/atan2 form stays
            # accurate near zero, where acos(dot) invents roundoff deltas even
            # for identical traces.
            qb *= np.where(np.sum(qa * qb, axis=-1, keepdims=True) < 0, -1, 1)
            angle = float(
                np.max(
                    np.degrees(
                        4
                        * np.arctan2(
                            np.linalg.norm(qa - qb, axis=-1), np.linalg.norm(qa + qb, axis=-1)
                        )
                    )
                )
            )
        images = {}
        if len(left["cameras"]) != 4 or left["cameras"].keys() != right["cameras"].keys():
            raise ValueError("Expected identical four-camera coverage")
        for name in left["cameras"]:
            pixels = [
                np.asarray(Image.open(artifact_path(root, summary["cameras"][name])), dtype=float)
                for root, summary in zip(roots, (left, right), strict=True)
            ]
            if (
                pixels[0].shape != pixels[1].shape
                or pixels[0].ndim != 3
                or pixels[0].shape[-1] != 3
                or not pixels[0].size
            ):
                raise ValueError("Camera shapes differ")
            mse = float(np.mean((pixels[0] - pixels[1]) ** 2))
            images[name] = None if mse == 0 else float(10 * np.log10(255**2 / mse))
        passed = (
            qpos <= TOLERANCES["qpos_rad"]
            and position <= TOLERANCES["block_position_m"]
            and angle <= TOLERANCES["block_orientation_deg"]
            and all(
                value is None or value >= TOLERANCES["image_psnr_db"] for value in images.values()
            )
        )
        rows.append(
            dict(
                task=task,
                scene_id=left["scene_id"],
                max_qpos_delta=qpos,
                max_block_position_m=position,
                max_block_orientation_deg=angle,
                camera_psnr_db=images,
                passed=passed,
                baseline_trace_sha256=left["trace_sha256"],
                candidate_trace_sha256=right["trace_sha256"],
            )
        )
    # A run with frozen component substitutions bisects divergence; it can never
    # accept a candidate, whatever its numerical result.
    substituted = any(
        run["candidate_identity"].get("frozen_components") for run in (left_run, right_run)
    )
    return dict(
        schema="tower-gpu-comparison-v2",
        acceptance_eligible=not substituted and not informational_cross_gpu,
        informational_cross_gpu=informational_cross_gpu,
        reference_identity=left_run["reference_identity"],
        baseline_identity=left_run["candidate_identity"],
        candidate_identity=right_run["candidate_identity"],
        benchmark_version=benchmark_version(right_run["candidate_identity"]),
        contract=left_run["contract"],
        candidate_contract=right_run["contract"],
        passed=all(row["passed"] for row in rows),
        tolerances=TOLERANCES,
        identical_image_psnr="null means identical pixels (infinite PSNR)",
        trials=rows,
    )


def assemble(output: Path) -> dict[str, Any]:
    """Publish a serial-schema summary for twelve isolated candidate task runs.

    Each task directory is copied unchanged from its shard before assembly.
    This performs no GPU work and does not admit partial or mixed revisions.
    """
    tasks = list(reference_contract()["contract"]["task_order"])
    present = {path.name for path in output.glob("task*") if path.is_dir()}
    if present != set(tasks):
        raise ValueError("Assembly requires every protocol task exactly once")
    documents = []
    rows = []
    for task in tasks:
        directory = output / task
        if directory.is_symlink():
            raise ValueError("Assembly task directories must not be links")
        document = read(artifact_path(directory, "summary.json"))
        if document.get("schema") != "tower-gpu-baseline-v2":
            raise ValueError("Assembly requires candidate v2 task summaries")
        normalized = read_summary(document)
        identity = normalized["candidate_identity"]
        if identity["implementation"] != "tower-sim" or identity.get("frozen_components"):
            raise ValueError("Assembly requires unsubstituted tower-sim recordings")
        if len(normalized["trials"]) != 1 or normalized["trials"][0]["task"] != task:
            raise ValueError("Assembly task summary does not match its directory")
        if documents:
            for field in ("candidate_identity", "reference_identity", "contract"):
                require_equal(documents[0][field], normalized[field], f"assembly/{field}")
        row = normalized["trials"][0]
        validate_artifacts(directory, row)
        with np.load(directory / "trace.npz", allow_pickle=False) as trace:
            validate_trace(trace, row)
        documents.append(normalized)
        rows.append(row)
    first = documents[0]
    result = dict(
        schema="tower-gpu-baseline-v2",
        candidate_identity=first["candidate_identity"],
        reference_identity=first["reference_identity"],
        contract=first["contract"],
        trials=rows,
    )
    profile = read(output / tasks[0] / "summary.json").get("host_profile")
    if profile is not None:
        result["host_profile"] = profile
    read_summary(result)
    with (output / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "_worker"):
        command = sub.add_parser(name)
        for field in ("assets", "expert-root", "output"):
            command.add_argument("--" + field, type=Path, required=True)
        command.add_argument("--benchmark", choices=("2026.11.0", "2026.11.1"), default="2026.11.1")
        command.add_argument("--data-root", type=Path)
        command.add_argument(
            "--v1-root", type=Path, help="Explicit v1 scene export for compatibility"
        )
        command.add_argument("--gpu", type=int, default=0)
        command.add_argument("--private-kit-root", type=Path)
        command.add_argument(
            "--implementation", choices=("frozen", "tower-sim", "public"), default="frozen"
        )
        command.add_argument(
            "--frozen-component",
            choices=COMPONENTS,
            action="append",
            default=[],
            help="Explicit candidate component substitution for divergence bisection",
        )
        command.add_argument(
            "--reference-root", type=Path, help="Frozen checkout for component substitution"
        )
        command.add_argument("--task", type=benchmark.task_argument)
        command.add_argument("--allow-desktop-service", action="store_true")
        command.add_argument(
            "--allow-desktop-gpu",
            action="store_true",
            help="Explicit four-path desktop admission with a recorded ten-second window",
        )
        command.add_argument(
            "--resume",
            action="store_true",
            help="Reuse only complete, hash-verified task artifacts",
        )
    command = sub.add_parser("assemble")
    command.add_argument("output", type=Path)
    command = sub.add_parser("compare")
    command.add_argument("baseline", type=Path)
    command.add_argument("candidate", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--informational-cross-gpu", action="store_true")
    command.add_argument(
        "--diagnostic",
        action="store_true",
        help="Allow runs with frozen component substitutions (never an acceptance result)",
    )
    for command in sub.choices.values():
        command.allow_abbrev = False
    args = parser.parse_args(argv)
    if (
        args.command in ("run", "_worker")
        and args.implementation == "public"
        and args.frozen_component
    ):
        parser.error("The public default cannot import frozen component substitutions")
    if (
        args.command in ("run", "_worker")
        and args.implementation == "frozen"
        and (args.frozen_component or args.reference_root)
    ):
        parser.error("Component substitution is only available with --implementation tower-sim")
    if args.command == "run":
        run(args)
    elif args.command == "_worker":
        worker(args)
    elif args.command == "assemble":
        assemble(args.output)
    else:
        result = compare(
            args.baseline, args.candidate, informational_cross_gpu=args.informational_cross_gpu
        )
        if not result["acceptance_eligible"] and not (
            args.diagnostic or args.informational_cross_gpu
        ):
            parser.error("Runs with frozen component substitutions require --diagnostic")
        write(args.output, result)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
