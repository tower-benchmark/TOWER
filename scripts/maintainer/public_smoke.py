"""Public asset smoke using the same construction factory as policy evaluation."""

import importlib
import json
import subprocess
import sys
import traceback
from types import SimpleNamespace


def worker(args):
    import numpy as np
    from tower_sim.bench import cameras, environment
    from tower_sim.bench.camera_quality import CameraQuality
    from tower_sim.bench.evidence import BatchEvidence, settling_metrics
    from tower_sim.bench.observation import encoder_observation
    from tower_sim.bench.resources import save_exclusive
    from tower_sim.bench.verification import dependency_identity

    from scripts.maintainer.kit_safety import configure, verify

    benchmark = importlib.import_module("benchmark")
    public = importlib.import_module("benchmark_candidate")
    identity = dict(
        data=public.verify(
            benchmark_version=getattr(args, "benchmark", "2026.11.1"),
            assets=getattr(args, "assets", None),
            data_root=getattr(args, "data_root", None),
            v1_root=getattr(args, "v1_root", None),
        ),
        release=public.verify_release(benchmark.ROOT / "dist/release-manifest.json"),
        dependencies=dependency_identity(),
    )
    public.verify_startup(identity["release"])
    benchmark.check_environment(sys.executable)
    work = args.output.resolve()
    kit = work.with_name(work.name + "-kit")
    work.mkdir(parents=True, exist_ok=False)
    save_exclusive(work / "implementation-identity.json", identity)
    configure(kit)
    benchmark_assets = importlib.import_module("benchmark_assets")

    scenes = benchmark_assets.prepare_scenes(
        args.task,
        args.seeds,
        args.assets,
        work / "scenes",
        reference=False,
        benchmark_version=getattr(args, "benchmark", "2026.11.1"),
        data_root=getattr(args, "data_root", None),
        v1_root=getattr(args, "v1_root", None),
    )
    rows = benchmark_assets.load_rows(scenes)
    request = SimpleNamespace(
        task=args.task, assets=args.assets, output=work, gpu=args.gpu, frozen_components=()
    )
    from tower_sim.bench.shader_cache import configuration

    runner = public.create_runner(
        benchmark_version=getattr(args, "benchmark", "2026.11.1"),
        data_root=getattr(args, "data_root", None),
        v1_root=getattr(args, "v1_root", None),
    )
    runner.shader_cache = configuration(args, work)
    prepared = runner.setup(request, rows=rows)
    camera = None
    try:
        verify(kit)
        context, env = prepared.context, prepared.context.env
        evidence = BatchEvidence(env, len(rows))
        evidence.append(env.elapsed)
        for tick in range(1, 401):
            env.physics.simulate(1 / 400, (tick - 1) / 400)
            env.physics.fetch_results()
            env.elapsed[:] = tick / 400
            if tick % 20 == 0:
                evidence.append(env.elapsed)
        settling = []
        for index, row in enumerate(rows):
            samples = evidence.records[index]
            result = settling_metrics(
                samples[0]["blocks"], samples[15]["blocks"], samples[-1]["blocks"]
            )
            settling.append(result)
            if not result["passed"]:
                raise ValueError(f"Frozen scene failed settling: {row['candidate']['seed']}")
            evidence.write(index, row["output"])
        env.elapsed[:] = 0.0
        env.simulation_time = 1.0
        sim = SimpleNamespace(
            app=context.app,
            env=env,
            settings=context.resources.read("settings"),
            read=lambda: env.read()[0],
            read_batch=env.read,
        )
        observation = encoder_observation(
            env.read()[0].qpos, context.compensation, env.lula.tcp_rotation, env.lula.tcp_offset
        )
        capture = work / "capture"
        capture.mkdir()
        before = env.read()
        camera = cameras.make_camera(
            sim, capture, resources=context.resources, near_guard=environment.wrist_guard
        )
        cameras.assert_unchanged(before, env.read())
        pixels = np.asarray(camera.observe()["rgb"])
        quality = CameraQuality()
        checks = [quality(frame) for frame in pixels.reshape(-1, 240, 320, 3)]
        if not checks or not all(row["valid"] for row in checks):
            raise ValueError("Invalid camera images in public asset smoke check")
        np.savez_compressed(work / "camera-frames.npz", rgb=pixels)
        from PIL import Image

        for index, views in enumerate(pixels):
            Image.fromarray(np.concatenate(list(views), axis=1)).save(
                work / f"scene-{index}-cameras.png"
            )
        context.verify_drives()
        prepared.collect_sources()
        result = dict(
            passed=True,
            diagnostic_only=True,
            policy_evaluation=False,
            task=args.task,
            canonical_task_id=benchmark.resolve(args.task).id,
            seeds=args.seeds,
            physics_hz=400,
            simulated_seconds=1.0,
            encoder_observation=observation,
            settling=settling,
            camera_checks=checks,
            sdk=benchmark.check_environment(sys.executable),
            asset_manifest_sha256=benchmark.sha256(args.assets / "manifest.json"),
            public=public.envelope(),
        )
        save_exclusive(
            work / "result.json",
            json.loads(json.dumps(result, default=lambda value: value.tolist())),
        )
    except BaseException:
        # Native fast shutdown can exit before the interpreter prints this error.
        save_exclusive(work / "error.json", dict(error=traceback.format_exc()))
        raise
    finally:
        if camera is not None:
            camera.close()
        prepared.close()


def run(args):
    benchmark = importlib.import_module("benchmark")
    public = importlib.import_module("benchmark_candidate")

    if args._worker:
        worker(args)
        return 0
    if args.output.exists():
        raise ValueError("Smoke output must be a new directory")
    from tower_sim.bench.shader_cache import child_arguments, configuration, finalize

    cache_config = configuration(args, args.output)
    command = [
        sys.executable,
        "-u",
        str(benchmark.ROOT / "tools/simulator/benchmark_portable.py"),
        "smoke",
        "--_worker",
    ]
    for key in ("assets", "task", "output", "gpu"):
        command.extend(["--" + key, str(getattr(args, key))])
    command.extend(child_arguments(args))
    command.extend(["--seeds", *map(str, args.seeds)])
    command.extend(["--benchmark", getattr(args, "benchmark", "2026.11.1")])
    if getattr(args, "data_root", None) is not None:
        command.extend(["--data-root", str(args.data_root.resolve())])
    if getattr(args, "v1_root", None) is not None:
        command.extend(["--v1-root", str(args.v1_root.resolve())])
    if args.accept_eula:
        command.append("--accept-eula")
    result = subprocess.run(command, env=public.child_environment(), check=False)
    receipt = args.output / "result.json"
    passed = False
    try:
        passed = (
            result.returncode == 0
            and receipt.is_file()
            and benchmark.read(receipt).get("passed") is True
        )
    finally:
        finalize(cache_config, args.output, passed=passed)
    if not passed:
        raise RuntimeError(
            f"Smoke worker did not produce a passing receipt (exit {result.returncode})"
        )
    print(receipt.read_text())
    return 0
