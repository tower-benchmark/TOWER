"""Explicit frozen pure-component probes in a disposable diagnostic process.

No automatic fallback. Native bootstrap/camera substitutions execute original helpers in the dedicated
diagnostic process; they never patch candidate modules.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts.maintainer.archive_root import archive_tools
from scripts.maintainer.golden_support import read, reference_scorer, sha
from scripts.maintainer.gpu_identity import COMPONENTS, reference_contract


def validate(root, components):
    if root is None or not set(components) <= set(COMPONENTS):
        raise ValueError("Explicit substitutions require a reference checkout and known components")
    bundle = Path(root).resolve() / "simulator/benchmark-202611"
    manifest = bundle / "SOURCE_MANIFEST.json"
    if sha(manifest) != reference_contract()["reference_identity"]["source_manifest_sha256"]:
        raise ValueError("Frozen component manifest differs")
    for relative, binding in read(manifest)["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or sha(path) != binding["sha256"]:
            raise ValueError(f"Frozen component changed: {relative}")
    return bundle


def function(path, name, globals_):
    """Execute the exact original function AST with explicit dependencies only."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    nodes: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(nodes) != 1:
        raise ValueError(f"Missing unique frozen function: {name}")
    namespace: dict[str, Any] = dict(globals_, __file__=str(path))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def require_same_function(original, current, name):
    def signature(path):
        nodes = [
            n
            for n in ast.parse(Path(path).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        ]
        if len(nodes) != 1:
            raise ValueError(f"Missing unique command function: {name}")
        return ast.dump(nodes[0], include_attributes=False)

    if signature(original) != signature(current):
        raise ValueError("External frozen command mapping differs")


def components(request, context, factories, validator):
    from tower_sim.bench.scoring import BenchScorer

    bundle = validate(request.reference_root, request.frozen_components)
    selected = set(request.frozen_components)
    runtime = archive_tools(bundle) / "runtime_worker.py"
    if "observation" in selected:
        observe = function(runtime, "encoder_observation", {"np": np})

        def observation(qpos, compensation, rotation, offset):
            sim = SimpleNamespace(
                read=lambda: SimpleNamespace(qpos=qpos),
                env=SimpleNamespace(lula=SimpleNamespace(tcp_rotation=rotation, tcp_offset=offset)),
            )
            replay = SimpleNamespace(j2comp=compensation, urdf_fk=compensation.urdf_fk)
            return observe(sim, replay)

        factories.observation = observation
    if "command_mapping" in selected:
        # Bind the relocated function itself to the archived command mapping.
        source = Path(request.reference_root) / "src/tower-eval/src/tower_eval/simulator/isaac.py"
        current = (
            Path(__file__).resolve().parents[2]
            / "src/tower-eval/src/tower_eval/simulator/commands.py"
        )
        require_same_function(source, current, "joint_commands")
        mapping = function(source, "joint_commands", {"np": np})
        config = context.resources.read("robot")
        context.env._commands = lambda targets, arm: mapping(
            targets,
            config["arms"][arm],
            context.env.robots[arm].dof_names,
            config["stage_units_per_meter"],
        )
    if selected & {"scoring", "early_stop"}:
        frozen = reference_scorer(Path(request.reference_root))
        if "early_stop" in selected:
            path = archive_tools(bundle) / "fast_early_stop.py"
            spec = importlib.util.spec_from_file_location("_p708_frozen_fast_stop", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.install(frozen)

        def scorer(task, **kwargs):
            instance = BenchScorer(task, **kwargs)
            if "scoring" in selected:
                instance.recording = lambda spec, rows, finger_geometry, **kw: (
                    frozen.from_physical_rows(spec, rows, finger_geometry)
                )
                instance.score = lambda recording: frozen.score_recording(
                    recording, success_rule=task.success_rule
                )
            if "early_stop" in selected and task.early_success_stop:
                name = task.identity.legacy_id
                instance.detector = (
                    frozen.FourPlaceHoldStop()
                    if name == "task1-2"
                    else frozen.Task2HoldStop()
                    if name == "task2-2"
                    else frozen.TaskAwareHoldStop(name)
                )
            return instance

        factories.scorer = scorer
    if "cameras" in selected:
        factories = camera_components(request, context, factories)
    return factories, validator


def native_context(request):
    """Prepare the original diagnostic namespace, only on an explicit selection.

    Original bootstrap/camera helpers retain their own historical patching. No
    candidate module or global is patched. The harness requires a fresh process.
    """
    import os
    import shutil
    import sys

    bundle = validate(request.reference_root, request.frozen_components)
    root = Path(request.reference_root).resolve()
    runtime = bundle / "runtime"
    os.environ.update(TOWER_ASSET_ROOT=str(request.assets), TOWER_SOURCE_ROOT=str(root))
    sys.path[:0] = [
        str(runtime / name)
        for name in ("support", archive_tools(bundle), "ik-capsule/runtime", "ik-capsule/tools")
    ]
    # Invoke the original portable functions with an explicitly bound reference
    # checkout descriptor, instead of replacing the maintainer's benchmark module.
    from scripts.maintainer import golden_support

    benchmark = SimpleNamespace(
        ROOT=root,
        RUNTIME=runtime,
        BUNDLE=bundle,
        verify_bundle=lambda: validate(root, request.frozen_components),
        read=read,
        sha256=sha,
    )
    portable_path = root / "tools/simulator/benchmark_portable.py"
    source = ast.parse(portable_path.read_text(), filename=str(portable_path))
    nodes: list[ast.stmt] = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in ("install_imports", "configure")
    ]
    namespace: dict[str, Any] = dict(
        Path=Path,
        os=os,
        sys=sys,
        json=__import__("json"),
        shutil=shutil,
        SimpleNamespace=SimpleNamespace,
        benchmark=benchmark,
        benchmark_assets=SimpleNamespace(
            verify=lambda assets: __import__(
                "scripts.maintainer.candidate_replay", fromlist=["data_identity"]
            ).data_identity(golden_support.ROOT, assets)
        ),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(portable_path), "exec"), namespace)
    work = request.output / "frozen-diagnostic"
    work.mkdir(exist_ok=False)
    configured = namespace["configure"](request.assets, work, request.gpu)
    configured.output = request.output / "simulation"
    configured.task = request.task

    def load(name):
        path = archive_tools(bundle) / f"{name}.py"
        spec = importlib.util.spec_from_file_location("_p708_diagnostic_" + name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    worker = load("batch_reference_worker")
    return SimpleNamespace(worker=worker, config=configured, load=load)


def bootstrap_context(request, resources, rows, task):
    from contextlib import ExitStack

    from tower_sim.bench.bootstrap import BenchContext
    from tower_sim.bench.drives import apply_v4
    from tower_sim.bench.j2comp import J2Compensation
    from tower_sim.bench.options import ControlOptions

    native = native_context(request)
    worker = native.worker
    # This is the original frozen GPU harness's explicit render configuration.
    original = worker.adapt_task12_config
    worker.adapt_task12_config = lambda config, cls: original(
        dict(config, render=True, camera_batch_mode="individual"), cls
    )
    app, env, legacy = worker.bootstrap(native.config, rows)
    stack = ExitStack()
    stack.callback(app.close)
    try:
        compensation = J2Compensation.from_cache(
            resources.path("scene_cache"),
            resources.path("scene"),
            options=ControlOptions(exact_j2_cache=False),
        )
        readback = read(native.config.output / "robot-runtime-readback.json")
        delays = np.array([readback["arms"][name]["delays_seconds"] for name in ("left", "back")])
        gains = apply_v4(env, native.config.output, resources.path("drive_profile"))
        context: Any = BenchContext(
            stack, app, env, resources, compensation, delays, gains, native.config.output
        )
        context.frozen_diagnostic = native
        context.frozen_replay = legacy
        return context
    except BaseException:
        stack.close()
        raise


def camera_components(request, context, factories):
    native = getattr(context, "frozen_diagnostic", None) or native_context(request)
    legacy = getattr(context, "frozen_replay", None)
    if legacy is None:
        legacy = native.worker.runtime.capsule_module(
            native.config.capsule, "runtime/replay.py", "replay"
        )
    probe = native.load("probe_batch_cameras")
    guard_module = __import__("camera_guard_terminal")
    serial = native.load("serial_camera_reference")
    factories.guard = guard_module.install
    factories.camera = lambda sim, output, **kw: probe.make_camera(
        sim, output, replay=legacy, source_mount_reference=context.resources.path("wrist_mounts")
    )
    factories.serial = lambda camera, index=0, **kw: serial.SerialCameraReference(camera, index)
    return factories
