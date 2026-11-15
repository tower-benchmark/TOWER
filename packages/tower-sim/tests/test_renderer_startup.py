"""Renderer startup contracts without importing the native SDK."""

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tower_sim.bench import app, bootstrap, shader_cache

ROOT = Path(__file__).resolve().parents[3]
FLAGS = {
    "--/app/updateOrder/checkForHydraRenderComplete=1000",
    "--/app/renderer/waitIdle=true",
    "--/app/hydraEngine/waitIdle=true",
}


@pytest.mark.parametrize("entry", ["evaluation", "diagnostic", "smoke", "replay"])
def test_production_launch_record(entry, tmp_path, monkeypatch):
    # Execute the production call expression with only native boundaries replaced.
    tree = ast.parse(Path(bootstrap.__file__).read_text())
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "grouped_runtime"
    )
    monkeypatch.setattr(app, "require_python312", lambda: None)
    output = tmp_path / entry
    output.mkdir()
    scope = dict(
        startup=app,
        native=lambda config: SimpleNamespace(close=lambda: None),
        gpu=0,
        output=output,
        sdk=tmp_path,
        factories=SimpleNamespace(experience=None),
        setup=lambda session: None,
    )
    original = list(sys.argv)
    with eval(compile(ast.Expression(call), "bootstrap", "eval"), scope) as launch:
        assert set(sys.argv) >= FLAGS
        launch(dict(headless=True, experience="provided.kit"))
        assert set(sys.argv) >= FLAGS
    assert sys.argv == original
    assert set(json.loads((output / "launch.json").read_text())["extra_args"]) >= FLAGS


@pytest.mark.parametrize("corrupt", ["kit/shadercache/shader", "cuda/kernel"])
def test_seed_roundtrip_and_corruption(tmp_path, monkeypatch, corrupt):
    monkeypatch.setenv("CUDA_CACHE_PATH", "caller-cache")
    warm = tmp_path / "warm"
    output = tmp_path / "warm-run"
    options = SimpleNamespace(shader_cache_warmup=warm)
    config = shader_cache.configuration(options, output)
    argv = ["worker", "--portable-root", str(tmp_path / "warm-run-kit")]
    with app.private_shader_cache(**config, argv=argv) as receipt:
        target = Path(receipt["private_cache"])
        (target / "shadercache").mkdir()
        (target / "shadercache/shader").write_bytes(b"compiled")
        cuda = Path(receipt["private_cuda_cache"])
        assert os.environ["CUDA_CACHE_PATH"] == str(cuda)
        assert receipt["cuda_kernel_cache_before"]["file_count"] == 0
        (cuda / "kernel").write_bytes(b"kernel")
    assert os.environ["CUDA_CACHE_PATH"] == "caller-cache"
    (output / "simulation").mkdir(parents=True)
    (output / "simulation/private-shader-cache.json").write_text(json.dumps(receipt))
    (output / "result.json").write_text(json.dumps(dict(passed=False)))
    shader_cache.finalize(options, output, passed=False)
    assert json.loads((warm / "warmup.json").read_text())["source_run_passed"] is False
    assert json.loads((output / "result.json").read_text())["passed"] is False

    # A fresh second process consumes the first pass's verified compilation
    # output; its own result must pass before its cache is used concurrently.
    parser = argparse.ArgumentParser()
    shader_cache.add_arguments(parser)
    options = parser.parse_args(
        [
            "--shader-cache-seed",
            str(warm / "cache"),
            "--shader-cache-digest",
            str(warm / "digest.sha256"),
            "--shader-cache-warmup",
            str(tmp_path / "complete"),
        ]
    )
    copied = parser.parse_args(shader_cache.child_arguments(options))
    second = tmp_path / "second"
    config = shader_cache.configuration(copied, second)
    with app.private_shader_cache(
        **config, argv=["worker", "--portable-root", str(tmp_path / "second-kit")]
    ) as receipt:
        target = Path(receipt["private_cache"])
        assert (target / "shadercache/shader").read_bytes() == b"compiled"
        (target / "shadercache/remaining").write_bytes(b"completed")
        assert (Path(receipt["private_cuda_cache"]) / "kernel").read_bytes() == b"kernel"
        assert receipt["cuda_kernel_cache_before"]["file_count"] == 1
        assert receipt["cuda_kernel_cache_before"]["size_bytes"] == 6
    (second / "simulation").mkdir(parents=True)
    (second / "simulation/private-shader-cache.json").write_text(json.dumps(receipt))
    shader_cache.finalize(copied, second, passed=True)
    complete = tmp_path / "complete"
    assert json.loads((complete / "warmup.json").read_text())["source_run_passed"] is True
    args = parser.parse_args(
        [
            "--shader-cache-seed",
            str(complete / "cache"),
            "--shader-cache-digest",
            str(complete / "digest.sha256"),
        ]
    )
    config = shader_cache.configuration(args, tmp_path / "run")
    with app.private_shader_cache(
        **config, argv=["worker", "--portable-root", str(tmp_path / "run-kit")]
    ) as receipt:
        target = Path(receipt["private_cache"])
        assert (target / "shadercache/remaining").read_bytes() == b"completed"
    assert not (tmp_path / "run-kit/cache").exists()
    assert not (tmp_path / "run-kit/cuda-cache").exists()
    assert os.environ["CUDA_CACHE_PATH"] == "caller-cache"
    (complete / "cache" / corrupt).write_bytes(b"changed")
    with (
        pytest.raises(RuntimeError, match="differs"),
        app.private_shader_cache(
            **config, argv=["worker", "--portable-root", str(tmp_path / "run-kit")]
        ),
    ):
        pytest.fail("Corrupt seed accepted")
    assert os.environ["CUDA_CACHE_PATH"] == "caller-cache"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--shader-cache-seed", "seed"],
        ["--shader-cache-digest", "digest"],
    ],
)
def test_invalid_cache_options(arguments):
    parser = argparse.ArgumentParser()
    shader_cache.add_arguments(parser)
    with pytest.raises(ValueError):
        shader_cache.child_arguments(parser.parse_args(arguments))


def test_replay_passes_cache_to_bootstrap():
    tree = ast.parse((ROOT / "scripts/maintainer/candidate_replay.py").read_text())
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "bootstrap"
    )
    keyword = next(k for k in call.keywords if k.arg == "shader_cache")
    sentinel = object()
    assert (
        eval(
            compile(ast.Expression(keyword.value), "replay", "eval"),
            {"self": SimpleNamespace(shader_cache=sentinel)},
        )
        is sentinel
    )


@pytest.mark.parametrize(
    "entry,arguments,handler",
    [
        ("benchmark", ["run", "--request", "request.json", "--isaac-python", "python"], "command"),
        ("benchmark_portable", ["run", "--request", "request.json"], "portable"),
        (
            "benchmark_portable",
            ["smoke", "--assets", "assets", "--task", "task1-2", "--seeds", "0", "--output", "out"],
            "portable",
        ),
        (
            "benchmark_diagnostic",
            [
                "--assets",
                "assets",
                "--output",
                "out",
                "--package",
                "package",
                "--ik-python",
                "python",
                "--task",
                "task1-2",
                "--seed",
                "0",
                "--policy-url",
                "http://localhost:19000",
            ],
            "diagnostic",
        ),
    ],
)
def test_public_cli_cache_options(entry, arguments, handler, monkeypatch, tmp_path):
    import importlib

    monkeypatch.syspath_prepend(str(ROOT / "tools/simulator"))
    public = importlib.import_module("benchmark_candidate")
    seen = []
    monkeypatch.setattr(public, handler, lambda options: seen.append(options) or 0)
    module = importlib.import_module(entry)
    assert (
        module.main(
            [
                *arguments,
                "--shader-cache-seed",
                str(tmp_path / "seed"),
                "--shader-cache-digest",
                str(tmp_path / "digest"),
            ]
        )
        == 0
    )
    config = shader_cache.configuration(seen[0], tmp_path / "output")
    assert config["seed"] == tmp_path / "seed"
    assert config["seed_sha"] == tmp_path / "digest"


def test_public_worker_cache_plumbing():
    for relative in (
        "tools/simulator/benchmark_candidate.py",
        "scripts/maintainer/public_smoke.py",
    ):
        source = (ROOT / relative).read_text()
        tree = ast.parse(source)
        assignments = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "shader_cache" for t in n.targets)
        ]
        assert len(assignments) == 1
        value = assignments[0].value
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
        assert value.func.id == "configuration"
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "child_arguments"
        ]
        assert len(calls) == 1
        finalizers = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "finalize"
        ]
        assert len(finalizers) == 1
    tree = ast.parse((ROOT / "tools/simulator/benchmark_candidate.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "launch"
    ]
    assert len(calls) == 3
    assert all(
        any(
            k.arg == "cache_options" and isinstance(k.value, ast.Name) and k.value.id == "args"
            for k in call.keywords
        )
        for call in calls
    )


def test_parent_removes_only_receipted_private_cache(tmp_path):
    output = tmp_path / "output"
    options = SimpleNamespace(
        shader_cache_seed=tmp_path / "seed", shader_cache_digest=tmp_path / "digest"
    )
    directory = tmp_path / "output-kit"
    target = directory / "cache"
    target.mkdir(parents=True)
    cuda = directory / "cuda-cache"
    cuda.mkdir()
    shader_cache.finalize(options, output, passed=False)
    assert target.exists()
    (output / "simulation").mkdir(parents=True)
    receipt = output / "simulation/private-shader-cache.json"
    receipt.write_text(json.dumps({"private_cache": str(tmp_path / "unowned")}))
    with pytest.raises(ValueError, match="differs"):
        shader_cache.finalize(options, output, passed=False)
    assert target.exists()
    receipt.write_text(
        json.dumps({"private_cache": str(target), "private_cuda_cache": str(tmp_path / "unowned")})
    )
    with pytest.raises(ValueError, match="CUDA cache receipt differs"):
        shader_cache.finalize(options, output, passed=False)
    assert target.exists() and cuda.exists()
    receipt.write_text(json.dumps({"private_cache": str(target), "private_cuda_cache": str(cuda)}))
    shader_cache.finalize(options, output, passed=False)
    assert not target.exists()
    assert not cuda.exists()
    assert directory.is_dir()


@pytest.mark.parametrize("fails", [False, True])
def test_sdk_completion_argv_is_scoped(monkeypatch, fails):
    monkeypatch.setattr(app, "require_python312", lambda: None)
    seen = []

    def launch(config):
        seen.append(set(sys.argv))
        if fails:
            raise RuntimeError("constructor failed")
        return object()

    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=launch))
    original = list(sys.argv)
    if fails:
        with pytest.raises(RuntimeError, match="constructor failed"):
            app.simulation_app({"extra_args": sorted(FLAGS)})
    else:
        app.simulation_app({"extra_args": sorted(FLAGS)})
    assert seen[0] >= FLAGS
    assert sys.argv == original
    if not fails:
        app.simulation_app({})
        assert not FLAGS & seen[1]


@pytest.mark.parametrize("matching", [False, True])
@pytest.mark.parametrize("kind", ["cache", "cuda-cache"])
def test_portable_cache_admission_preserves_existing_files(tmp_path, matching, kind):
    directory = tmp_path / "run-kit"
    target = directory / kind
    target.mkdir(parents=True)
    marker = target / "existing"
    marker.write_text("retain")
    actual = directory if matching else tmp_path / "other-kit"
    error = FileExistsError if matching else ValueError
    with (
        pytest.raises(error),
        app.private_shader_cache(
            seed=tmp_path / "seed",
            seed_sha=tmp_path / "digest",
            private_root=tmp_path,
            name="run-kit",
            portable=True,
            argv=["worker", "--portable-root", str(actual)],
        ),
    ):
        pytest.fail("Existing or mismatched worker cache accepted")
    assert marker.read_text() == "retain"


@pytest.mark.parametrize("component", ["kit", "cuda"])
def test_partial_cache_bundle_is_rejected(tmp_path, component):
    seed = tmp_path / "seed"
    (seed / component).mkdir(parents=True)
    digest = tmp_path / "digest"
    digest.write_text(app.digest_tree(seed))
    with (
        pytest.raises(ValueError, match="Kit and CUDA cache directories"),
        app.private_shader_cache(
            seed=seed,
            seed_sha=digest,
            private_root=tmp_path,
            name="worker",
            portable=True,
            argv=["worker", "--portable-root", str(tmp_path / "worker")],
        ),
    ):
        pytest.fail("Incomplete cache bundle accepted")


@pytest.mark.archive
def test_cold_exposure_initialization_drives_render_products(tmp_path, monkeypatch):
    from test_cameras import CameraSDK
    from tower_sim.bench.ordinary_capture import OrdinaryFourCapture

    sdk = CameraSDK(monkeypatch, 1, tmp_path)
    key = "/rtx/post/tonemap/cm2Factor"
    sdk.controls.pop(key)
    original = sdk.rep.orchestrator.step

    def step(**kwargs):
        assert kwargs == dict(
            rt_subframes=2, delta_time=0.0, pause_timeline=True, wait_for_render=True
        )
        original(**kwargs)
        if sdk.counter == 4:
            sdk.controls[key] = 1.0

    monkeypatch.setattr(sdk.rep.orchestrator, "step", step)
    camera = OrdinaryFourCapture(
        sdk.sim, tmp_path, resources=sdk.resources, near_guard=lambda *args: 100.0
    )
    receipt = json.loads((tmp_path / "renderer-readiness.json").read_text())
    assert receipt["ready"] is True and receipt["render_steps"] == 2
    assert camera.controls.get(key) == 1.0
    assert all(value is not None for _, value in camera.expected_attributes)
    assert camera.steps == 4
    assert all(
        not product.hydra_texture.get_updates_enabled() for product in camera.products.values()
    )


@pytest.mark.parametrize("case", ["warm", "delayed", "timeout", "step_limit", "error"])
def test_exposure_readiness_is_bounded_and_restores_products(tmp_path, monkeypatch, case):
    from tower_sim.bench import ordinary_capture

    value = 1.0 if case == "warm" else None
    calls, toggles = [], []
    ticks = iter(range(100))
    monkeypatch.setattr(ordinary_capture.time, "monotonic", lambda: next(ticks))

    def step(subframes):
        nonlocal value
        calls.append(subframes)
        if case == "error":
            raise RuntimeError("render failure")
        if case == "delayed" and len(calls) == 2:
            value = 1.0

    capture = SimpleNamespace(
        output=tmp_path,
        controls=SimpleNamespace(get=lambda key: value),
        products={
            "camera": SimpleNamespace(
                hydra_texture=SimpleNamespace(set_updates_enabled=toggles.append)
            )
        },
        _step=step,
    )
    wait = ordinary_capture.OrdinaryFourCapture._wait_for_exposure
    if case in ("timeout", "step_limit"):
        with pytest.raises(TimeoutError, match="Renderer exposure initialization timed out"):
            wait(capture, timeout_s=2 if case == "timeout" else 240, max_steps=3)
        assert len(calls) == (1 if case == "timeout" else 3)
    elif case == "error":
        with pytest.raises(RuntimeError, match="render failure"):
            wait(capture)
    else:
        wait(capture)
        assert calls == ([] if case == "warm" else [2, 2])
    assert toggles == ([] if case == "warm" else [True, False])
    receipt = json.loads((tmp_path / "renderer-readiness.json").read_text())
    assert receipt["ready"] is (case in ("warm", "delayed"))
    assert receipt["value"] == value
