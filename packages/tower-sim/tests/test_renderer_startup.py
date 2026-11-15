"""Renderer startup contracts without importing the native SDK."""

import argparse
import ast
import json
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


def test_seed_roundtrip_and_corruption(tmp_path):
    warm = tmp_path / "warm"
    output = tmp_path / "warm-run"
    options = SimpleNamespace(shader_cache_warmup=warm)
    config = shader_cache.configuration(options, output)
    argv = ["worker", "--portable-root", str(tmp_path / "warm-run-kit")]
    with app.private_shader_cache(**config, argv=argv) as receipt:
        target = Path(receipt["private_cache"])
        (target / "shadercache").mkdir()
        (target / "shadercache/shader").write_bytes(b"compiled")
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
    (complete / "cache/shadercache/shader").write_bytes(b"changed")
    with (
        pytest.raises(RuntimeError, match="differs"),
        app.private_shader_cache(
            **config, argv=["worker", "--portable-root", str(tmp_path / "run-kit")]
        ),
    ):
        pytest.fail("Corrupt seed accepted")


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
    shader_cache.finalize(options, output, passed=False)
    assert target.exists()
    (output / "simulation").mkdir(parents=True)
    receipt = output / "simulation/private-shader-cache.json"
    receipt.write_text(json.dumps({"private_cache": str(tmp_path / "unowned")}))
    with pytest.raises(ValueError, match="differs"):
        shader_cache.finalize(options, output, passed=False)
    assert target.exists()
    receipt.write_text(json.dumps({"private_cache": str(target)}))
    shader_cache.finalize(options, output, passed=False)
    assert not target.exists()
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
def test_portable_cache_admission_preserves_existing_files(tmp_path, matching):
    directory = tmp_path / "run-kit"
    target = directory / "cache"
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
