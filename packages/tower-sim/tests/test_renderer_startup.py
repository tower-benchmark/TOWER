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
    with eval(compile(ast.Expression(call), "bootstrap", "eval"), scope) as launch:
        launch(dict(headless=True, experience="provided.kit"))
    assert set(json.loads((output / "launch.json").read_text())["extra_args"]) >= FLAGS


def test_seed_roundtrip_and_corruption(tmp_path):
    warm = tmp_path / "warm"
    with app.private_shader_cache(
        seed=None, seed_sha=None, private_root=tmp_path, name="warm", warmup=True, argv=[]
    ) as receipt:
        (warm / "nv_shadercache" / "shader").write_bytes(b"compiled")
    output = tmp_path / "warm-run"
    (output / "simulation").mkdir(parents=True)
    (output / "simulation/private-shader-cache.json").write_text(json.dumps(receipt))
    options = SimpleNamespace(shader_cache_warmup=warm)
    shader_cache.finalize(options, output, passed=False)
    assert not (warm / "digest.sha256").exists()
    shader_cache.finalize(options, output, passed=True)
    parser = argparse.ArgumentParser()
    shader_cache.add_arguments(parser)
    args = parser.parse_args(
        [
            "--shader-cache-seed",
            str(warm / "nv_shadercache"),
            "--shader-cache-digest",
            str(warm / "digest.sha256"),
        ]
    )
    copied = parser.parse_args(shader_cache.child_arguments(args))
    config = shader_cache.configuration(copied, tmp_path / "run")
    with app.private_shader_cache(**config, argv=[]) as receipt:
        assert (Path(receipt["private_cache"]) / "shader").read_bytes() == b"compiled"
    assert not (tmp_path / "run-shader-cache").exists()
    (warm / "nv_shadercache" / "shader").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differs"), app.private_shader_cache(**config, argv=[]):
        pytest.fail("Corrupt seed accepted")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--shader-cache-seed", "seed"],
        ["--shader-cache-digest", "digest"],
        [
            "--shader-cache-seed",
            "seed",
            "--shader-cache-digest",
            "digest",
            "--shader-cache-warmup",
            "warm",
        ],
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
    directory = tmp_path / "output-shader-cache"
    target = directory / "nv_shadercache"
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
    assert not directory.exists()


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
