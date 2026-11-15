import ast
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from tower_sim.bench.metrics import Metrics, TrialRequestID
from tower_sim.bench.observation_validity import DeliveryGuard, compare_pixels
from tower_sim.bench.termination import abort_evidence

from scripts.maintainer.archive_root import ArchivedPath, ArchivedToolPath
from tests.characterization.runtime_scenarios import ROOT


@pytest.mark.archive
@pytest.mark.parametrize("tick", [0, 1, 13, 20, 168000, -1, 168001, 1.5])
def test_termination_equivalence(frozen_loader, tick):
    original = frozen_loader("_p705_termination", ArchivedToolPath("termination.py"))
    outputs = []
    for fn in (original.abort_evidence, abort_evidence):
        calls = []

        def record(name, calls=calls):
            return lambda tick: calls.append((name, tick))

        try:
            result = fn(
                tick=tick,
                state_ticks=[0],
                camera_ticks=[0],
                advance_held_tick=record("held"),
                record_state=record("state"),
                record_camera=record("camera"),
            )
        except ValueError as exc:
            result = str(exc)
        outputs.append((result, calls))
    assert outputs[0] == outputs[1]


@pytest.mark.archive
@pytest.mark.parametrize(
    "kind", ["ok", "stale", "nan", "shape", "dark", "reference", "mismatch", "error"]
)
def test_observation_delivery_equivalence(frozen_loader, monkeypatch, kind):
    monkeypatch.syspath_prepend(str(ArchivedToolPath("")))
    original = frozen_loader(
        "_p705_delivery", ArchivedToolPath("observation_validity.py")
    ).DeliveryGuard()
    new = DeliveryGuard()
    pixels = np.zeros((4, 240, 320, 3), np.uint8)
    quality = {
        i: dict(valid=kind not in ("dark", "reference", "mismatch", "error")) for i in range(4)
    }

    def reference():
        if kind == "error":
            raise RuntimeError("fixture failed")
        return pixels.copy() + (20 if kind == "mismatch" else 0), dict(
            physical_state_preserved=True
        )

    options = dict(
        render_step=1,
        synchronized=True,
        paused=True,
        delta_time_s=0,
        independent_reference=reference if kind in ("reference", "mismatch", "error") else None,
    )
    if kind == "shape":
        pixels = pixels[0]
    stamp = float("nan") if kind == "nan" else 0.0
    a, b = (
        original.validate(pixels, stamp, quality, **options),
        new.validate(pixels, stamp, quality, **options),
    )
    # NaN is intentionally rejected but retained in the diagnostic stamp.
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    if kind == "stale":
        assert original.validate(pixels, 0.0, quality, **options) == new.validate(
            pixels, 0.0, quality, **options
        )


@pytest.mark.archive
def test_pixel_comparison_equivalence(frozen_loader):
    old = frozen_loader(
        "_p705_pixels", ArchivedToolPath("serial_camera_reference.py")
    ).compare_pixels
    image = np.random.default_rng(2).integers(0, 256, (4, 240, 320, 3), dtype=np.uint8)
    for delta in (0, 1, 9):
        assert compare_pixels(image, image + delta) == old(image, image + delta)


@pytest.mark.archive
def test_metrics_equivalence_and_injected_io(frozen_loader, monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_BASELINE_METRICS", str(tmp_path / "original.jsonl"))
    monkeypatch.setenv("TOWER_BASELINE_TRIAL", "trial")
    original = frozen_loader("_p705_metrics", ArchivedToolPath("baseline_metrics.py")).METRICS
    records = []
    sink = SimpleNamespace(
        emit=lambda row: records.append(json.loads(json.dumps(row))), flush=lambda: None
    )
    clock = SimpleNamespace(value=0)

    def now():
        clock.value += 1000000000
        return clock.value

    import time

    monkeypatch.setattr(time, "monotonic_ns", now)
    monkeypatch.setattr(time, "time_ns", lambda: 123)
    new = Metrics(sink, TrialRequestID("trial"), epoch_ns=lambda: 123, monotonic_ns=now)
    for meter in (original, new):
        clock.value = 0
        for tick in (0, 640):
            meter.tick(tick)
            assert meter.headers(7, tick) == {
                "Content-Type": "application/json",
                "X-Baseline-Request-ID": f"trial:7:{tick}",
                "X-Baseline-Capture": "",
            }
            assert meter.call("fn", lambda: 42) == 42
        meter.end_loop()
        meter.close()
    old_records = [
        json.loads(line) for line in (tmp_path / "original.jsonl").read_text().splitlines()
    ][1:]
    assert records == old_records
    new.close()
    assert records == old_records


def test_metrics_trace_barrier_and_decorator():
    events, profile = [], []
    sink = SimpleNamespace(emit=events.append, flush=lambda: None)
    profiler = SimpleNamespace(
        **{
            name: lambda *a, name=name: profile.append((name, a))
            for name in ("start", "stop", "range_push", "range_pop")
        }
    )
    meter = Metrics(sink, TrialRequestID("trial"), trace=True, profiler=profiler)
    meter.tick(1280)
    assert meter.headers(1, 1280)["X-Baseline-Capture"] == "start"
    assert meter.decorated("fn")(lambda x: x + 1)(2) == 3
    meter.tick(2560)
    assert meter.headers(1, 2560)["X-Baseline-Capture"] == "stop"
    meter.barrier(
        lambda: 3, lambda a, b: events.append(dict(unchanged=a == b)), wait_for_release=lambda: None
    )
    meter.close()
    assert [row[0] for row in profile] == ["start", "range_push", "range_pop", "stop"]
    assert dict(unchanged=True) in events
    with pytest.raises(ValueError):
        Metrics(sink, TrialRequestID("trial"), trace=True)


def test_metrics_ws_error_boundary():
    records = []
    meter = Metrics(
        SimpleNamespace(emit=records.append, flush=lambda: None), TrialRequestID("trial")
    )

    def predict(payload, *, headers, rpc_observer):
        rpc_observer("rpc_start")
        try:
            raise RuntimeError("synthetic failure")
        finally:
            rpc_observer("rpc_end", transport="websocket")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        meter.predict(SimpleNamespace(predict=predict), {}, headers=meter.headers(0, 0))
    assert [r["event"] for r in records] == ["rpc_start", "rpc_end"]
    assert records[-1]["transport"] == "websocket"
    assert meter.stats["policy.rpc_elapsed_sum"][1] == 1


def environment_reads(source):
    """Find os environment access, including aliases and .get/copy/index forms."""
    tree = ast.parse(source)
    os_names = {"os"}
    env_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_names.update(a.asname or a.name for a in node.names if a.name == "os")
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            env_names.update(
                a.asname or a.name for a in node.names if a.name in {"environ", "getenv"}
            )
    found = []

    def visit(node, scope=None):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = node.name
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in os_names
            and node.attr in {"environ", "getenv"}
        ) or (isinstance(node, ast.Name) and node.id in env_names):
            found.append((scope, node.lineno))
        # A from-import itself reads the environment object at import time.
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "os"
            and any(a.name in {"environ", "getenv", "*"} for a in node.names)
        ):
            found.append((scope, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree)
    return found


@pytest.mark.parametrize(
    "source",
    [
        "import os; os.environ.get('X')",
        "import os; os.environ['X']",
        "import os; os.environ.copy()",
        "import os; os.getenv('X')",
        "import os as process; process.environ.get('X')",
        "from os import getenv as read; read('X')",
        "from os import environ as env; env.get('X')",
    ],
)
def test_environment_guard_detects_reads(source):
    assert environment_reads(source)


def test_static_dependency_direction_and_environment_boundary():
    for package, forbidden in (
        (ROOT / "packages/tower-sim/src", {"jenga_generator", "replay_runner_" + "isaac_sim"}),
        (ROOT / "src/tower-eval/src", {"tower_sim"}),
    ):
        for path in package.rglob("*.py"):
            source = path.read_text()
            tree = ast.parse(source)
            parents = {
                child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
            }
            for node in ast.walk(tree):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                roots = {name.split(".")[0] for name in names}
                retained_edges = {}
                relative = path.relative_to(package).as_posix().removeprefix("tower_sim/")
                for name in names:
                    if name.split(".")[0] in forbidden:
                        assert name in retained_edges.get(relative, set()), (path, name)
                if "tower-sim" in str(package) and roots & {
                    "isaacsim",
                    "omni",
                    "pxr",
                    "carb",
                    "jenga_generator",
                    "replay_runner_" + "isaac_sim",
                }:
                    parent = parents.get(node)
                    while parent is not None and not isinstance(
                        parent, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        parent = parents.get(parent)
                    assert parent is not None, (path, node.lineno, "eager native/bridge import")
            if "tower-sim" in str(package):
                for scope, line in environment_reads(source):
                    assert (path.name, scope) in {
                        ("ik_client.py", "_subprocess_environment"),
                        ("app.py", "process_environment"),
                        ("eula.py", "require_eula"),
                    }, (path, line)
                for text in ("spec_from_file_location", "sys.path"):
                    assert text not in source, (path, text)
                if relative == "bench/public_surface.py":
                    # The release gate names the directory it rejects; it does
                    # not read from that directory or import archived code.
                    source = source.replace('"benchmark-202611"', '"<forbidden-archive-directory>"')
                assert "benchmark-202611" not in source.replace(
                    "tower-benchmark-202611-v1", "<protocol-schema>"
                ), path


@pytest.mark.archive
def test_tower_sim_executed_import_closure(tmp_path):
    # Replay explicitly depends on canonical tower_eval scoring. Keep the frozen
    # bundle and native imports blocked; the AST test forbids the back-edge.
    script = """
import importlib, importlib.abc, pkgutil, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tower_sim.generation' or fullname.startswith('tower_sim.generation.'):
            raise RuntimeError(fullname)
        if fullname.split('.')[0] in {'jenga_generator','replay_runner_' + 'isaac_sim','isaacsim','omni','pxr','carb','lula','torch','pinocchio','casadi'}:
            raise RuntimeError(fullname)
sys.meta_path.insert(0, Guard())
import tower_sim
import tower_eval.simulator
assert "tower_eval.simulator.isaac" not in sys.modules
from pathlib import Path
frozen = Path(sys.argv[1]).resolve()
def walk(package):
    for item in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        if item.name in {"tower_sim.backend.model", "tower_sim.generation"}:
            continue
        module = importlib.import_module(item.name)
        if item.ispkg:
            walk(module)
walk(tower_sim)
loaded = [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)]
assert not [path for path in loaded if path.is_relative_to(frozen)], frozen
"""
    # Compare against the bundle path, not a substring: the frozen IK conda
    # environment is itself named tower-benchmark-202611-ik.
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ArchivedPath("")),
            str(ROOT / "src/tower-datagenerator"),
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        check=True,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_metrics_rejects_nonfinite_before_sink(value):
    records = []
    meter = Metrics(
        SimpleNamespace(emit=records.append, flush=lambda: None), TrialRequestID("trial")
    )
    with pytest.raises(ValueError):
        meter.emit("invalid", nested={"values": [value]})
    assert records == []
