"""Fail-closed entry and provenance boundaries."""

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_policy_evaluation import authorization as authorization
from test_policy_evaluation import diagnostic_recording as diagnostic_recording
from test_replay_lifecycle import simulation as simulation
from tower_sim.bench import policy_entry, replay, source_receipt
from tower_sim.bench.resources import Resources, sha
from tower_sim.bench.source_receipt import ImportReceipt, validate_imports

from scripts.maintainer import candidate_policy

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("operation", ["run", "_worker"])
def test_formal_cli_is_disabled(operation, tmp_path):
    with pytest.raises(ValueError, match="Formal runs require a ReleaseVerification"):
        candidate_policy.main(
            [operation, "--request", str(tmp_path / "missing"), "--ik-python", "unused"]
        )


def test_existing_output_refused_before_launch(tmp_path, monkeypatch):
    from scripts.maintainer import gpu_baseline

    output = tmp_path / "out"
    run = output / "simulation/seed-1"
    run.mkdir(parents=True)
    for parent in (output, run):
        (parent / "diagnostic-result.json").write_text('{"original": true}')
    before = {p: p.read_bytes() for p in output.rglob("*.json")}
    request = tmp_path / "request.json"
    request.write_text(json.dumps(dict(output=str(output), gpu=0)))
    monkeypatch.setenv("TOWER_GPU_LOCK", str(tmp_path / "lock"))
    monkeypatch.setattr(
        gpu_baseline, "gpu_idle", lambda *a: pytest.fail("output reached admission")
    )
    with pytest.raises(ValueError, match="new directory"):
        candidate_policy.main(
            ["run", "--diagnostic", "--request", str(request), "--ik-python", "unused"]
        )
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("existing", ["parent", "seed"])
def test_parent_receipts_never_overwrite(existing, tmp_path):
    run = tmp_path / "simulation/seed-1"
    run.mkdir(parents=True)
    path = (tmp_path if existing == "parent" else run) / "diagnostic-result.json"
    path.write_text('{"original": true}')
    request = dict(output=str(tmp_path), task="task1-2", seeds=[1], assets=str(ROOT), duration=1)
    with pytest.raises(FileExistsError):
        candidate_policy.finish(request, 1, diagnostic=True)
    assert path.read_text() == '{"original": true}'


@pytest.mark.parametrize(
    "field",
    [
        None,
        "ready",
        "package_digest",
        "checkpoint_sha256",
        "horizon",
        "execute_steps",
        "control_hz",
        "denoise_steps",
        "seed",
        "prompt",
        "missing-protocol",
    ],
)
def test_diagnostic_preparation_frozen_contract(field, tmp_path, monkeypatch):
    protocol = Resources.from_roots(checkout=ROOT, assets=ROOT).read("protocol")
    package = dict(package_digest="p", checkpoint_sha256="c")
    path = tmp_path / "package.json"
    path.write_text(json.dumps(package))
    health = dict(
        ready=True,
        identity=dict(
            package, horizon=32, execute_steps=16, control_hz=10, denoise_steps=10, seed=42
        ),
    )
    request = dict(
        package=str(path),
        policy_url="ws://localhost:19000",
        task="task1-2",
        prompt=protocol["tasks"]["task1-2"]["prompt"],
    )
    if field == "prompt":
        request["prompt"] = "changed"
    elif field == "ready":
        health["ready"] = False
    elif field and field != "missing-protocol":
        health["identity"].pop(field)
    closed = []
    worker = NS(authorize=lambda: health, close=lambda: closed.append(True))
    monkeypatch.setattr(policy_entry, "validate_entry", lambda *a: None)
    monkeypatch.setattr(policy_entry.policy, "from_package", lambda **kw: worker)

    def prepare():
        return policy_entry.prepare(
            [],
            None,
            request,
            metrics=NS(call=lambda name, fn: fn()),
            solver=None,
            diagnostic=True,
            benchmark_protocol=None if field == "missing-protocol" else protocol,
        )

    if field is None:
        assert prepare() == (worker, None)
    else:
        with pytest.raises(ValueError):
            prepare()
        if field not in ("prompt", "missing-protocol"):
            assert closed == [True]


@pytest.mark.golden
def test_prefix_score_is_explicit(diagnostic_recording):
    output, run, task = diagnostic_recording
    result = policy_entry.collect_diagnostic(output, "task6-2", 1, 0, bench_task=task, duration=1)
    assert result["prefix_only"] is True
    assert result["score"]["prefix_only"] is True
    assert (
        result["benchmark_deadline_s"]
        == result["score"]["benchmark_deadline_s"]
        == task.identity.deadline_s
    )


@pytest.mark.parametrize("field", ["protocol", "runtime_manifest", "manifest", "package"])
def test_authorization_survives_optimization(authorization, field, tmp_path):
    request, health, rows, task = authorization
    path = Path(request[field])
    path.write_text(path.read_text() + " ")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps([request, health, rows]))
    code = """
import json,sys
from pathlib import Path
from tower_sim.bench.evaluation import EvaluationContext
from tower_sim.bench.resources import Resources
from tower_sim.bench.tasks import BenchTask
request,health,rows=json.loads(Path(sys.argv[1]).read_text())
task=BenchTask.from_resources('task1-2',Resources.from_roots(checkout=Path(sys.argv[2]),assets=Path(sys.argv[2])))
try: EvaluationContext(request,health,rows,task)
except ValueError: sys.exit(0)
sys.exit(9)
"""
    subprocess.run([sys.executable, "-O", "-c", code, str(inputs), str(ROOT)], check=True)


def test_no_optimization_sensitive_authorization():
    for relative in ("evaluation.py", "policy_entry.py", "policy.py", "replay.py"):
        path = Path(replay.__file__).with_name(relative)
        assert not any(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(path.read_text())))


@pytest.mark.parametrize("stem", ["gpu_baseline", "candidate_policy"])
def test_entry_digest_is_captured_before_restore(stem, tmp_path, monkeypatch):
    # Execute the real entry's stdlib preamble while modified, then restore it.
    original = ROOT / f"scripts/maintainer/{stem}.py"
    source = original.read_bytes()
    path = tmp_path / original.name
    path.write_bytes(source + b"\n# modified at process start\n")
    tree = ast.parse(source)
    stop = next(
        i
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__tower_entry_sha256__" for t in n.targets)
    )
    scope = dict(__file__=str(path))
    exec(compile(ast.Module(body=tree.body[: stop + 1], type_ignores=[]), str(path), "exec"), scope)
    path.write_bytes(source)
    module = NS(**scope)
    # Inventory under test is independent of other tests' temporary imports.
    monkeypatch.setattr(
        source_receipt, "sys", NS(modules={"__main__": module}, addaudithook=sys.addaudithook)
    )
    receipt = ImportReceipt()
    try:
        captured = receipt.collect()["modules"]["__main__"]
    finally:
        receipt.close()
    binding = dict(path=str(path), sha256=sha(path))
    with pytest.raises(ValueError, match="Forbidden"):
        validate_imports(
            {"modules": {"__main__": captured}},
            forbidden_stems={stem},
            allowed_modules={"__main__": [binding]},
        )


def test_policy_start_failure_closes_session(simulation):
    closed = []

    def start(*a):
        raise RuntimeError("startup failure")

    worker = NS(package={}, start=start, close=lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="startup failure"):
        simulation.run(policy=worker)
    assert closed == [True]


@pytest.mark.parametrize(
    "rejected,failed,at_terminal",
    [(True, True, True), (False, True, False), (False, False, True), (False, False, False)],
)
def test_append_camera_callsite_passes_eligibility(rejected, failed, at_terminal):
    # Execute the production call with independent terminal/guard inputs.
    tree = ast.parse(Path(replay.__file__).read_text())
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and ast.unparse(n.func) == "policy.append_camera"
    )
    calls = []
    scope = dict(
        policy=NS(append_camera=lambda *a, **k: calls.append(k)),
        index=0,
        archives=[object()],
        at_tick=40,
        newly_rejected=[0] if rejected else [],
        failures=[{} if failed else None],
        terminal_ticks=[40 if at_terminal else 80],
    )
    eval(compile(ast.Expression(call), "append-camera-callsite", "eval"), scope)
    assert calls == [dict(rejected=rejected, terminal=failed or at_terminal)]


def test_formal_prediction_identity_checked_before_ik_or_save(tmp_path):
    from test_policy_protocol import action_block, decisions
    from tower_sim.bench import control

    seen = []

    def verify(identity):
        seen.append(identity)
        raise ValueError("Inference identity changed")

    scope = dict(
        policy=NS(prediction=lambda *a: dict(identity={"drift": True})),
        evaluation=NS(verify_prediction=verify),
        pending={},
        index=0,
        observation={},
        tick=0,
        control=control,
        row=dict(output=tmp_path),
        failures=[None],
        terminal_ticks=[None],
    )
    assert decisions(scope, action_block(Path(replay.__file__))) == "abort"
    assert seen == [{"drift": True}]
    assert not list(tmp_path.iterdir())


def test_rpc_duration_excludes_payload_encoding():
    from tower_sim.bench.metrics import Metrics

    clock = [0]
    events = []
    metrics = Metrics(
        NS(emit=events.append, flush=lambda: None), lambda *a: "r", monotonic_ns=lambda: clock[0]
    )

    def predict(payload, *, headers, rpc_observer):
        clock[0] = 10_000_000_000  # PNG decoding precedes the exchange timer.
        rpc_observer("rpc_start")
        clock[0] += 2_000_000_000
        rpc_observer("rpc_end", transport="websocket")
        clock[0] += 5_000_000_000  # Identity checks follow the exchange timer.
        return {}

    metrics.predict(NS(predict=predict), {}, headers={"X-Baseline-Request-ID": "r"})
    assert events[-1]["rpc_s"] == 2.0
    assert metrics.stats["policy.rpc_elapsed_sum"] == [2.0, 1]
    # Request identity and elapsed duration remain observable.
    assert list(events[-1])[-3:] == ["request_id", "rpc_s", "transport"]


def test_library_prepare_refuses_formal_runs(monkeypatch):
    built = []
    monkeypatch.setattr(policy_entry.policy, "from_package", lambda **kw: built.append(kw))
    with pytest.raises(ValueError, match="Formal runs require a ReleaseVerification"):
        policy_entry.prepare([], None, {}, metrics=None, solver=None)
    assert built == []
