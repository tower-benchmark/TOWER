"""Frozen executable request/action blocks versus the integrated candidate hooks."""

import ast
import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from tower_eval.policies.legacy_predict import LegacyPredictAdapter
from tower_sim.bench import control, policy, policy_entry, replay
from tower_sim.bench.metrics import Metrics, TrialRequestID
from tower_sim.bench.mme_history import EpisodeHistory
from tower_sim.bench.policy_ik import PolicyIK
from tower_sim.bench.resources import save

ROOT = Path(__file__).resolve().parents[3]
# Reuse the raw-header server and frozen AST loader, not its adapter under test.
parity = NS(
    **runpy.run_path(str(ROOT / "src/tower-eval/tests/unit/test_legacy_transport_parity.py"))
)
server = parity.server
frozen_modules = parity.frozen_modules
worker_name = parity.worker_name


def metrics(events=None):
    return Metrics(
        NS(emit=(events.append if events is not None else lambda record: None), flush=lambda: None),
        TrialRequestID(parity.TRIAL),
        trace=True,
        profiler=NS(
            start=lambda: None,
            stop=lambda: None,
            range_push=lambda name: None,
            range_pop=lambda: None,
        ),
    )


def action_block(path):
    tree = ast.parse(path.read_text())
    owner = parity.one(
        ast.walk(tree),
        lambda n: (
            isinstance(n, ast.Try)
            and any(
                isinstance(b, ast.Assign)
                and ast.unparse(b.targets[0]) == "(endpoints, diagnostics)"
                for b in n.body
            )
        ),
    )
    return parity.statements(path.name, [owner])


def decisions(scope, code):
    try:
        exec(code, scope)
        return "reject" if scope["failures"][0] else "enqueue"
    except (RuntimeError, ValueError, AssertionError, OSError):
        return "abort"


@pytest.mark.archive
@pytest.mark.parametrize(
    "case",
    [
        "action",
        "invalid422",
        "aperture",
        "aperture-extra",
        "5xx",
        "html-5xx",
        "drift",
        "nan",
        "inf-width",
        "ik-rejection",
        "timeout",
    ],
)
@pytest.mark.parametrize("tick", [0, 640])
def test_wire_decisions_and_evidence(
    server, frozen_modules, worker_name, tmp_path, monkeypatch, case, tick
):
    old = parity.FrozenClient(
        worker_name, frozen_modules.metrics, frozen_modules.control, server.url
    )
    observation, archive, history, _ = parity.inputs(frozen_modules, tick)
    server.reply = parity.prediction(observation)
    if case in ("invalid422", "aperture", "aperture-extra", "5xx"):
        server.status = 500 if case == "5xx" else 422
        server.reply = {"code": "invalid_policy_output", "error": "recorded invalid output"}
        if case.startswith("aperture"):
            server.reply = {"code": "invalid_observation", "error": "Negative gripper aperture"}
            if case == "aperture-extra":
                server.reply["extra"] = True
        server.reply["profiling"] = {"server_s": 0.1}
    elif case == "drift":
        server.reply["identity"]["extra"] = True
    elif case == "nan":
        server.reply["joint_encoder_targets"][0][0][0] = float("nan")
    elif case == "inf-width":
        server.reply["gripper_width_m"][0][0] = float("inf")
    elif case == "ik-rejection":
        server.reply.update(
            mode="eef_delta", eef_encoder_targets=np.tile(np.eye(4), (32, 2, 1, 1)).tolist()
        )
    if case == "html-5xx":
        server.status, server.reply = 500, b"<html>unavailable</html>"
    solver_result = dict(
        accepted=False, joints=None, diagnostics=[dict(reason="pose_bounds_or_time_rejection")]
    )
    solver = NS(solve=lambda *a, **k: solver_result, identity={})
    # Inject only the frozen native-service boundary. It must reject without enqueue.
    monkeypatch.setitem(
        frozen_modules.control.encoder_endpoints.__globals__, "deployed_solve", solver.solve
    )
    package = dict(
        package_digest=parity.IDENTITY["package_digest"],
        checkpoint_sha256="checkpoint",
        kind="MME-VLA" if "_mme" in worker_name else "ordinary",
    )
    row = dict(candidate={"seed": parity.SEED}, output=tmp_path / "candidate")
    row["output"].mkdir()
    (row["output"] / "chunks").mkdir()
    candidate_events = []
    current = policy.from_package(
        package=package,
        url=server.url,
        prompt=old.scope["args"].prompt,
        task="task1-2",
        metrics=metrics(candidate_events),
        solver=solver,
    )
    old.startup_health()
    current.authorize()
    old.health()
    health = current.start(tmp_path)
    current.initialize([row], parity.CAMERAS, health)
    if current.histories:
        candidate_history = current.histories[0]
        candidate_history.prefix.episode_id = history.episode_id
        for at in range(0, tick + 1, 40):
            candidate_history.append_validated(
                NS(times=[at / 400], _latest_pixels=parity.frame(at)),
                at,
                rejected=False,
                terminal=False,
            )
    old_output = tmp_path / "frozen"
    old_output.mkdir()
    (old_output / "chunks").mkdir()
    old.scope.update(
        row=dict(row, output=old_output),
        observation=observation,
        archives=[archive],
        histories=[history],
        tick=tick,
        failures=[None],
        terminal_ticks=[None],
        failure_terminal_tick=replay.failure_terminal_tick,
    )
    old.scope["worker"].save = frozen_modules.runtime.save
    exec(old.request_code, old.scope)
    if case == "timeout":
        original = __import__("urllib.request", fromlist=["urlopen"]).urlopen

        def timeout(req, **kwargs):
            if getattr(req, "full_url", "").endswith("/predict"):
                raise TimeoutError("recorded timeout")
            return original(req, **kwargs)

        monkeypatch.setattr("urllib.request.urlopen", timeout)
    old_queue = []
    new_queue = []
    old.scope["queues"] = [NS(enqueue=lambda *a: old_queue.append(a))]
    old.scope["pending"] = {
        0: NS(result=lambda: frozen_modules.metrics.http_json(old.scope["request"]))
    }
    begin = len(server.requests)
    old_decision = decisions(old.scope, action_block(parity.FROZEN / worker_name))
    old_requests = server.requests[begin:]
    begin = len(server.requests)
    pending = current.prefetch({0: observation}, [archive], tick)
    scope = dict(
        index=0,
        row=row,
        tick=tick,
        observation=observation,
        policy=current,
        evaluation=None,
        pending=pending,
        metrics=current.metrics,
        control=control,
        save=save,
        queues=[NS(enqueue=lambda *a: new_queue.append(a))],
        failures=[None],
        terminal_ticks=[None],
        failure_terminal_tick=replay.failure_terminal_tick,
    )
    new_decision = decisions(scope, action_block(Path(replay.__file__)))
    current.close()
    frozen_modules.metrics.stream.flush()
    frozen_events = [
        json.loads(line) for line in frozen_modules.metrics.path.read_text().splitlines()
    ]

    def rpc(events):
        return [
            {k: v for k, v in e.items() if k not in {"epoch_ns", "mono_ns", "rpc_s"}}
            for e in events
            if e["event"] in {"rpc_start", "rpc_end"}
        ]

    assert [e["event"] for e in rpc(candidate_events)] == ["rpc_start", "rpc_end"]
    assert rpc(candidate_events) == rpc(frozen_events)
    assert new_decision == old_decision
    assert server.requests[begin:] == old_requests
    assert scope["failures"] == old.scope["failures"]
    assert scope["terminal_ticks"] == old.scope["terminal_ticks"]
    assert len(new_queue) == len(old_queue)
    if new_queue:
        np.testing.assert_array_equal(new_queue[0][1], old_queue[0][1])

    def files(path):
        return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*.json")}

    assert files(row["output"]) == files(old_output)
    assert new_decision == (
        "enqueue"
        if case == "action"
        else "abort"
        if case in ("aperture-extra", "5xx", "html-5xx", "drift", "timeout", "nan", "inf-width")
        else "reject"
    )


@pytest.mark.archive
@pytest.mark.parametrize("tick", [0, 640, 1280, 2560])
def test_mme_prefix_exact(frozen_modules, tick):
    _, _, old, _ = parity.inputs(frozen_modules, tick)
    new = EpisodeHistory(old.episode_id, parity.CAMERAS, 256)
    for at in range(0, tick + 1, 40):
        new.append_validated(
            NS(times=[at / 400], _latest_pixels=parity.frame(at)),
            at,
            rejected=False,
            terminal=False,
        )
    assert new.request(tick) == old.request(tick)
    assert not new.append_validated(None, tick + 40, rejected=True, terminal=False)
    assert not new.append_validated(None, tick + 40, rejected=False, terminal=True)
    new.close()
    old.close()
    assert new.receipt() == old.receipt()


@pytest.mark.parametrize(
    "identity",
    [
        {"package_digest": "transport-fixture"},
        parity.IDENTITY,
        dict(parity.IDENTITY, nested={"extra": [1, 2]}),
    ],
)
def test_package_authorization_and_full_identity(server, identity, tmp_path):
    server.health["identity"] = copy.deepcopy(identity)
    adapter = LegacyPredictAdapter(
        server.url, expected_identity={"package_digest": "transport-fixture", "unused": True}
    )
    assert adapter.health() == server.health
    observation = dict(images=dict(zip(parity.CAMERAS, parity.frame(0))))
    server.reply = {"identity": copy.deepcopy(identity)}
    assert adapter.predict(observation) == server.reply
    server.reply["identity"]["drift"] = True
    with pytest.raises(ValueError, match="full identity"):
        adapter.predict(observation)
    server.health["identity"]["package_digest"] = "wrong"
    with pytest.raises(ValueError, match="authorized package"):
        adapter.health()


def test_public_rejects_command_replay_before_other_inputs():
    with pytest.raises(ValueError, match="reference-replay"):
        policy_entry.validate_entry({"command_replay": True}, [], None)


def test_exact_flip_tripwire_without_native_call():
    from nero_kinematics.fk.numpy_oracle import fk

    calls = []
    native = NS(solve=lambda *a, **k: calls.append((a, k)))
    solver = PolicyIK(None, client=native)
    seed = np.tile((control.GENERATION_LOWER + control.GENERATION_UPPER) / 2, (2, 1))
    poses = np.tile(fk(seed[0]), (16, 2, 1, 1))
    poses[0, 0, :3, :3] @= np.diag([1, -1, -1])
    result = solver.solve(poses, seed)
    assert result["accepted"] is False and result["joints"] is None
    assert result["diagnostics"][0]["reason"] == "reversed_180_grasp"
    assert not calls


@pytest.mark.parametrize(
    "reason",
    [
        "seed_outside_kill_line",
        "pose_bounds_or_time_rejection",
        "step_from_unclipped_seed",
        "physical_pose_bounds_or_time",
    ],
)
def test_planner_rejection_never_retries(reason):
    from nero_kinematics.fk.numpy_oracle import fk

    calls = []
    result = dict(accepted=False, joints=None, diagnostics=[dict(reason=reason)])

    def solve(*a, **k):
        calls.append((a, k))
        return result

    solver = PolicyIK(None, client=NS(solve=solve))
    seed = np.zeros((2, 7))
    poses = np.tile(fk(seed[0]), (16, 2, 1, 1))
    assert solver.solve(poses, seed) is result
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0][0], poses)


def test_policy_imports_without_optional_image_or_simulator_dependencies():
    import subprocess
    import sys

    script = """import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'PIL','h5py','isaacsim','omni','pxr','carb','pinocchio','casadi','jenga_generator','batch_policy_worker'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import tower_sim.bench.policy_entry,tower_sim.bench.policy_ik
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("degrees,tilt", [(179, 0), (180, 0.2), (120, 0)])
def test_near_flip_targets_pass_unchanged_once(degrees, tilt):
    from nero_kinematics.fk.numpy_oracle import fk
    from scipy.spatial.transform import Rotation

    calls = []
    result = dict(accepted=False)

    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    solver = PolicyIK(None, client=NS(solve=solve))
    seed = np.tile((control.GENERATION_LOWER + control.GENERATION_UPPER) / 2, (2, 1))
    poses = np.tile(fk(seed[0]), (16, 2, 1, 1))
    poses[0, 0, :3, :3] @= Rotation.from_euler("xy", [degrees, tilt], degrees=True).as_matrix()
    original = poses.copy()
    assert solver.solve(poses, seed) is result
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0][0], original)
