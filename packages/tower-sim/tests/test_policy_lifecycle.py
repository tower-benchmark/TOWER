"""Policy requests drive the real shared loop; compare frozen status serialization."""

import ast
import json
import shutil
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_policy_protocol import action_block, decisions, parity
from test_policy_protocol import frozen_modules as frozen_modules
from test_policy_protocol import server as server
from test_replay_lifecycle import simulation as simulation
from tower_sim.bench import control, policy, policy_entry, replay


def frozen_finalizer():
    tree = parity.tree("batch_policy_worker.py")
    trial = parity.one(
        ast.walk(tree),
        lambda n: (
            isinstance(n, ast.Try)
            and any(
                isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "failure" for x in n.body
            )
        ),
    )
    return parity.statements("batch_policy_worker.py", [trial])


@pytest.mark.archive
@pytest.mark.parametrize(
    "case", ["action", "invalid422", "aperture", "5xx", "drift", "nan", "ik-rejection"]
)
@pytest.mark.parametrize("mme", [False, True])
def test_policy_shared_lifecycle_status(
    simulation, server, frozen_modules, case, mme, tmp_path, monkeypatch
):
    s = simulation
    camera = s.camera
    camera.names = {name: str(i) for i, name in enumerate(parity.CAMERAS)}
    # Constant bounded observed encoder pose supplied by the CPU simulation fixture.
    observation = s.factories.observation()
    server.reply = parity.prediction(observation)
    if case in ("invalid422", "aperture", "5xx"):
        server.status = 500 if case == "5xx" else 422
        server.reply = {"code": "invalid_policy_output", "error": "recorded invalid output"}
        if case == "aperture":
            server.reply = {"code": "invalid_observation", "error": "Negative gripper aperture"}
    elif case == "drift":
        server.reply["identity"]["extra"] = True
    elif case == "nan":
        server.reply["gripper_width_m"][0][0] = float("nan")
    elif case == "ik-rejection":
        server.reply.update(
            mode="eef_delta", eef_encoder_targets=np.tile(np.eye(4), (16, 2, 1, 1)).tolist()
        )
    package = dict(
        package_digest=parity.IDENTITY["package_digest"],
        checkpoint_sha256="checkpoint",
        kind="MME-VLA" if mme else "ordinary",
    )
    from test_policy_protocol import metrics

    worker = policy.from_package(
        package=package,
        url=server.url,
        prompt="recorded",
        task="task1-2",
        metrics=metrics(),
        solver=NS(
            identity={},
            solve=lambda *a, **k: dict(
                accepted=False,
                joints=None,
                diagnostics=[dict(reason="pose_bounds_or_time_rejection")],
            ),
        ),
    )
    # Drive the frozen response/action decision independently before the candidate.
    frozen = parity.FrozenClient(
        "batch_policy_worker.py", frozen_modules.metrics, frozen_modules.control, server.url
    )
    frozen.startup_health()
    frozen.health()
    dest = tmp_path / "frozen-final"
    dest.mkdir()
    (dest / "chunks").mkdir()
    _, archive, _, _ = parity.inputs(frozen_modules, 0)
    frozen.scope.update(
        row=dict(s.row, output=dest),
        observation=observation,
        archives=[archive],
        tick=0,
        failures=[None],
        terminal_ticks=[None],
        queues=[NS(enqueue=lambda *a: None)],
    )
    failure_fn = parity.one(
        parity.tree("batch_policy_worker.py").body,
        lambda n: isinstance(n, ast.FunctionDef) and n.name == "failure_terminal_tick",
    )
    exec(parity.statements("batch_policy_worker.py", [failure_fn]), frozen.scope)
    frozen.scope["worker"].save = frozen_modules.runtime.save
    monkeypatch.setitem(
        frozen_modules.control.encoder_endpoints.__globals__, "deployed_solve", worker.solver.solve
    )
    exec(frozen.request_code, frozen.scope)
    frozen.scope["pending"] = {
        0: NS(result=lambda: frozen_modules.metrics.http_json(frozen.scope["request"]))
    }
    frozen_decision = decisions(
        frozen.scope, action_block(parity.FROZEN / "batch_policy_worker.py")
    )
    complete = frozen_decision != "abort"
    failure = frozen.scope["failures"][0]
    terminal = (
        frozen.scope["terminal_ticks"][0]
        if failure
        else s.task.duration_ticks
        if complete
        else None
    )
    stamp = terminal / 400 if terminal is not None else 0.0
    server.requests.clear()
    worker.authorize()
    if case in ("5xx", "drift", "nan"):
        with pytest.raises((RuntimeError, ValueError)):
            s.run(policy=worker)
    else:
        assert s.run(policy=worker) == 0
    worker.close()
    status = json.loads((s.row["output"] / "status.json").read_text())
    assert status["diagnostic_only"] and not status["valid_trial"]
    assert status["evidence_complete"] == (case not in ("5xx", "drift", "nan"))
    assert status["policy_action_failure"] == (case in ("invalid422", "aperture", "ik-rejection"))
    assert s.events[-1] == "camera.close"
    requests = [r for r in server.requests if r[1] == "/predict"]
    assert len(requests) == 1
    assert all(r[1] in ("/health", "/predict") for r in server.requests)
    assert len([r for r in server.requests if r[1] == "/health"]) == 2
    payload = json.loads(requests[0][-1])
    assert "seed" not in payload
    if mme:
        assert payload["history"]["indices"] == [0]
        assert payload["episode_id"] == str(s.row["output"].resolve())
        if status["evidence_complete"]:
            history = json.loads((s.row["output"] / "mme-history.json").read_text())
            assert history["closed"] and history["extra_render_or_physics_calls"] == 0
    if case == "ik-rejection":
        assert (
            json.loads((s.row["output"] / "chunks/00000-ik-rejected.json").read_text())[0]["reason"]
            == "pose_bounds_or_time_rejection"
        )
    # Status inputs come from the frozen decision, never candidate status/failure.
    shutil.copytree(s.row["output"], dest, dirs_exist_ok=True)
    failure_file = s.row["output"] / "policy-action-failure.json"
    scope = dict(
        index=0,
        output=dest,
        row=s.row,
        worker=NS(save=frozen_modules.runtime.save, sha=lambda p: None),
        evidence=NS(write=lambda *a: None, records=[[{"time": 0.0}, {"time": stamp}]]),
        commands=[json.loads((dest / "commands.json").read_text())],
        camera=s.camera,
        archives=[NS(times=[0.0, stamp], close=lambda **k: None)],
        render_checks=1,
        args=NS(output=tmp_path / "unused"),
        failures=[failure],
        successes=[None],
        terminal_ticks=[terminal],
        ended=[complete],
        complete=complete,
        terminal_evidence_recorded=replay.terminal_evidence_recorded,
        package=package,
        cleanup_errors=[],
    )
    exec(frozen_finalizer(), scope)
    assert (dest / "status.json").read_bytes() == (s.row["output"] / "status.json").read_bytes()
    if failure:
        assert (dest / "policy-action-failure.json").read_bytes() == failure_file.read_bytes()


def test_diagnostic_exit_zero_is_insufficient(tmp_path):
    result = policy_entry.collect_diagnostic(tmp_path, "task1-2", 1, 0)
    assert not result["evidence_complete"]
    assert (
        result["diagnostic_only"] and not result["formal_eligible"] and not result["pilot_approved"]
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "field", ["joint_encoder_targets", "eef_encoder_targets", "gripper_width_m"]
)
def test_nonfinite_executed_targets_never_reach_solver_or_enqueue(field, value):
    obs = dict(joint_encoder=np.zeros((2, 7)).tolist(), gripper_width_m=[0.05, 0.05])
    response = parity.prediction(obs)
    if field == "eef_encoder_targets":
        response.update(
            mode="eef_delta", eef_encoder_targets=np.tile(np.eye(4), (32, 2, 1, 1)).tolist()
        )
    data = np.asarray(response[field])
    data.flat[0] = value
    response[field] = data.tolist()
    with pytest.raises(
        control.PolicyActionError,
        match={
            "joint_encoder_targets": "Malformed joint targets",
            "eef_encoder_targets": "Malformed EEF targets",
            "gripper_width_m": "Invalid gripper aperture",
        }[field],
    ):
        control.encoder_endpoints(
            response,
            obs,
            solver=NS(solve=lambda *a, **k: pytest.fail("nonfinite reached native service")),
        )
