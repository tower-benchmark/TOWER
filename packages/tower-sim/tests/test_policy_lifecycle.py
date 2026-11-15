"""Policy lifecycle controller failure contracts; decoded parity lives in test_ws_transport_parity."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_replay_lifecycle import simulation as simulation
from tower_sim.bench import control, policy_entry

from scripts.maintainer.generate_policy_golden import prediction


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
    response = prediction(dict(obs, prompt="synthetic 0"))
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


@pytest.mark.parametrize("case", ["action", "output", "aperture", "transport", "identity"])
def test_shared_loop_policy_outcomes(simulation, case, tmp_path):
    import json

    from tower_eval.policies.runtime_ws import PolicyOutputError, PolicyTransportError
    from tower_sim.bench import policy
    from tower_sim.bench.metrics import Metrics, TrialRequestID

    from scripts.maintainer.generate_policy_golden import IDENTITY

    calls = []

    def predict(obs, *, headers, rpc_observer):
        rpc_observer("rpc_start")
        try:
            if case in ("output", "aperture"):
                message = (
                    "Negative gripper aperture"
                    if case == "aperture"
                    else "synthetic invalid output"
                )
                code = "invalid_observation" if case == "aperture" else "invalid_policy_output"
                raise PolicyOutputError(message, body={"code": code, "error": message})
            if case == "transport":
                raise PolicyTransportError("synthetic failure")
            if case == "identity":
                raise ValueError("full identity differs")
            return prediction(dict(obs, prompt="synthetic 0"))
        finally:
            rpc_observer("rpc_end", transport="websocket")

    worker = policy.OrdinaryPolicy(
        package=dict(IDENTITY, checkpoint_sha256="synthetic-checkpoint"),
        url="ws://localhost:19000",
        prompt="synthetic 0",
        task="synthetic",
        metrics=Metrics(NS(emit=lambda row: None, flush=lambda: None), TrialRequestID("synthetic")),
        solver=NS(identity={}),
        adapter=NS(
            health=lambda: dict(ready=True, identity=IDENTITY),
            predict=predict,
            reset=lambda **kw: calls.append("reset"),
            close=lambda: calls.append("close"),
        ),
    )
    worker.authorize()
    if case in ("transport", "identity"):
        with pytest.raises((RuntimeError, ValueError)):
            simulation.run(policy=worker)
    else:
        assert simulation.run(policy=worker) == 0
    status = json.loads((simulation.row["output"] / "status.json").read_text())
    assert status["evidence_complete"] == (case not in ("transport", "identity"))
    assert status["policy_action_failure"] == (case in ("output", "aperture"))
    assert calls == ["reset", "close"]


@pytest.mark.parametrize(
    "case", ["valid", "aperture", "nonrigid", "anchor", "shape", "nan", "ikreject"]
)
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_real_ws_shared_loop_status_bytes(simulation, monkeypatch, tmp_path, case, cleanup_failure):
    import json
    import runpy

    from scipy.spatial.transform import Rotation
    from tower_sim.bench import policy
    from tower_sim.bench.metrics import Metrics, TrialRequestID
    from tower_sim.bench.observation import encoder_observation

    from scripts.maintainer.generate_policy_golden import (
        CAMERAS,
        FIXTURE,
        IDENTITY,
        ROOT,
        solver,
        variant,
    )

    service = runpy.run_path(str(ROOT / "src/tower-eval/tests/unit/test_ws_transport_parity.py"))[
        "service"
    ]
    with service(monkeypatch) as (server, model, calls):
        model.policy.predict = lambda obs: variant(obs, case)
        if cleanup_failure:

            def fail(metadata):
                raise RuntimeError("synthetic trial_end failure")

            model.on_trial_end = fail
        simulation.camera.names = {name: str(i) for i, name in enumerate(CAMERAS)}
        simulation.factories.observation = lambda qpos, *args: encoder_observation(
            qpos,
            NS(encoder_from_physical=lambda q, *a: q),
            Rotation.identity(),
            np.zeros(3),
            fk=lambda *args: {"gripper_base": np.eye(4)},
        )
        worker = policy.OrdinaryPolicy(
            package=dict(IDENTITY, checkpoint_sha256="synthetic-checkpoint"),
            url=server.url,
            prompt="synthetic 640",
            task="synthetic",
            metrics=Metrics(
                NS(emit=lambda r: None, flush=lambda: None), TrialRequestID("synthetic")
            ),
            solver=NS(identity={}, solve=solver(case == "ikreject").solve),
        )
        worker.authorize()
        if case in {"nan", "shape"}:
            with pytest.raises(ValueError):
                simulation.run(policy=worker)
        else:
            assert simulation.run(policy=worker) == 0
        actual = (simulation.row["output"] / "status.json").read_text()
        golden = json.loads(FIXTURE.read_text())
        assert actual == golden["statuses"][case]
        output = simulation.row["output"]
        files = {
            p.relative_to(output).as_posix(): p.read_text()
            for p in output.rglob("*.json")
            if p.parent.name == "chunks"
            or p.name
            in {
                "status.json",
                "policy-action-failure.json",
                "policy-input-rejection.json",
                "policy-error.json",
            }
        }
        assert files == {
            name: json.dumps(value, indent=2, allow_nan=False) + "\n"
            for name, value in golden["loop_files"][case].items()
        }
        if cleanup_failure:
            errors = json.loads((tmp_path / "policy-cleanup-errors.json").read_text())
            assert errors[0]["message"] == "synthetic trial_end failure"


def test_worker_status_records_sequential_policy(tmp_path):
    import ast
    from pathlib import Path

    from tower_sim.bench import replay
    from tower_sim.bench.resources import save

    tree = ast.parse(Path(replay.__file__).read_text())
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "save"
        and "worker-status.json" in ast.unparse(n)
    )
    eval(
        compile(ast.Expression(call), "worker-status", "eval"),
        dict(
            save=save,
            context=NS(output=tmp_path),
            policy=NS(stage="synthetic"),
            tick=400,
            ended=[False] * 3,
            env=NS(n=3),
        ),
    )
    import json

    result = json.loads((tmp_path / "worker-status.json").read_text())
    assert result["serialized_neural_inference"] is True
    assert result["concurrent_request_limit"] == 1


def test_nonjson_error_body_is_not_written(tmp_path):
    from concurrent.futures import Future

    from tower_eval.policies.runtime_ws import PolicyOutputError
    from tower_sim.bench import policy
    from tower_sim.bench.metrics import Metrics, TrialRequestID

    worker = policy.OrdinaryPolicy(
        package={"package_digest": "synthetic"},
        url="ws://localhost",
        prompt="synthetic",
        task="synthetic",
        solver=None,
        metrics=Metrics(NS(emit=lambda r: None, flush=lambda: None), TrialRequestID("synthetic")),
    )
    worker.rows = [{"output": tmp_path}]
    pending = Future()
    pending.set_exception(
        PolicyOutputError("synthetic", body={"message": "not structured"}, body_is_json=False)
    )
    with pytest.raises(control.PolicyActionError):
        worker.prediction({0: pending}, 0, {}, 0)
    assert not (tmp_path / "policy-error.json").exists()
