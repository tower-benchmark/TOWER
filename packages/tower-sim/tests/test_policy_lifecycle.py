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
