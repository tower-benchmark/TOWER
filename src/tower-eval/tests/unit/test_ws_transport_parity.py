"""Real WebSocket runtime against decoded-data evidence from the base revision."""

import asyncio
import copy
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace as NS

import numpy as np
import pytest
from tower_eval.policies.runtime_ws import (
    PolicyOutputError,
    PolicyTransportError,
    RuntimePolicyAdapter,
    map_error,
)
from tower_sim.bench import control, policy
from tower_sim.bench.evaluation import EvaluationContext
from tower_sim.bench.metrics import Metrics, TrialRequestID
from tower_sim.bench.resources import save

from scripts.maintainer.generate_policy_golden import (
    CAMERAS,
    FIXTURE,
    IDENTITY,
    ROOT,
    observation,
    prediction,
    solver,
)


def exact(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    elif isinstance(a, float):
        assert isinstance(b, float)
        assert np.float64(a).tobytes() == np.float64(b).tobytes()
    else:
        assert type(a) is type(b) and a == b


@contextmanager
def service(monkeypatch):
    pytest.importorskip("msgpack_numpy")
    pytest.importorskip("websockets.asyncio.server")
    monkeypatch.setenv("NO_PROXY", "*")
    checkout = ROOT / "third_party/XPolicyLab"
    monkeypatch.syspath_prepend(str(checkout.parent))
    monkeypatch.syspath_prepend(str(checkout))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from XPolicyLab.policy.OpenWAM_TOWER.model import Model
    from XPolicyLab.utils.process_data import decode_obs_images

    calls = []
    model = Model.__new__(Model)
    model.reset()
    identities = copy.deepcopy(IDENTITY)

    def predict(obs):
        tick = int(obs["prompt"].split()[-1])
        index = round(obs["eef_encoder"][0, 0, 3] * 100)
        expected = (np.arange(36).reshape(3, 4, 3) + index + tick).astype(np.uint8)
        for pixels in obs["images"].values():
            assert pixels.dtype == np.uint8 and pixels.shape == expected.shape
            assert pixels.tobytes() == expected.tobytes()
        result = prediction(obs)
        result["identity"] = copy.deepcopy(identities)
        return result

    model.policy = NS(predict=predict, identity=identities)
    for name in (
        "get_identity",
        "prepare_case",
        "reset",
        "update_obs",
        "get_action",
        "get_prediction",
    ):
        original = getattr(model, name)

        def wrapped(*args, name=name, original=original):
            calls.append(name)
            if name == "update_obs":
                assert decode_obs_images(args[0]) is args[0]
                for name in CAMERAS:
                    assert args[0]["vision"][name]["color"].dtype == np.uint8
            return original(*args)

        setattr(model, name, wrapped)
    model.on_trial_end = lambda metadata: calls.append("trial_end")
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    server = PolicyServer(model, PolicyServerConfig(host="localhost", port=0))
    asyncio.run_coroutine_threadsafe(server.start(), loop).result(10)
    try:
        yield server, model, calls
    finally:
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
        loop.close()


def test_base_decoded_data_equivalence(monkeypatch, tmp_path):
    golden = json.loads(FIXTURE.read_text())
    events = []
    with service(monkeypatch) as (server, model, calls):
        worker = policy.OrdinaryPolicy(
            package=IDENTITY,
            url=server.url,
            prompt="synthetic 0",
            task="synthetic",
            metrics=Metrics(
                NS(emit=events.append, flush=lambda: None), TrialRequestID("synthetic")
            ),
            solver=solver(),
        )
        exact(worker.authorize(), golden["health"])
        health = worker.start(tmp_path)
        exact(json.loads((tmp_path / "policy-health.json").read_text()), golden["health"])
        rows = [{"candidate": {"seed": i}, "output": tmp_path / str(i)} for i in range(3)]
        for row in rows:
            row["output"].mkdir()
        worker.initialize(rows, CAMERAS, health)
        context = EvaluationContext.__new__(EvaluationContext)
        context.health = health
        try:
            for tick in (0, 640):
                worker.prompt = f"synthetic {tick}"
                obs = {i: observation(i, tick) for i in range(3)}
                pending = worker.prefetch(
                    obs, [NS(latest=obs[i]["images"]) for i in range(3)], tick
                )
                for i in range(3):
                    record = next(
                        r for r in golden["records"] if r["tick"] == tick and r["env_idx"] == i
                    )
                    if i == 1:
                        with pytest.raises(control.PolicyActionError, match="Frozen aperture"):
                            worker.prediction(pending, i, obs[i], tick)
                        exact(
                            json.loads(
                                (rows[i]["output"] / "policy-input-rejection.json").read_text()
                            ),
                            record["rejection"],
                        )
                    else:
                        response = worker.prediction(pending, i, obs[i], tick)
                        exact(response, record["response"])
                        context.verify_prediction(response["identity"])
                        exact(response["identity"], record["verified_identity"])
                        commands, diagnostics = control.encoder_endpoints(
                            response, obs[i], solver=solver()
                        )
                        expected = np.asarray(record["commands"], dtype=np.float64)
                        assert commands.dtype == expected.dtype and commands.shape == expected.shape
                        assert commands.tobytes() == expected.tobytes()
                        exact(diagnostics, record["diagnostics"])
                        save(
                            tmp_path / "evidence.json",
                            dict(
                                observation_time_s=tick / 400, observation=obs[i], response=response
                            ),
                        )
                        exact(
                            json.loads((tmp_path / "evidence.json").read_text()), record["evidence"]
                        )
        finally:
            worker.close()
        assert calls == ["get_identity", "get_identity", "prepare_case", "reset"] + [
            "update_obs",
            "get_action",
            "get_prediction",
            "update_obs",
            "get_action",
            "update_obs",
            "get_action",
            "get_prediction",
        ] * 2 + ["trial_end"]
    assert [e["event"] for e in events if e["event"].startswith("rpc_")] == [
        "rpc_start",
        "rpc_end",
    ] * 6
    assert worker.metrics.stats["policy.rpc_elapsed_sum"][1] == 6


@pytest.mark.parametrize(
    "category,message,action",
    [
        ("PolicyOutputError", "bad output", True),
        ("ValueError", "Negative gripper aperture", True),
        ("ValueError", "other input", False),
        ("RuntimeError", "unknown failure", False),
    ],
)
def test_real_error_mapping(monkeypatch, category, message, action):
    with service(monkeypatch) as (server, model, calls):
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        adapter.health()
        exception = type(category, (Exception,), {})

        def fail(obs):
            raise exception(message)

        model.policy.predict = fail
        try:
            with pytest.raises(PolicyOutputError if action else PolicyTransportError) as raised:
                adapter.predict(observation(0, 0))
            assert isinstance(raised.value, PolicyOutputError) == action
            if action:
                assert raised.value.body["error"] == message
        finally:
            adapter.close()


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("timeout"),
        ConnectionError("closed"),
        ValueError("Negative gripper aperture"),
        RuntimeError("unknown"),
    ],
)
def test_local_errors_are_infrastructure(exc):
    assert type(map_error(exc)) is PolicyTransportError


def test_codec_dtype_and_shape(monkeypatch):
    with service(monkeypatch):
        from client_server.ws.protocol.codec import decode_frame, encode_frame

        for dtype in (np.float64, np.float32, np.uint8, np.int64):
            value = np.arange(12, dtype=dtype).reshape(2, 2, 3)
            if dtype == np.float64:
                value.flat[0] = -0.0
                value.flat[1] = np.nextafter(1.0, 2.0)
            result = decode_frame(encode_frame({"value": value}))["value"]
            assert value.dtype == result.dtype and value.shape == result.shape
            assert value.tobytes() == result.tobytes()


def test_identity_drift_and_digest(monkeypatch):
    with service(monkeypatch) as (server, model, calls):
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        adapter.health()
        model.policy.identity["extra"] = True
        adapter.health()
        try:
            with pytest.raises(ValueError, match="full identity"):
                adapter.predict(observation(0, 0))
            model.policy.identity["package_digest"] = "different"
            with pytest.raises(ValueError, match="authorized package"):
                adapter.health()
        finally:
            adapter.close()


def test_mme_rejected_before_connect():
    worker = policy.from_package(
        package=dict(IDENTITY, kind="MME-VLA"),
        url="ws://localhost:19000",
        prompt="synthetic",
        task="synthetic",
        metrics=None,
        solver=None,
    )
    with pytest.raises(ValueError, match="MME-VLA needs an XPolicyLab WebSocket adapter"):
        worker.authorize()
    assert worker.adapter.bridge is None
    worker.close()


@pytest.mark.parametrize("field", ["gripper_width_m", "joint_encoder_targets", "native_actions"])
def test_full_prediction_prefix_must_match_bitwise(monkeypatch, field):
    with service(monkeypatch) as (server, model, calls):
        original = model.get_prediction

        def changed():
            value = original()
            array = np.asarray(value[field], dtype=np.float64)
            array.flat[0] = np.nextafter(array.flat[0], 1.0)
            value[field] = array.tolist()
            return value

        model.get_prediction = changed
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        try:
            with pytest.raises(PolicyTransportError, match="Endpoint rows differ"):
                adapter.predict(observation(0, 0))
        finally:
            adapter.close()


def test_optional_full_prediction_is_not_invented(monkeypatch):
    with service(monkeypatch) as (server, model, calls):
        model.get_prediction = None
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        try:
            response = adapter.predict(observation(0, 0))
            assert response["missing_evidence"] == [
                "full_prediction",
                "unexecuted_horizon",
                "inference_seconds",
            ]
            assert len(response["joint_encoder_targets"]) == 16
            assert "inference_seconds" not in response
        finally:
            adapter.close()


@pytest.mark.parametrize(
    "url", ["", "localhost:19000", "https://localhost", "http://localhost", "ws:///missing"]
)
def test_runtime_url_requires_websocket(url):
    with pytest.raises(ValueError, match="ws:// or wss://"):
        RuntimePolicyAdapter(url, expected_identity=IDENTITY)


def test_request_anchor_checked_before_control(monkeypatch):
    with service(monkeypatch) as (server, model, calls):
        original = model.get_action

        def changed():
            rows = original()
            rows[-1]["chunk_anchor_encoder"][0, 0, 3] = np.nextafter(0.0, 1.0)
            return rows

        model.get_action = changed
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        try:
            with pytest.raises(PolicyTransportError, match="anchor differs"):
                adapter.predict(observation(0, 640))
        finally:
            adapter.close()


@pytest.mark.parametrize("identity", [{}, {"package_digest": ""}, {"package_digest": None}])
def test_package_digest_required(identity):
    with pytest.raises(ValueError, match="package_digest"):
        RuntimePolicyAdapter("ws://localhost:19000", expected_identity=identity)


def test_missing_diagnostic_fields_are_explicit():
    from tower_eval.policies.runtime_ws import reconstruct

    row = dict(
        left_arm_joint_state=np.zeros(7),
        right_arm_joint_state=np.zeros(7),
        left_ee_joint_state=[0.05],
        right_ee_joint_state=[0.05],
        identity=IDENTITY,
    )
    result = reconstruct([row], "joint_absolute", None, None)
    for field in ("native_actions", "applied_j8", "units"):
        assert field not in result and field in result["missing_evidence"]


def test_invalid_png_is_infrastructure(monkeypatch):
    with service(monkeypatch) as (server, model, calls):
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        obs = observation(0, 0)
        obs["images"][CAMERAS[0]] = "not a png"
        try:
            with pytest.raises(PolicyTransportError):
                adapter.predict(obs)
            assert calls == ["get_identity"]
        finally:
            adapter.close()
