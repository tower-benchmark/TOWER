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
from tower_sim.bench.metrics import Metrics, TrialRequestID

from scripts.maintainer.generate_policy_golden import (
    CAMERAS,
    FIXTURE,
    IDENTITY,
    ROOT,
    capture,
    images,
    observation,
    payload,
    pixels,
    prediction,
    solver,
    variant,
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
        for rgb in obs["images"].values():
            assert rgb.dtype == np.uint8 and rgb.shape == expected.shape
            assert rgb.tobytes() == expected.tobytes()
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
    server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
    asyncio.run_coroutine_threadsafe(server.start(), loop).result(10)
    try:
        yield server, model, calls
    finally:
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
        loop.close()


@pytest.mark.parametrize(
    "case", ["valid", "aperture", "nonrigid", "anchor", "shape", "world", "nan"]
)
def test_base_decoded_data_equivalence(monkeypatch, tmp_path, case):
    golden = json.loads(FIXTURE.read_text())
    events = []
    with service(monkeypatch) as (server, model, calls):
        if case != "valid":
            model.policy.predict = lambda obs: variant(obs, case)
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
        records = []
        try:
            for tick in (0, 640) if case == "valid" else (640,):
                worker.prompt = f"synthetic {tick}"
                obs = {i: observation(i, tick) for i in reversed(range(3))}
                # Runtime observations deliberately contain exactly the three encoder fields.
                assert set(obs[0]) == {"joint_encoder", "eef_encoder", "gripper_width_m"}
                records.extend(
                    capture(
                        worker,
                        dict(sorted(obs.items())),
                        [NS(latest=images(i, tick)) for i in range(3)],
                        tick,
                        tmp_path,
                    )
                )
                for i in range(3):
                    assert pixels(i, tick).tolist() == golden["pixels"][str(tick)][i]
        finally:
            worker.close()
        exact(records, golden["cases"][case])
        assert calls[-1] == "trial_end"
    count = 6 if case == "valid" else 3
    assert [e["event"] for e in events if e["event"].startswith("rpc_")] == [
        "rpc_start",
        "rpc_end",
    ] * count
    assert all(e["transport"] == "websocket" for e in events if e["event"] == "rpc_end")
    assert worker.metrics.stats["policy.rpc_elapsed_sum"][1] == count


@pytest.mark.parametrize(
    "category,message,action",
    [
        ("PolicyOutputError", "bad output", True),
        ("ValueError", "Negative gripper aperture", True),
        ("ValueError", "other input", False),
        ("RuntimeError", "unknown failure", False),
        ("RuntimeError", "Negative gripper aperture", False),
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
                adapter.predict(payload(0, 0))
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
            with pytest.raises(PolicyTransportError, match="full identity"):
                adapter.predict(payload(0, 0))
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

        def changed(index):
            value = original(index)
            array = np.asarray(value[field], dtype=np.float64)
            array.flat[0] = np.nextafter(array.flat[0], 1.0)
            value[field] = array.tolist()
            return value

        model.get_prediction = changed
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        try:
            with pytest.raises(PolicyTransportError, match="different request"):
                adapter.predict(payload(0, 0))
        finally:
            adapter.close()


def test_optional_full_prediction_is_not_invented(monkeypatch):
    with service(monkeypatch) as (server, model, calls):
        model.get_prediction = None
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        try:
            response = adapter.predict(payload(0, 0))
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
    result = reconstruct([row], "joint_absolute", None)
    for field in ("native_actions", "applied_j8", "units"):
        assert field not in result and field in result["missing_evidence"]


def test_invalid_png_is_infrastructure(monkeypatch):
    with service(monkeypatch) as (server, model, calls):
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        obs = payload(0, 0)
        obs["images"][CAMERAS[0]] = "not a png"
        try:
            with pytest.raises(PolicyTransportError):
                adapter.predict(obs)
            assert calls == ["get_identity"]
        finally:
            adapter.close()


def test_array_prediction_and_nonfinite_tail(monkeypatch):
    with service(monkeypatch) as (server, model, calls):
        original = model.policy.predict

        def arrays(obs):
            result = original(obs)
            result["gripper_width_m"][-1][0] = float("nan")
            for key in ("native_actions", "gripper_width_m", "joint_encoder_targets"):
                result[key] = np.asarray(result[key], dtype=np.float64)
            return result

        model.policy.predict = arrays
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        try:
            result = adapter.predict(payload())
            expected = prediction(payload())
            expected["gripper_width_m"][-1][0] = float("nan")
            for key in ("native_actions", "gripper_width_m", "joint_encoder_targets"):
                assert isinstance(result[key], list)
                assert (
                    np.asarray(result[key]).tobytes()
                    == np.asarray(expected[key], dtype=np.float64).tobytes()
                )
        finally:
            adapter.close()


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_non_float64_predictions_rejected(monkeypatch, dtype):
    with service(monkeypatch) as (server, model, calls):
        original = model.get_prediction

        def narrowed(index):
            result = original(index)
            result["native_actions"] = np.asarray(result["native_actions"], dtype=dtype)
            return result

        model.get_prediction = narrowed
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        try:
            with pytest.raises(PolicyTransportError, match="float64 dtype"):
                adapter.predict(payload())
        finally:
            adapter.close()


def test_nested_identity_and_numeric_equality(monkeypatch):
    with service(monkeypatch) as (server, model, calls):
        model.policy.identity["nested"] = {"flags": [1, {"enabled": True}]}
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        snapshot = adapter.health()
        snapshot["identity"]["nested"]["flags"][0] = 99
        model.policy.identity["nested"]["flags"] = [True, {"enabled": 1.0}]
        try:
            assert adapter.predict(payload())["identity"] == model.policy.identity
            model.policy.identity["nested"]["flags"][1]["enabled"] = False
            with pytest.raises(PolicyTransportError, match="full identity"):
                adapter.predict(payload())
        finally:
            adapter.close()


@pytest.mark.parametrize(
    "code,category,message,action",
    [
        ("call_failed", "ValueError", "Negative gripper aperture extra", False),
        ("invalid_frame", "PolicyOutputError", "bad", False),
        ("internal", "ValueError", "Negative gripper aperture", False),
        ("infer_failed", "ValueError", "Negative gripper aperture", True),
        ("infer_failed", "PolicyOutputError", "bad", True),
    ],
)
def test_error_code_and_exact_aperture(code, category, message, action):
    exc = RuntimeError(message)
    exc.code = NS(value=code)
    exc.details = {"exception_type": category}
    result = map_error(exc, "synthetic-request")
    assert isinstance(result, PolicyOutputError) == action
    if not action:
        assert result.body == dict(
            code=code, message=message, exception_type=category, request_id="synthetic-request"
        )


def test_identity_output_error_is_infrastructure(monkeypatch):
    with service(monkeypatch) as (server, model, calls):

        def fail():
            raise type("PolicyOutputError", (ValueError,), {})("identity unavailable")

        model.get_identity = fail
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        try:
            with pytest.raises(PolicyTransportError) as error:
                adapter.health()
            assert error.value.body["exception_type"] == "PolicyOutputError"
        finally:
            adapter.close()


@pytest.mark.parametrize("bad", [None, [1], [{}]])
def test_malformed_rows_are_transport(monkeypatch, bad):
    with service(monkeypatch) as (server, model, calls):
        original = model.get_action

        def malformed():
            original()
            return bad

        model.get_action = malformed
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        try:
            with pytest.raises(PolicyTransportError) as error:
                adapter.predict(payload(), headers={"X-Baseline-Request-ID": "synthetic-request"})
            assert error.value.body["request_id"] == "synthetic-request"
        finally:
            adapter.close()


def test_order_rejection_before_transport_and_no_remaining_calls(monkeypatch, tmp_path):
    with service(monkeypatch) as (server, model, calls):
        worker = policy.OrdinaryPolicy(
            package=IDENTITY,
            url=server.url,
            prompt="synthetic 0",
            task="synthetic",
            metrics=Metrics(
                NS(emit=lambda r: None, flush=lambda: None), TrialRequestID("synthetic")
            ),
            solver=solver(),
        )
        health = worker.authorize()
        worker.start(tmp_path)
        rows = [{"candidate": {"seed": i}, "output": tmp_path / str(i)} for i in range(3)]
        for row in rows:
            row["output"].mkdir()
        worker.initialize(rows, CAMERAS, health)
        seen = []

        def predict(obs):
            i = round(obs["eef_encoder"][0, 0, 3] * 100)
            seen.append(i)
            if i == 0:
                raise ValueError("Negative gripper aperture")
            raise RuntimeError("synthetic infrastructure failure")

        model.policy.predict = predict
        observations = {i: observation(i, 0) for i in (2, 1, 0)}
        try:
            pending = worker.prefetch(observations, [NS(latest=images(i, 0)) for i in range(3)], 0)
            with pytest.raises(control.PolicyActionError):
                worker.prediction(pending, 0, observations[0], 0)
            with pytest.raises(RuntimeError):
                worker.prediction(pending, 1, observations[1], 0)
            assert seen == [0, 1] and 2 not in pending
            assert (rows[0]["output"] / "policy-input-rejection.json").is_file()
            error = json.loads((rows[1]["output"] / "policy-error.json").read_text())
            assert (
                error["exception_type"] == "RuntimeError" and error["request_id"] == "synthetic:1:0"
            )
        finally:
            worker.close()


def test_two_clients_atomic_infer_and_mismatched_audit(monkeypatch):
    import base64
    import io
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image
    from tower_eval.policies.xpolicylab import observation_to_xpolicylab

    with service(monkeypatch) as (server, model, calls):
        adapters = [RuntimePolicyAdapter(server.url, expected_identity=IDENTITY) for _ in range(2)]
        observations = [payload(0, 640), payload(10, 640)]
        for obs in observations:
            obs["env_idx"] = 0
        for adapter in adapters:
            adapter.health()

        def infer(pair):
            adapter, obs = pair
            mapped = dict(
                obs,
                images={
                    k: np.asarray(Image.open(io.BytesIO(base64.b64decode(v))))
                    for k, v in obs["images"].items()
                },
            )
            client = adapter.bridge._client
            return client._run(client._client.infer(observation_to_xpolicylab(mapped))).payload[
                "actions"
            ]

        try:
            with ThreadPoolExecutor(2) as pool:
                results = list(pool.map(infer, zip(adapters, observations)))
            for rows, obs in zip(results, observations):
                assert rows[0]["left_ee_transform"][0, 3] == obs["eef_encoder"][0][0][3]
            original = adapters[0].bridge._call

            def interleave(name, index=None):
                if name == "get_prediction":
                    adapters[1].predict(observations[1])
                return original(name, index)

            adapters[0].bridge._call = interleave
            with pytest.raises(
                PolicyTransportError, match="different request; is it shared by another client"
            ):
                adapters[0].predict(observations[0])
        finally:
            for adapter in adapters:
                adapter.close()


@pytest.mark.parametrize("category", ["PolicyOutputError", "ValueError", "KeyError"])
def test_audit_errors_are_transport(monkeypatch, category):
    with service(monkeypatch) as (server, model, calls):

        def fail(index):
            raise type(category, (Exception,), {})("Negative gripper aperture")

        model.get_prediction = fail
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        try:
            with pytest.raises(PolicyTransportError) as caught:
                adapter.predict(payload())
            if category == "KeyError":
                assert "missing prediction for env 0 (shared server or policy bug)" in str(
                    caught.value
                )
            else:
                assert caught.value.body["exception_type"] == category
        finally:
            adapter.close()


@pytest.mark.parametrize("mode", ["joint_absolute", "eef_delta"])
@pytest.mark.parametrize(
    "field", ["mode", "coordinate_frame", "control_hz", "execute_steps", "j8_actuator_mask"]
)
def test_explicit_eef_and_standard_joint_contract(monkeypatch, mode, field):
    from tower_eval.policies.runtime_ws import reconstruct

    with service(monkeypatch) as (server, model, calls):
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        adapter.health()
        try:
            adapter.predict(payload(0, 640 if mode == "eef_delta" else 0))
            rows = model.get_action()
            for row in rows:
                del row[field]
            if mode == "eef_delta":
                with pytest.raises(ValueError, match="explicit"):
                    reconstruct(rows, "joint_absolute", None)
            else:
                assert reconstruct(rows, mode, None)[field] == prediction(payload())[field]
        finally:
            adapter.close()


@pytest.mark.parametrize("field", ["identity", "units"])
def test_each_row_metadata_is_checked(monkeypatch, field):
    with service(monkeypatch) as (server, model, calls):
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        adapter.health()
        original = adapter.bridge.infer

        def corrupt(obs):
            rows = original(obs)
            rows[1][field] = {"changed": True}
            return rows

        adapter.bridge.infer = corrupt
        if field == "identity":
            model.get_prediction = None
        try:
            with pytest.raises(PolicyTransportError):
                adapter.predict(payload())
        finally:
            adapter.close()


def test_png_format_not_just_decodable_image(monkeypatch):
    import base64
    import io

    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(pixels(0, 0)).save(stream, format="BMP")
    with service(monkeypatch) as (server, model, calls):
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        obs = payload()
        obs["images"][CAMERAS[0]] = base64.b64encode(stream.getvalue()).decode()
        try:
            with pytest.raises(PolicyTransportError, match="must be PNG"):
                adapter.predict(obs)
        finally:
            adapter.close()


def test_timeouts_identity_normalization_and_reconnect_stdout(monkeypatch, capsys):
    with service(monkeypatch) as (server, model, calls):
        model.policy.identity["array"] = np.array([1.0, 2.0], dtype=np.float64)
        adapter = RuntimePolicyAdapter(server.url, expected_identity=IDENTITY)
        try:
            assert adapter.health()["identity"]["array"] == [1.0, 2.0]
            client = adapter.bridge._client
            assert client._client.config.request_timeout_s == 300.0
            original = client.call

            def check(name, payload=None):
                if name == "get_identity":
                    assert client._client.config.request_timeout_s == 30.0
                return original(name, payload)

            monkeypatch.setattr(client, "call", check)
            adapter.health()
            capsys.readouterr()
            client._run(client._client._reset_connection_state())
            adapter.predict(payload())
            captured = capsys.readouterr()
            assert captured.out == ""
            assert "CONNECT" in captured.err
        finally:
            adapter.close()


def test_atomic_infer_prevents_interleaved_update(monkeypatch):
    from tower_eval.policies.xpolicylab import observation_to_xpolicylab

    with service(monkeypatch) as (server, model, calls):
        adapters = [RuntimePolicyAdapter(server.url, expected_identity=IDENTITY) for _ in range(2)]
        for adapter in adapters:
            adapter.health()
        other = dict(
            payload(10, 640), images={name: pixels(10, 640) for name in CAMERAS}, env_idx=0
        )
        original = adapters[0].bridge._call
        interleaved = []

        def hook(name, value=None):
            result = original(name, value)
            if name == "update_obs":
                adapters[1].bridge._call("update_obs", observation_to_xpolicylab(other))
                interleaved.append(True)
            return result

        adapters[0].bridge._call = hook
        try:
            result = adapters[0].predict(payload(0, 640))
            assert result["eef_encoder_targets"][0][0][0][3] == 0.0
            assert not interleaved
        finally:
            for adapter in adapters:
                adapter.close()


def test_adapter_infer_wire_frames_and_steps(monkeypatch):
    from tower_eval.policies.xpolicylab import XPolicyLabAdapter, observation_to_xpolicylab

    with service(monkeypatch) as (server, model, calls):
        received = []
        dispatch = server._dispatch_frame

        async def record(frame):
            if frame.message_type.name in {"INFER", "CALL"}:
                received.append((frame.message_type.name, frame.step))
            return await dispatch(frame)

        monkeypatch.setattr(server, "_dispatch_frame", record)
        adapter = XPolicyLabAdapter(server.url)
        observation = dict(payload(), images={name: pixels(0, 0) for name in CAMERAS})
        try:
            for step in range(2):
                rows = adapter.infer(observation_to_xpolicylab(observation))
                assert len(rows) == IDENTITY["execute_steps"]
                # Observe decoded wire frames at the server, below either client wrapper.
                assert received == [("INFER", index) for index in range(step + 1)]
        finally:
            adapter.close()
