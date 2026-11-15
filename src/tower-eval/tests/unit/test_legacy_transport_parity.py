# SPDX-License-Identifier: Apache-2.0
"""Execute frozen transport statements, never a handwritten copy of the client.

Only simulation/recording inputs are stubbed. AST selection keeps source locations
and fails if a block is missing/ambiguous. The frozen files remain byte-identical.
"""

from __future__ import annotations

import ast
import base64
import copy
import io
import json
import runpy
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tower_eval.config import load_config
from tower_eval.policies.legacy_predict import (
    LegacyPredictAdapter,
    PolicyOutputError,
    PolicyTransportError,
)
from tower_eval.policies.xpolicylab import CAMERAS

from scripts.maintainer.archive_root import ArchivedToolPath

ROOT = Path(__file__).resolve().parents[4]
FROZEN = ArchivedToolPath()
IDENTITY = {"package_digest": "transport-fixture", "seed": 42, "position_frames": 256}
TRIAL = "transport-parity"
SEED = 17
# Empty deliberately: both clients use urllib, the same process, and one server.
# Even Host, User-Agent, Content-Length, Connection and header order must match.
HEADER_ALLOWLIST = frozenset()


def one(nodes, predicate):
    found = [node for node in nodes if predicate(node)]
    assert len(found) == 1, f"Expected one frozen statement, found {len(found)}"
    return found[0]


def tree(name):
    return ast.parse((FROZEN / name).read_text(), filename=str(FROZEN / name))


def statements(name, nodes):
    return compile(ast.Module(body=nodes, type_ignores=[]), name, "exec")


def assignment(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


class FrozenClient:
    def __init__(self, name, metrics, control, url):
        execute = one(
            tree(name).body, lambda n: isinstance(n, ast.FunctionDef) and n.name == "execute"
        )
        nodes = list(ast.walk(execute))
        self.request_code = statements(name, [one(nodes, lambda n: assignment(n, "request"))])
        health = one(
            nodes,
            lambda n: isinstance(n, ast.With) and "'/health'" in ast.unparse(n.items[0]),
        )
        identity = one(execute.body, lambda n: assignment(n, "identity"))
        ready = one(
            execute.body,
            lambda n: isinstance(n, ast.If) and ast.unparse(n.test).startswith("not health.get"),
        )
        self.health_code = statements(name, [health, identity, ready])
        response = one(
            nodes,
            lambda n: (
                isinstance(n, ast.Try)
                and any(ast.unparse(h.type) == "urllib.error.HTTPError" for h in n.handlers)
            ),
        )
        owner = one(nodes, lambda n: isinstance(n, ast.Try) and response in n.body)
        endpoint = one(
            owner.body,
            lambda n: (
                isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "(endpoints, diagnostics)"
            ),
        )
        self.response_code = statements(name, owner.body[: owner.body.index(endpoint) + 1])
        self.parsed_code = statements(name, owner.body[: owner.body.index(endpoint)])
        startup = one(
            ast.walk(tree("v4_policy_eval.py")),
            lambda n: isinstance(n, ast.With) and "'/health'" in ast.unparse(n.items[0]),
        )
        package_check = one(
            ast.walk(tree("v4_evaluation.py")),
            lambda n: (
                isinstance(n, ast.Assert)
                and ast.unparse(n.test).startswith("health['identity']['package_digest']")
            ),
        )
        verify = one(
            ast.walk(tree("v4_evaluation.py")),
            lambda n: isinstance(n, ast.FunctionDef) and n.name == "verify_prediction",
        )
        self.startup_code = statements("v4_policy_eval.py", [startup])
        self.package_code = statements("v4_evaluation.py", [package_check])
        self.scope = dict(
            urllib=urllib,
            json=json,
            np=np,
            METRICS=metrics,
            control=control,
            kin=SimpleNamespace(
                LIMITS=np.stack((control.GENERATION_LOWER, control.GENERATION_UPPER), axis=1)
            ),
            worker=SimpleNamespace(save=lambda path, value: None),
            args=SimpleNamespace(
                policy_url=url, prompt='Place the "block" — carefully', task="task1-2"
            ),
            package={"package_digest": IDENTITY["package_digest"]},
            row={"candidate": {"seed": SEED}, "output": Path("unused")},
            index=0,
            command_replay=False,
            formal_context=None,
        )
        exec(statements("v4_evaluation.py", [verify]), self.scope)

    def startup_health(self):
        exec(self.startup_code, self.scope)
        health = self.scope["health"]
        context = SimpleNamespace(
            health=health, package=self.scope["package"], protocol=self.scope["package"]
        )
        exec(self.package_code, dict(self.scope, self=context))
        context.verify_prediction = lambda identity: self.scope["verify_prediction"](
            context, identity
        )
        self.scope["formal_context"] = context
        return health

    def health(self):
        exec(self.health_code, self.scope)
        return self.scope["health"]

    def predict(self, observation, archive, history, tick, *, execute=True):
        self.scope.update(
            observation=observation, archives=[archive], histories=[history], tick=tick
        )
        exec(self.request_code, self.scope)
        self.scope["pending"] = {
            0: SimpleNamespace(
                result=lambda: self.scope["METRICS"].http_json(self.scope["request"])
            )
        }
        exec(self.response_code if execute else self.parsed_code, self.scope)
        return self.scope["predicted"], self.scope["endpoints"] if execute else None

    def endpoints(self, response, observation):
        return self.scope["control"].encoder_endpoints(response, observation, self.scope["kin"])[0]


@pytest.fixture
def frozen_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_BASELINE_METRICS", str(tmp_path / "metrics.jsonl"))
    monkeypatch.setenv("TOWER_BASELINE_TRIAL", TRIAL)
    monkeypatch.setenv("TOWER_BASELINE_TRACE", "1")
    metrics = runpy.run_path(str(FROZEN / "baseline_metrics.py"))["METRICS"]
    history = runpy.run_path(str(FROZEN / "mme_history.py"))
    monkeypatch.setitem(sys.modules, "mme_history", SimpleNamespace(**history))

    def no_ik(*args, **kwargs):
        pytest.fail("Transport parity must not launch an IK process")

    monkeypatch.setitem(sys.modules, "generation_ik_client", SimpleNamespace(solve=no_ik))
    control = SimpleNamespace(**runpy.run_path(str(FROZEN / "control.py")))
    runtime = SimpleNamespace(**runpy.run_path(str(FROZEN / "runtime_worker.py")))
    hook = SimpleNamespace(**runpy.run_path(str(FROZEN / "mme_history_hook.py")))
    yield SimpleNamespace(metrics=metrics, control=control, runtime=runtime, hook=hook)
    metrics.close()


@pytest.fixture(params=["batch_policy_worker.py", "batch_policy_worker_mme.py"])
def worker_name(request):
    return request.param


class HeaderReader:
    def __init__(self, stream):
        self.stream = stream
        self.lines = []

    def readline(self, *args):
        line = self.stream.readline(*args)
        self.lines.append(line)
        return line


@pytest.fixture
def server():
    state = SimpleNamespace(
        requests=[],
        health={"ready": True, "identity": copy.deepcopy(IDENTITY)},
        status=200,
        reply=None,
        health_status=200,
        disconnect=False,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def parse_request(self):
            reader = HeaderReader(self.rfile)
            original = self.rfile
            self.rfile = reader
            try:
                parsed = super().parse_request()
            finally:
                self.rfile = original
            self.header_bytes = b"".join(reader.lines)
            return parsed

        def do_GET(self):
            self.respond(state.health_status, state.health)

        def do_POST(self):
            self.respond(state.status, state.reply)

        def respond(self, status, body):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state.requests.append(
                (self.command, self.path, self.raw_requestline, self.header_bytes, raw)
            )
            if state.disconnect:
                self.close_connection = True
                return
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    state.url = f"http://127.0.0.1:{http.server_port}"
    try:
        yield state
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def frame(tick):
    # Distinct per tick, camera and channel, so history order/resampling is visible.
    return np.random.default_rng(tick).integers(0, 256, (4, 240, 320, 3), dtype=np.uint8)


def inputs(modules, tick):
    # Nontrivial PNG bytes; no renderer or calibration stub.
    archive = object.__new__(modules.runtime.CameraArchive)
    archive.camera = SimpleNamespace(names=CAMERAS)
    history = modules.hook.EpisodeHistory("parity-episode", CAMERAS, 256)
    for at_tick in range(0, tick + 1, 40):
        archive._latest_pixels = pixels = frame(at_tick)
        archive._latest_encoded = None
        archive.times = [at_tick / 400]
        history.append_validated(archive, at_tick, rejected=False, terminal=False)
    joints = (modules.control.GENERATION_LOWER + modules.control.GENERATION_UPPER) / 2
    observation = dict(
        joint_encoder=np.tile(joints, (2, 1)).tolist(),
        eef_encoder=np.tile(np.eye(4), (2, 1, 1)).tolist(),
        gripper_width_m=[0.02, 0.08],
    )
    return observation, archive, history, pixels


def prediction(observation):
    return dict(
        identity=copy.deepcopy(IDENTITY),
        mode="joint_absolute",
        coordinate_frame="encoder_arm_base",
        control_hz=10,
        execute_steps=16,
        j8_actuator_mask=[False, False],
        units={"joint": "rad", "translation": "m", "gripper_width": "m", "predicted_j8": "mm"},
        joint_encoder_targets=np.tile(observation["joint_encoder"], (32, 1, 1)).tolist(),
        gripper_width_m=np.tile(observation["gripper_width_m"], (32, 1)).tolist(),
    )


def payload(client, observation, archive, history, tick, pixels=None):
    result = dict(
        observation,
        images=archive.latest if pixels is None else dict(zip(CAMERAS, pixels)),
        prompt=client.scope["args"].prompt,
    )
    if "_mme" in client.scope["worker_name"]:
        result.update(
            task="task1-2",
            episode_id=history.episode_id,
            observation_time_s=tick / 400,
            history=history.request(tick),
        )
    return result


def baseline_headers(tick, trace=True):
    # Callers pass only the baseline headers; the adapter supplies Content-Type.
    return {
        "X-Baseline-Request-ID": f"{TRIAL}:{SEED}:{tick}",
        "X-Baseline-Capture": ("start" if tick == 1280 else "stop" if tick == 2560 else "")
        if trace
        else "",
    }


def clients(server, modules, worker_name):
    frozen = FrozenClient(worker_name, modules.metrics, modules.control, server.url)
    frozen.scope["worker_name"] = worker_name
    adapter = LegacyPredictAdapter(server.url, expected_identity=IDENTITY)
    return frozen, adapter


def authorize(frozen, adapter):
    assert frozen.startup_health() == adapter.health()
    assert frozen.health() == adapter.health()


def assert_wire_equal(frozen_requests, adapter_requests):
    assert not HEADER_ALLOWLIST
    assert frozen_requests == adapter_requests


@pytest.mark.archive
@pytest.mark.parametrize("tick", [0, 640, 1280, 2560])
@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("header_source", ["frozen", "baseline-only"])
def test_raw_wire_and_parsed_actions(
    server, frozen_modules, worker_name, tick, encoded, header_source
):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    observation, archive, history, pixels = inputs(frozen_modules, tick)
    server.reply = prediction(observation)
    frozen.startup_health()
    frozen.health()
    response, actions = frozen.predict(observation, archive, history, tick)
    wire = list(server.requests)
    server.requests.clear()
    adapter.health()
    adapter.health()
    obs = payload(frozen, observation, archive, history, tick, None if encoded else pixels)
    sent_headers = (
        frozen_modules.metrics.headers(SEED, tick)
        if header_source == "frozen"
        else baseline_headers(tick)
    )
    actual = adapter.predict(obs, headers=sent_headers)
    assert_wire_equal(wire, server.requests)
    assert actual == response
    np.testing.assert_array_equal(frozen.endpoints(actual, observation), actions)
    np.testing.assert_array_equal(adapter.act(obs).actions, actions)
    sent = json.loads(wire[-1][-1])
    if "_mme" in worker_name:
        assert sent["history"] == history.request(tick)
        assert sent["history"]["prefix_frames"] == tick // 40 + 1
        assert len(sent["history"]["indices"]) == min(tick // 40 + 1, 32)
        digests = sent["history"]["png_sha256"]
        assert len(set(digests)) == len(digests)
        assert digests == [history.prefix.frames[i][1] for i in sent["history"]["indices"]]
        assert sent["history"]["indices"] == sorted(sent["history"]["indices"])
    else:
        assert "history" not in sent
    for name in CAMERAS:
        assert base64.b64decode(sent["images"][name]).startswith(b"\x89PNG\r\n\x1a\n")
    history.close()


@pytest.mark.archive
@pytest.mark.parametrize(
    "health_identity",
    [
        {"package_digest": IDENTITY["package_digest"]},
        dict(IDENTITY, seed=99, extra={"versions": [1, 2]}),
    ],
)
def test_health_pins_server_identity_not_caller_shape(
    server, frozen_modules, worker_name, health_identity
):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    server.health["identity"] = health_identity
    authorize(frozen, adapter)
    observation, archive, history, _ = inputs(frozen_modules, 0)
    server.reply = dict(prediction(observation), identity=health_identity)
    expected, actions = frozen.predict(observation, archive, history, 0)
    actual = adapter.predict(
        payload(frozen, observation, archive, history, 0), headers=baseline_headers(0)
    )
    assert actual == expected
    np.testing.assert_array_equal(frozen.endpoints(actual, observation), actions)
    # A caller mutating the returned health object cannot change the pinned identity.
    adapter.health()["identity"]["extra"] = "mutated"
    assert adapter.predict(payload(frozen, observation, archive, history, 0)) == expected


@pytest.mark.archive
@pytest.mark.parametrize("change", ["package", "not_ready"])
def test_health_rejection(server, frozen_modules, worker_name, change):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    if change == "package":
        server.health["identity"]["package_digest"] = "different"
        with pytest.raises(AssertionError):
            frozen.startup_health()
    else:
        server.health["ready"] = False
    with pytest.raises(ValueError):
        frozen.health()
    with pytest.raises(ValueError):
        adapter.health()


@pytest.mark.archive
@pytest.mark.parametrize(
    "drift",
    [
        dict(IDENTITY, seed=0),
        {"package_digest": IDENTITY["package_digest"]},
        dict(IDENTITY, extra=True),
    ],
)
def test_prediction_identity_drift(server, frozen_modules, worker_name, drift):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    authorize(frozen, adapter)
    observation, archive, history, _ = inputs(frozen_modules, 0)
    server.reply = dict(prediction(observation), identity=drift)
    with pytest.raises(AssertionError, match="identity changed"):
        frozen.predict(observation, archive, history, 0)
    with pytest.raises(ValueError, match="full identity"):
        adapter.predict(payload(frozen, observation, archive, history, 0))


@pytest.mark.archive
@pytest.mark.parametrize(
    ("status", "body", "action"),
    [
        (422, {"code": "invalid_policy_output", "error": "NaN"}, True),
        (422, {"code": "invalid_observation", "error": "Negative gripper aperture"}, True),
        (
            422,
            {"code": "invalid_observation", "error": "Negative gripper aperture", "extra": True},
            False,
        ),
        (422, {"code": "invalid_observation", "error": "Other input"}, False),
        (500, {"code": "invalid_policy_output", "error": "server failure"}, False),
        (503, {"code": "unavailable"}, False),
    ],
)
@pytest.mark.parametrize("profiling", [False, True])
def test_action_vs_transport_errors(
    server, frozen_modules, worker_name, status, body, action, profiling
):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    authorize(frozen, adapter)
    observation, archive, history, _ = inputs(frozen_modules, 0)
    server.status = status
    server.reply = dict(body, profiling={"elapsed": 0.01}) if profiling else body
    with pytest.raises(frozen_modules.control.PolicyActionError if action else RuntimeError):
        frozen.predict(observation, archive, history, 0)
    with pytest.raises(PolicyOutputError if action else PolicyTransportError) as error:
        adapter.predict(payload(frozen, observation, archive, history, 0))
    assert error.value.body == body


@pytest.mark.archive
@pytest.mark.parametrize("endpoint", ["health", "predict"])
def test_malformed_json_and_disconnect(server, frozen_modules, worker_name, endpoint):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    authorize(frozen, adapter)
    observation, archive, history, _ = inputs(frozen_modules, 0)
    call_frozen = (
        frozen.health
        if endpoint == "health"
        else lambda: frozen.predict(observation, archive, history, 0)
    )
    call_adapter = (
        adapter.health
        if endpoint == "health"
        else lambda: adapter.predict(payload(frozen, observation, archive, history, 0))
    )
    if endpoint == "health":
        server.health = b"{broken"
    else:
        server.reply = b"{broken"
    for call in (call_frozen, call_adapter):
        with pytest.raises(json.JSONDecodeError):
            call()
    server.disconnect = True
    with pytest.raises(OSError):
        call_frozen()
    with pytest.raises(PolicyTransportError):
        call_adapter()


@pytest.mark.archive
@pytest.mark.parametrize(
    "field", ["joint_encoder_targets", "gripper_width_m", "eef_encoder_targets"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_actions_rejected_by_frozen_controller(
    server, frozen_modules, worker_name, field, value
):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    authorize(frozen, adapter)
    observation, archive, history, _ = inputs(frozen_modules, 0)
    server.reply = prediction(observation)
    if field == "eef_encoder_targets":
        server.reply.update(
            mode="eef_delta",
            eef_encoder_targets=np.tile(observation["eef_encoder"], (32, 1, 1, 1)).tolist(),
        )
        server.reply[field][0][0][0][0] = value
    elif field == "joint_encoder_targets":
        server.reply[field][0][0][0] = value
    else:
        server.reply[field][0][0] = value
    with pytest.raises(frozen_modules.control.PolicyActionError):
        frozen.predict(observation, archive, history, 0)
    actual = adapter.predict(payload(frozen, observation, archive, history, 0))
    # predict is the lossless transport boundary; command validation stays in control.
    with pytest.raises(frozen_modules.control.PolicyActionError):
        frozen.endpoints(actual, observation)


@pytest.mark.archive
@pytest.mark.parametrize(
    "failure", [TimeoutError("mock timeout"), urllib.error.URLError("mock connection")]
)
def test_mocked_timeouts(server, frozen_modules, worker_name, monkeypatch, failure):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    authorize(frozen, adapter)
    observation, archive, history, _ = inputs(frozen_modules, 0)
    seen = []

    def fail(req, *, timeout):
        seen.append(timeout)
        raise failure

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(type(failure)):
        frozen.startup_health()
    with pytest.raises(type(failure)):
        frozen.health()
    with pytest.raises(PolicyTransportError):
        adapter.health()
    with pytest.raises(type(failure)):
        frozen.predict(observation, archive, history, 0)
    with pytest.raises(PolicyTransportError):
        adapter.predict(payload(frozen, observation, archive, history, 0))
    assert seen == [30, 30, 30, 300, 300]


def test_configurable_timeouts(tmp_path, monkeypatch):
    config = tmp_path / "eval.toml"
    config.write_text(
        '[policy]\nserver = "http://localhost"\n[policy.config]\nhealth_timeout_s = 2.5\npredict_timeout_s = 7\n'
    )
    loaded = load_config(config, ["policy.config.predict_timeout_s=9"])
    adapter = LegacyPredictAdapter.from_config(loaded.policy, expected_identity=IDENTITY)
    seen = []

    def respond(req, *, timeout):
        seen.append(timeout)
        return io.BytesIO(json.dumps({"ready": True, "identity": IDENTITY}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    adapter.health()
    adapter.predict({"images": dict.fromkeys(CAMERAS, "already-encoded")})
    assert seen == [2.5, 9]


@pytest.mark.parametrize("name", ["health_timeout_s", "predict_timeout_s"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "30"])
def test_invalid_timeout(name, value):
    with pytest.raises(ValueError, match="finite and positive"):
        LegacyPredictAdapter("http://localhost", expected_identity=IDENTITY, **{name: value})


@pytest.mark.archive
@pytest.mark.parametrize("use_startup_identity", [False, True])
def test_identity_drift_between_startup_and_worker_health(
    server, frozen_modules, worker_name, use_startup_identity
):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    assert frozen.startup_health() == adapter.health()
    server.health["identity"]["seed"] = 99
    assert frozen.health() == adapter.health()  # Both health checks authorize only the digest.
    observation, archive, history, _ = inputs(frozen_modules, 0)
    server.reply = prediction(observation)
    if not use_startup_identity:
        server.reply["identity"] = copy.deepcopy(server.health["identity"])
    with pytest.raises((AssertionError, RuntimeError), match="identity changed"):
        frozen.predict(observation, archive, history, 0)
    with pytest.raises(ValueError, match="full identity"):
        adapter.predict(payload(frozen, observation, archive, history, 0))


@pytest.mark.archive
def test_no_trace_headers_and_no_invented_reset_or_seed(server, frozen_modules, worker_name):
    frozen_modules.metrics.trace = False
    frozen, adapter = clients(server, frozen_modules, worker_name)
    observation, archive, history, _ = inputs(frozen_modules, 1280)
    server.reply = prediction(observation)
    frozen.health()
    frozen.predict(observation, archive, history, 1280)
    wire = list(server.requests)
    server.requests.clear()
    adapter.reset(seed=999, metadata={"trial_id": "does-not-change-inference"})
    adapter.predict(
        payload(frozen, observation, archive, history, 1280),
        headers=baseline_headers(1280, trace=False),
    )
    assert_wire_equal(wire, server.requests)
    assert "seed" not in json.loads(server.requests[-1][-1])


@pytest.mark.archive
@pytest.mark.parametrize("status", [422, 500])
@pytest.mark.parametrize("body", [b"{broken", b"\xff"])
def test_malformed_error_body_stays_transport_failure(
    server, frozen_modules, worker_name, status, body
):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    authorize(frozen, adapter)
    observation, archive, history, _ = inputs(frozen_modules, 0)
    server.status, server.reply = status, body
    with pytest.raises((json.JSONDecodeError, UnicodeError)):
        frozen.predict(observation, archive, history, 0)
    with pytest.raises(PolicyTransportError):
        adapter.predict(payload(frozen, observation, archive, history, 0))


@pytest.mark.archive
def test_health_422_is_not_an_action_rejection(server, frozen_modules, worker_name):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    server.health_status = 422
    server.health = {"code": "invalid_policy_output"}
    with pytest.raises(urllib.error.HTTPError):
        frozen.health()
    with pytest.raises(PolicyTransportError):
        adapter.health()


@pytest.mark.archive
def test_nonfinite_unexecuted_tail_is_preserved(server, frozen_modules, worker_name):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    authorize(frozen, adapter)
    observation, archive, history, _ = inputs(frozen_modules, 0)
    server.reply = prediction(observation)
    server.reply["joint_encoder_targets"][16][0][0] = float("inf")
    response, actions = frozen.predict(observation, archive, history, 0)
    actual = adapter.predict(payload(frozen, observation, archive, history, 0))
    assert actual == response
    np.testing.assert_array_equal(frozen.endpoints(actual, observation), actions)


@pytest.mark.archive
def test_eef_response_parity_before_native_ik(server, frozen_modules, worker_name):
    frozen, adapter = clients(server, frozen_modules, worker_name)
    authorize(frozen, adapter)
    observation, archive, history, _ = inputs(frozen_modules, 640)
    targets = np.tile(observation["eef_encoder"], (32, 1, 1, 1))
    targets[:, :, :3, 3] = np.arange(32)[:, None, None] / 1000
    server.reply = prediction(observation)
    server.reply.pop("joint_encoder_targets")
    server.reply.update(
        mode="eef_delta",
        eef_encoder_targets=targets.tolist(),
        chunk_anchor_encoder=observation["eef_encoder"],
    )
    server.requests.clear()
    expected, _ = frozen.predict(observation, archive, history, 640, execute=False)
    wire = list(server.requests)
    server.requests.clear()
    obs = payload(frozen, observation, archive, history, 640)
    actual = adapter.predict(obs, headers=baseline_headers(640))
    assert actual == expected
    assert_wire_equal(wire, server.requests)
    chunk = adapter.act(obs)
    for index, row in enumerate(chunk.actions):
        np.testing.assert_array_equal(
            row["eef_encoder_targets"], expected["eef_encoder_targets"][index]
        )
        np.testing.assert_array_equal(row["gripper_width_m"], expected["gripper_width_m"][index])
        np.testing.assert_array_equal(row["chunk_anchor_encoder"], expected["chunk_anchor_encoder"])
    assert len(chunk.actions) == expected["execute_steps"]


@pytest.mark.parametrize(
    "raw",
    [
        b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{",  # IncompleteRead
        b"garbage status line\r\n\r\n",  # BadStatusLine
    ],
)
def test_malformed_http_response_is_transport_failure(raw):
    import socket

    listener = socket.create_server(("127.0.0.1", 0))

    def serve():
        connection, _ = listener.accept()
        with connection:
            connection.recv(65536)
            connection.sendall(raw)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        adapter = LegacyPredictAdapter(
            f"http://127.0.0.1:{listener.getsockname()[1]}", expected_identity=IDENTITY
        )
        with pytest.raises(PolicyTransportError):
            adapter.health()
    finally:
        thread.join(timeout=5)
        listener.close()
