"""Generate synthetic decoded-data evidence using the base revision's transport.

Run with ``python -m scripts.maintainer.generate_policy_golden``. The base is
read from git; no archived implementation is copied into the product tree.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

BASE = "5a52a1d417b70a62bd25d242a59f11f79796c805"
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "src/tower-eval/tests/fixtures/policy_golden.json"
IDENTITY = {"package_digest": "synthetic-package", "seed": 42, "horizon": 32, "execute_steps": 16}
CAMERAS = ("right_front_camera", "left_back_camera", "left_wrist_camera", "back_wrist_camera")


def pixels(index, tick):
    return (np.arange(36).reshape(3, 4, 3) + index + tick).astype(np.uint8)


def images(index, tick):
    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(pixels(index, tick)).save(stream, format="PNG")
    return {name: base64.b64encode(stream.getvalue()).decode() for name in CAMERAS}


def observation(index, tick):
    anchor = np.tile(np.eye(4), (2, 1, 1))
    anchor[:, 0, 3] = index / 100
    return dict(
        joint_encoder=np.zeros((2, 7)).tolist(),
        eef_encoder=anchor.tolist(),
        gripper_width_m=[-0.01 if index == 1 else 0.05, 0.05],
    )


def payload(index=0, tick=0):
    return dict(
        observation(index, tick),
        images=images(index, tick),
        coordinate_frame="encoder_arm_base",
        prompt=f"synthetic {tick}",
        env_idx=index,
    )


def variant(obs, case):
    result = prediction(obs)
    if case == "aperture":
        result["gripper_width_m"][0][0] = 0.2
    elif case == "nonrigid":
        result["eef_encoder_targets"][0][0][0][0] = 2.0
    elif case == "anchor":
        result["chunk_anchor_encoder"][0][0][3] += 0.25
    elif case == "shape":
        result["eef_encoder_targets"][0][0] = [[1.0] * 3] * 3
    elif case == "nan":
        result["gripper_width_m"][0][0] = float("nan")
    return result


def prediction(obs) -> dict[str, Any]:
    if obs["gripper_width_m"][0] < 0:
        raise ValueError("Negative gripper aperture")
    tick = int(obs["prompt"].split()[-1])
    mode = "eef_delta" if tick else "joint_absolute"
    joints = np.zeros((32, 2, 7), dtype=np.float64)
    joints[:, :, 0] = np.arange(32)[:, None] / 10000
    joints[0, 0, 0] = -0.0
    widths = np.tile([np.nextafter(0.05, 1), 0.04], (32, 1))
    result = dict(
        mode=mode,
        coordinate_frame="encoder_arm_base",
        control_hz=10,
        execute_steps=16,
        units={"joint": "rad", "translation": "m", "gripper_width": "m", "predicted_j8": "mm"},
        applied_j8=[0, 0],
        j8_actuator_mask=[False, False],
        identity=dict(IDENTITY),
        native_actions=np.arange(512, dtype=np.float64).reshape(32, 16).tolist(),
        gripper_width_m=widths.tolist(),
        inference_seconds=0.125,
    )
    if mode == "joint_absolute":
        result["joint_encoder_targets"] = joints.tolist()
    else:
        result["chunk_anchor_encoder"] = np.asarray(obs["eef_encoder"]).tolist()
        targets = np.tile(obs["eef_encoder"], (32, 1, 1, 1))
        targets[:, :, 0, 3] += np.arange(32)[:, None] / 10000
        result["eef_encoder_targets"] = targets.tolist()
    return result


def solver(reject=False):
    return SimpleNamespace(
        solve=lambda poses, seed, **kw: dict(
            accepted=not reject,
            joints=np.zeros((16, 2, 7)).tolist(),
            diagnostics=[{"reason": "synthetic_ik_rejection"}] if reject else [],
        )
    )


def capture(worker, observations, archives, tick, directory):
    from tower_sim.bench import control
    from tower_sim.bench.evaluation import EvaluationContext
    from tower_sim.bench.resources import save

    pending = worker.prefetch(observations, archives, tick)
    context = EvaluationContext.__new__(EvaluationContext)
    context.health = worker.authorize()
    records = []
    for i, obs in observations.items():
        dest = directory / str(i)
        record = dict(tick=tick, env_idx=i)
        try:
            response = worker.prediction(pending, i, obs, tick)
            context.verify_prediction(response["identity"])
            save(
                dest / "chunk.json",
                dict(observation_time_s=tick / 400, observation=obs, response=response),
            )
            commands, diagnostics = control.encoder_endpoints(response, obs, solver=worker.solver)
            record.update(commands=commands.tolist(), diagnostics=diagnostics)
        except (ValueError, RuntimeError) as exc:
            record["exception"] = type(exc).__name__
            record["message"] = str(exc)
        record["files"] = {p.name: json.loads(p.read_text()) for p in dest.glob("*.json")}
        records.append(record)
    return records


def generate_base(destination):
    from tower_sim.bench import policy as runtime
    from tower_sim.bench.metrics import Metrics, TrialRequestID

    case = "valid"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.reply(200, dict(ready=True, identity=IDENTITY))

        def do_POST(self):
            from PIL import Image

            obs = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            obs["images"] = {
                name: np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"))
                for name, value in obs["images"].items()
            }
            try:
                self.reply(200, variant(obs, case))
            except ValueError as exc:
                self.reply(422, {"code": "invalid_observation", "error": str(exc)})

        def reply(self, code, body):
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    class Sink:
        def emit(self, record):
            pass

        def flush(self):
            pass

    server = ThreadingHTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    cases = {}
    statuses = {}
    loop_files = {}
    try:
        for case in ("valid", "aperture", "nonrigid", "anchor", "shape", "nan", "ikreject"):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                worker = runtime.OrdinaryPolicy(
                    package=dict(IDENTITY, checkpoint_sha256="synthetic-checkpoint"),
                    url=f"http://localhost:{server.server_port}",
                    prompt="synthetic 0",
                    task="synthetic",
                    metrics=Metrics(Sink(), TrialRequestID("synthetic")),
                    solver=solver(case == "ikreject"),
                )
                health = worker.authorize()
                worker.start(output)
                rows = [{"candidate": {"seed": i}, "output": output / str(i)} for i in range(3)]
                for row in rows:
                    row["output"].mkdir()
                worker.initialize(rows, CAMERAS, health)
                records = []
                try:
                    for tick in (0, 640) if case == "valid" else (640,):
                        worker.prompt = f"synthetic {tick}"
                        obs = {i: observation(i, tick) for i in range(3)}
                        records.extend(
                            capture(
                                worker,
                                obs,
                                [SimpleNamespace(latest=images(i, tick)) for i in range(3)],
                                tick,
                                output,
                            )
                        )
                finally:
                    worker.close()
                cases[case] = records
        # Run the historical shared loop, preserving its actual status serialization.
        from importlib import import_module

        import pytest

        simulation = import_module("test_replay_lifecycle").simulation

        for case in ("valid", "aperture", "nonrigid", "anchor", "shape", "nan", "ikreject"):
            with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
                output = Path(directory)
                sim = simulation.__wrapped__(output, patch, SimpleNamespace(param=1))
                sim.camera.names = {name: str(i) for i, name in enumerate(CAMERAS)}
                sim.factories.observation = lambda *a: observation(0, 640)
                worker = runtime.OrdinaryPolicy(
                    package=dict(IDENTITY, checkpoint_sha256="synthetic-checkpoint"),
                    url=f"http://localhost:{server.server_port}",
                    prompt="synthetic 640",
                    task="synthetic",
                    metrics=Metrics(Sink(), TrialRequestID("synthetic")),
                    solver=SimpleNamespace(identity={}, solve=solver(case == "ikreject").solve),
                )
                worker.authorize()
                with contextlib.suppress(ValueError, RuntimeError):
                    sim.run(policy=worker)
                statuses[case] = (sim.row["output"] / "status.json").read_text()
                loop_files[case] = {
                    p.relative_to(sim.row["output"]).as_posix(): json.loads(p.read_text())
                    for p in sim.row["output"].rglob("*.json")
                    if p.parent.name == "chunks"
                    or p.name
                    in {
                        "status.json",
                        "policy-action-failure.json",
                        "policy-input-rejection.json",
                        "policy-error.json",
                    }
                }
        # Assert every imported product/test module came from the base worktree.
        base = Path.cwd().resolve()
        for name, module in sys.modules.items():
            if (
                name.startswith(("tower_sim", "tower_eval", "nero_kinematics", "scripts."))
                or name == "test_replay_lifecycle"
            ):
                source = getattr(module, "__file__", None)
                if source is not None and not Path(source).resolve().is_relative_to(base):
                    raise RuntimeError(f"Non-base module imported: {name}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    destination.write_text(
        json.dumps(
            dict(
                base=BASE,
                health=health,
                cases=cases,
                statuses=statuses,
                loop_files=loop_files,
                pixels={
                    str(tick): [pixels(i, tick).tolist() for i in range(3)] for tick in (0, 640)
                },
            ),
            separators=(",", ":"),
        )
        + "\n"
    )


def generate():
    import os

    with tempfile.TemporaryDirectory(prefix="tower-golden-") as directory:
        checkout = Path(directory) / "base"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(checkout), BASE], cwd=ROOT, check=True
        )
        try:
            roots = [
                checkout,
                checkout / "src/tower-eval/src",
                checkout / "packages/tower-sim/src",
                checkout / "packages/nero-kinematics/src",
                checkout / "packages/tower-sim/tests",
            ]
            env = dict(
                os.environ,
                PYTHONPATH=os.pathsep.join(map(str, roots)),
                PYTHONDONTWRITEBYTECODE="1",
                NO_PROXY="*",
            )
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--base-output", str(FIXTURE)],
                cwd=checkout,
                env=env,
                check=True,
            )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(checkout)], cwd=ROOT, check=True
            )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--base-output":
        generate_base(Path(sys.argv[2]))
    else:
        generate()
