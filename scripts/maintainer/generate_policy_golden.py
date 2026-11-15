"""Generate synthetic decoded-data evidence using the base revision's transport.

Run with ``python -m scripts.maintainer.generate_policy_golden``. The base is
read from git; no archived implementation is copied into the product tree.
"""

from __future__ import annotations

import base64
import importlib.util
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


def observation(index, tick):
    from PIL import Image

    pixels = (np.arange(36).reshape(3, 4, 3) + index + tick).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    anchor = np.tile(np.eye(4), (2, 1, 1))
    anchor[:, 0, 3] = index / 100
    return dict(
        coordinate_frame="encoder_arm_base",
        prompt=f"synthetic {tick}",
        joint_encoder=np.zeros((2, 7)).tolist(),
        eef_encoder=anchor.tolist(),
        gripper_width_m=[-0.01 if index == 1 else 0.05, 0.05],
        images={name: base64.b64encode(stream.getvalue()).decode() for name in CAMERAS},
    )


def prediction(obs):
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


def solver():
    return SimpleNamespace(
        solve=lambda poses, seed, **kw: dict(
            accepted=True, joints=np.zeros((16, 2, 7)).tolist(), diagnostics=[]
        )
    )


def base_module(relative, name) -> Any:
    source = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT)
    spec = importlib.util.spec_from_loader(name, loader=None)
    if spec is None:
        raise RuntimeError("Cannot create historical module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, relative, "exec"), module.__dict__)
    return module


def generate():
    base_module(
        "src/tower-eval/src/tower_eval/policies/legacy_predict.py",
        "tower_eval.policies.legacy_predict",
    )
    base_module(
        "packages/tower-sim/src/tower_sim/bench/mme_history.py", "tower_sim.bench.mme_history"
    )
    runtime = base_module(
        "packages/tower-sim/src/tower_sim/bench/policy.py", "tower_sim.bench._golden_policy"
    )
    control = base_module(
        "packages/tower-sim/src/tower_sim/bench/control.py", "tower_sim.bench._golden_control"
    )
    evaluation = base_module(
        "packages/tower-sim/src/tower_sim/bench/evaluation.py", "tower_sim.bench._golden_evaluation"
    )
    resources = base_module(
        "packages/tower-sim/src/tower_sim/bench/resources.py", "tower_sim.bench._golden_resources"
    )
    # Resolve the historical runtime's import without retaining historical code.

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
                self.reply(200, prediction(obs))
            except ValueError as exc:
                self.reply(422, {"code": "invalid_observation", "error": str(exc)})

        def reply(self, code, body):
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    server = ThreadingHTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    from tower_sim.bench.metrics import Metrics, TrialRequestID

    class Sink:
        def emit(self, record):
            pass

        def flush(self):
            pass

    records = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            worker = runtime.OrdinaryPolicy(
                package=IDENTITY,
                url=f"http://localhost:{server.server_port}",
                prompt="synthetic 0",
                task="synthetic",
                metrics=Metrics(
                    Sink(),
                    TrialRequestID("synthetic"),
                ),
                solver=solver(),
            )
            health = worker.authorize()
            worker.start(output)
            rows = [{"candidate": {"seed": i}, "output": output / str(i)} for i in range(3)]
            for row in rows:
                row["output"].mkdir()
            worker.initialize(rows, CAMERAS, health)
            context = evaluation.EvaluationContext.__new__(evaluation.EvaluationContext)
            context.health = health
            try:
                for tick in (0, 640):
                    worker.prompt = f"synthetic {tick}"
                    observations = {i: observation(i, tick) for i in range(3)}
                    archives = [SimpleNamespace(latest=observations[i]["images"]) for i in range(3)]
                    pending = worker.prefetch(observations, archives, tick)
                    for i, obs in observations.items():
                        try:
                            response = worker.prediction(pending, i, obs, tick)
                        except runtime.control.PolicyActionError:
                            rejection = json.loads(
                                (rows[i]["output"] / "policy-input-rejection.json").read_text()
                            )
                            records.append(dict(tick=tick, env_idx=i, rejection=rejection))
                            continue
                        context.verify_prediction(response["identity"])
                        commands, diagnostics = control.encoder_endpoints(
                            response, obs, solver=solver()
                        )
                        evidence = dict(
                            observation_time_s=tick / 400, observation=obs, response=response
                        )
                        resources.save(output / "evidence.json", evidence)
                        records.append(
                            dict(
                                tick=tick,
                                env_idx=i,
                                response=response,
                                verified_identity=response["identity"],
                                commands=commands.tolist(),
                                diagnostics=diagnostics,
                                evidence=json.loads((output / "evidence.json").read_text()),
                            )
                        )
            finally:
                worker.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(dict(base=BASE, health=health, records=records), separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    generate()
