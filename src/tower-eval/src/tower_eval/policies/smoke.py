# SPDX-License-Identifier: Apache-2.0
"""Launch XPolicyLab's external CPU demo for an end-to-end transport smoke."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from .adapter import ActionChunk
from .xpolicylab import CAMERAS, FRAME, XPolicyLabAdapter, client_root


def smoke_observation() -> dict[str, Any]:
    """Synthetic fixture only; never used as a real robot observation."""
    pixels = np.zeros((24, 32, 3), dtype=np.uint8)
    pixels[..., 0], pixels[..., 2] = 193, 17
    return {
        "coordinate_frame": FRAME,
        "instruction": "CPU transport smoke",
        "joint_encoder": np.zeros((2, 7)),
        "eef_encoder": np.tile(np.eye(4), (2, 1, 1)),
        "gripper_width_m": np.full(2, 0.05),
        "images": {name: pixels.copy() for name in CAMERAS},
    }


def cpu_smoke_chunk(chunk: ActionChunk) -> ActionChunk:
    """Explicit test-only reinterpretation of synthetic demo joints; never robot data."""
    if chunk.metadata.get("mode") != "joint_absolute":
        raise ValueError("CPU smoke supports only synthetic joint actions")
    return ActionChunk(
        chunk.actions,
        chunk.dt_s,
        {**chunk.metadata, "coordinate_frame": "physical", "synthetic_fixture": True},
    )


def run_xpolicylab_smoke() -> dict[str, Any]:
    root = client_root()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join((str(root), os.environ.get("PYTHONPATH", "")))
    )
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(root / "setup_policy_server.py"),
                "--config_path",
                str(root / "policy/demo_policy/deploy.yml"),
                "--overrides",
                f"port={port}",
                "host=127.0.0.1",
                "env_cfg_type=nero_dual",
                "action_type=joint",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        adapter = None
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError("XPolicyLab demo exited: " + log.read()[-4000:])
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("XPolicyLab demo startup timed out")
            adapter = XPolicyLabAdapter(f"ws://127.0.0.1:{port}")
            adapter.reset(seed=42)
            observation = smoke_observation()
            chunk = adapter.act(observation)
            if np.asarray(chunk.actions).shape != (2, 16):
                raise RuntimeError("Incorrect demo action shape")
            batches = adapter.act_batch(
                [dict(observation, env_idx=7), dict(observation, env_idx=2)]
            )
            if len(batches) != 2 or any(np.asarray(c.actions).shape != (2, 16) for c in batches):
                raise RuntimeError("Incorrect demo batch shape")
            # Exercise the public evaluator with explicit synthetic modalities.
            from ..envs import TowerEnv
            from ..evaluation import evaluate_policy
            from ..simulator import CpuReferenceBackend
            from .local import LocalPolicyAdapter

            connected = adapter

            def act_cpu(_observation: Any, _info: Any = None) -> ActionChunk:
                return cpu_smoke_chunk(connected.act(observation))

            result = evaluate_policy(
                TowerEnv(CpuReferenceBackend(), max_steps=3),
                LocalPolicyAdapter(act_cpu),
                seed=42,
                max_steps=3,
            )
            return {"status": "ok", "policy": "xpolicylab://demo_policy", "batch_size": 2, **result}
        finally:
            try:
                if adapter is not None:
                    adapter.close()
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
