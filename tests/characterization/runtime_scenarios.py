"""CPU runtime adapters and fixed inputs (no simulator or hardware startup)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.maintainer.archive_root import ArchivedPath, ArchivedToolPath

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ArchivedPath("runtime")


def runtime_module(name: str) -> Any:
    sys.path.insert(0, str(ArchivedToolPath("")))
    spec = importlib.util.spec_from_file_location(
        "_p0_" + name, ArchivedToolPath("") / (name + ".py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command_queue_stream() -> dict[str, Any]:
    control = runtime_module("control")
    home = np.tile(np.r_[(control.GENERATION_LOWER + control.GENERATION_UPPER) / 2, 0.06], 2)
    calls = []

    def convert(q: Any) -> Any:
        calls.append(q.tolist())
        result = q.copy()
        result[[1, 9]] += [0.002, -0.003]
        return result

    queue = control.CommandQueue(home, np.arange(14).reshape(2, 7) / 1000, convert)
    endpoints = np.stack(
        [home + np.r_[np.full(7, v), 0.0, np.full(7, -v), 0.0] for v in (0.01, 0.02, 0.03)]
    )
    queue.enqueue(0.0, endpoints[:2])
    stream = []
    for tick in (0, 1, 31, 32, 34, 50, 72, 74, 100, 112):
        encoder, physical = queue.target(tick)
        stream.append(dict(tick=tick, encoder=encoder.tolist(), physical=physical.tolist()))
        encoder[:] = 999  # Returned arrays must not mutate the cached command.
    queue.enqueue(0.2, endpoints[2:])
    stream.append(dict(tick=140, target=[q.tolist() for q in queue.target(140)]))
    rejected = []
    for stamp, values in ((0.0, endpoints), (0.3, np.full((1, 16), np.nan))):
        try:
            queue.enqueue(stamp, values)
        except ValueError as error:
            rejected.append(str(error))
    return dict(stream=stream, conversion_calls=calls, rejected=rejected)


def encoder_observation_output() -> Any:
    worker = runtime_module("runtime_worker")
    q = np.arange(16, dtype=float) / 100
    q[[7, 15]] = [0.04, 0.08]

    def compensate(joints: Any, width: float, arm: str) -> Any:
        result = joints.copy()
        result[1] -= width * (1 if arm == "left" else 2)
        return result

    def fk(joints: Any, width: float) -> Any:
        matrix = np.eye(4)
        matrix[:3, 3] = joints[:3]
        return {"gripper_base": matrix}

    replay = SimpleNamespace(
        j2comp=SimpleNamespace(encoder_from_physical=compensate), urdf_fk=SimpleNamespace(fk=fk)
    )
    sim = SimpleNamespace(
        read=lambda: SimpleNamespace(qpos=q),
        env=SimpleNamespace(
            lula=SimpleNamespace(
                tcp_rotation=Rotation.from_euler("z", 90, degrees=True),
                tcp_offset=np.array([0.01, 0.02, 0.03]),
            )
        ),
    )
    return worker.encoder_observation(sim, replay)
