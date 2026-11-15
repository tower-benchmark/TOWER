# SPDX-License-Identifier: Apache-2.0
"""Backend- and policy-agnostic evaluation loop for the public API."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..constants import _SCHEMAS
from ..envs import TowerEnv
from ..policies.adapter import PolicyAdapter, normalize_actions


def _hash_observation(digest: Any, value: Any) -> None:
    """Hash an observation deterministically, including image array bytes."""
    if isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=str):
            _hash_observation(digest, str(key))
            _hash_observation(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence" + str(len(value)).encode())
        for item in value:
            _hash_observation(digest, item)
        return
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        digest.update(b"array" + str(contiguous.dtype).encode())
        digest.update(repr(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
        return
    if isinstance(value, np.generic):
        _hash_observation(digest, value.item())
        return
    digest.update(json.dumps(value, sort_keys=True, default=str).encode())


@dataclass
class _Progress:
    total_reward: float = 0.0
    steps: int = 0
    terminated: bool = False
    truncated: bool = False
    observation_digest: Any = None

    def __post_init__(self):
        self.observation_digest = hashlib.sha256()


def _rollout(env, policy, seed, limit, progress):
    observation, info = env.reset(seed=seed)
    policy.reset(
        seed=seed,
        metadata={
            "protocol": _SCHEMAS["config"],
            "action_protocol": _SCHEMAS["dual_action"]
            if hasattr(env.action_space, "description")
            else "absolute_joint16",
        },
    )
    while progress.steps < limit and not (progress.terminated or progress.truncated):
        _hash_observation(progress.observation_digest, observation)
        for action in normalize_actions(
            policy.act(observation, info),
            env.action_dt_s,
            accepted_frame=env.action_coordinate_frame,
        ):
            observation, reward, progress.terminated, progress.truncated, info = env.step(action)
            progress.total_reward += float(reward)
            progress.steps += 1
            if progress.steps >= limit or progress.terminated or progress.truncated:
                break
    return info


def run_episode(
    env: TowerEnv,
    policy: PolicyAdapter,
    *,
    seed: int = 0,
    max_steps: int | None = None,
    output: str | Path | None = None,
    config: Any | None = None,
) -> dict[str, Any]:
    """Run one episode; close the environment and leave the policy to its owner.

    The environment supplies reset/step, action timing and coordinate-frame
    validation. Any backend wrapped by TowerEnv can use the same loop; this
    module performs no backend discovery or process/environment configuration.
    """
    if seed < 0:
        env.close()
        raise ValueError("seed must be non-negative")
    limit = max_steps or env.max_steps
    started = time.perf_counter()
    progress = _Progress()
    try:
        info = _rollout(env, policy, seed, limit, progress)
    except BaseException:
        # Keep the episode error; a secondary close failure must not replace it.
        with contextlib.suppress(Exception):
            env.close()
        raise
    env.close()
    result = _episode_result(env, policy, seed, config, started, info, progress)
    if output is not None:
        _write_result(output, result)
    return result


def _episode_result(env, policy, seed, config, started, info, progress):
    space: Any = env.action_space
    return {
        "schema": _SCHEMAS["result"],
        "evaluator_version": __version__,
        "seed": seed,
        "steps": progress.steps,
        "total_reward": progress.total_reward,
        "terminated": progress.terminated,
        "truncated": progress.truncated,
        "task_success": info.get("task_success"),
        "provenance": {
            "backend": getattr(env.backend, "capabilities", {"name": type(env.backend).__name__}),
            "policy": {"name": type(policy).__name__},
            "action_space": space.description()
            if hasattr(space, "description")
            else {"low": np.asarray(space.low).tolist(), "high": np.asarray(space.high).tolist()},
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode()
            ).hexdigest()
            if config is not None
            else None,
            "observation_sha256": progress.observation_digest.hexdigest(),
        },
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "metrics": {
            "valid": False,
            "invalid_reason": "backend does not provide physical task events",
            "task_success_rate": 0.0,
            "grasp_success_rate": 0.0,
            "place_success_rate": 0.0,
            "long_horizon_progress_rate": 0.0,
            "disturbed_block_count": None,
        },
    }


def _write_result(output, result):
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(destination)
    except BaseException:
        with suppress(FileNotFoundError):
            Path(temporary_name).unlink()
        raise
