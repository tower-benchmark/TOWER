# SPDX-License-Identifier: Apache-2.0
"""Environment-side XPolicyLab bridge; no policy modules are imported here."""

from __future__ import annotations

import contextlib
import importlib
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .adapter import ActionChunk

CAMERAS = (
    "right_front_camera",
    "left_back_camera",
    "left_wrist_camera",
    "back_wrist_camera",
)
FRAME = "encoder_arm_base"


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return array.copy()


def _transform(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = _array(value, shape, label)
    rotation = array[..., :3, :3]
    if (
        not np.allclose(array[..., 3, :], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-5)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5)
    ):
        raise ValueError(f"{label} must contain proper rigid transforms")
    return array


def observation_to_xpolicylab(observation: dict[str, Any]) -> dict[str, Any]:
    """Map already-compensated encoder observations without changing coordinates."""
    if observation.get("coordinate_frame") != FRAME:
        raise ValueError("XPolicyLab requires explicit encoder_arm_base observations")
    instruction = observation.get("instruction", observation.get("prompt"))
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty instruction is required")
    vision = {}
    for name in CAMERAS:
        pixels = np.asarray(observation["images"][name])
        if pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
            raise ValueError(f"{name} must be RGB uint8 HxWx3")
        vision[name] = {"color": pixels.copy()}
    return {
        "vision": vision,
        "state": {
            "joint_encoder": _array(observation["joint_encoder"], (2, 7), "joint_encoder"),
            "eef_encoder": _transform(observation["eef_encoder"], (2, 4, 4), "eef_encoder"),
            "gripper_width_m": _array(observation["gripper_width_m"], (2,), "gripper_width_m"),
        },
        "instruction": instruction,
        "coordinate_frame": FRAME,
        **({"env_idx": int(observation["env_idx"])} if "env_idx" in observation else {}),
    }


def decode_chunk(chunk: Any, mode: str, *, expected_anchor: Any = None) -> ActionChunk:
    """Decode XPL endpoints, retaining encoder targets for the runtime controller."""
    if mode not in {"joint_absolute", "eef_delta"}:
        raise ValueError(f"Unsupported TOWER action mode: {mode}")
    if not isinstance(chunk, list) or not chunk:
        raise ValueError("XPolicyLab returned an empty or malformed action chunk")
    anchor = None
    if mode == "eef_delta":
        if expected_anchor is None:
            raise ValueError("EEF decoding requires the request's eef_encoder anchor")
        anchor = _transform(expected_anchor, (2, 4, 4), "request anchor")
    actions = []
    for row in chunk:
        if not isinstance(row, dict):
            raise ValueError("Every XPolicyLab endpoint must be a mapping")
        # Untagged standard XPL joint dictionaries are supported by explicit
        # joint_absolute configuration (including demo_policy). EEF is tagged.
        if row.get("mode", mode) != mode or row.get("coordinate_frame", FRAME) != FRAME:
            raise ValueError("Policy action mode/frame does not match the TOWER contract")
        if mode == "eef_delta" and row.get("coordinate_frame") != FRAME:
            raise ValueError("EEF targets require an explicit encoder frame")
        expected_units = {
            "joint": "rad",
            "translation": "m",
            "gripper_width": "m",
            "predicted_j8": "mm",
        }
        if "units" in row and row["units"] != expected_units:
            raise ValueError("Policy action units do not match the TOWER contract")
        if row.get("control_hz", 10) != 10:
            raise ValueError("TOWER requires 10 Hz endpoints")
        if row.get("j8_actuator_mask", [False, False]) != [False, False]:
            raise ValueError("Predicted J8 must remain masked")
        if "execute_steps" in row and row["execute_steps"] != len(chunk):
            raise ValueError("Policy chunk length differs from its execution prefix")
        widths = np.array(
            [
                _array(row[f"{arm}_ee_joint_state"], (1,), f"{arm} gripper")[0]
                for arm in ("left", "right")
            ]
        )
        if np.any(widths < 0) or np.any(widths > 0.1 + 4e-6):
            raise ValueError("Policy gripper aperture is outside the TOWER command contract")
        if mode == "joint_absolute":
            joints = [_array(row[f"{arm}_arm_joint_state"], (7,), arm) for arm in ("left", "right")]
            actions.append(np.concatenate((joints[0], widths[:1], joints[1], widths[1:])))
        else:
            returned_anchor = _transform(row["chunk_anchor_encoder"], (2, 4, 4), "anchor")
            if anchor is None or not np.array_equal(returned_anchor, anchor):
                raise ValueError("Policy chunk anchor differs from the request's eef_encoder")
            targets = np.stack(
                [_transform(row[f"{arm}_ee_transform"], (4, 4), arm) for arm in ("left", "right")]
            )
            actions.append(
                {
                    "mode": mode,
                    "coordinate_frame": FRAME,
                    "eef_encoder_targets": targets,
                    "gripper_width_m": widths,
                    "chunk_anchor_encoder": returned_anchor,
                    "j8_actuator_mask": [False, False],
                }
            )
    return ActionChunk(
        np.asarray(actions) if mode == "joint_absolute" else actions,
        0.1,
        {"mode": mode, "coordinate_frame": FRAME, "identity": chunk[0].get("identity", {})},
    )


def client_root(root: str | Path | None = None) -> Path:
    """Find a checkout in source installs, or use an explicit installed-client path."""
    if root is not None:
        candidates = [Path(root).expanduser().resolve()]
    else:
        candidates = [parent / "third_party/XPolicyLab" for parent in Path(__file__).parents]
    for candidate in candidates:
        if (candidate / "client_server/ws/model_client.py").is_file():
            return candidate
    raise RuntimeError("Initialize third_party/XPolicyLab or set policy.config.client_root")


class XPolicyLabAdapter:
    def __init__(
        self,
        url: str,
        *,
        mode: str = "joint_absolute",
        timeout_s: float = 120.0,
        root: str | Path | None = None,
        max_connect_attempts: int = 200,
        connect_timeout_s: float = 10.0,
        max_connect_seconds: float = 180.0,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("XPolicyLab server URL must not be empty")
        if urlsplit(url).scheme not in {"ws", "wss"} or not urlsplit(url).hostname:
            raise ValueError("XPolicyLab URL must use ws:// or wss:// with a hostname")
        if mode not in {"joint_absolute", "eef_delta"}:
            raise ValueError("mode must be joint_absolute or eef_delta")
        checkout = client_root(root)
        sys.path.insert(0, str(checkout))
        try:
            module = importlib.import_module("client_server.ws.model_client")
        except ImportError as exc:
            raise RuntimeError("Install tower-eval[xpolicylab] transport dependencies") from exc
        finally:
            sys.path.remove(str(checkout))
        self.mode = mode
        self._episode_active = False
        self._closed = False
        # Upstream writes connection diagnostics to stdout, including from its
        # loop thread. Keep command stdout available for machine-readable JSON.
        with contextlib.redirect_stdout(sys.stderr):
            self._client = module.WsModelClient(
                url=url,
                evaluation_id=uuid.uuid4().hex,
                trial_id=uuid.uuid4().hex,
                action_case_id=uuid.uuid4().hex,
                request_timeout_s=timeout_s,
                max_connect_attempts=max_connect_attempts,
                connect_timeout_s=connect_timeout_s,
                max_connect_seconds=max_connect_seconds,
                connect_retry_delay_s=1.0,
            )

    def _call(self, name: str, payload: Any = None) -> Any:
        with contextlib.redirect_stdout(sys.stderr):
            if name != "get_identity":
                return self._client.call(name, payload)
            config = self._client._client.config
            timeout = config.request_timeout_s
            try:
                if name == "get_identity":
                    config.request_timeout_s = 30.0
                return self._client.call(name, payload)
            finally:
                config.request_timeout_s = timeout

    def infer(self, observation: Any) -> Any:
        """Atomically update an observation and obtain its action rows."""
        # WsModelClient has no sync INFER wrapper at XPolicyLab tower-1.0.1.
        # Keep protocol access and diagnostics redirection at this boundary.
        with contextlib.redirect_stdout(sys.stderr):
            client = self._client
            response = client._run(
                client._client.infer(
                    observation,
                    trial_id=client.trial_id,
                    action_case_id=client.action_case_id,
                    repeat_index=client.repeat_index,
                    step=client._step,
                )
            )
            client._step += 1
            return response.payload["actions"]

    def _end_episode(self) -> None:
        if self._episode_active:
            self._episode_active = False
            self._call("trial_end", {})

    def reset(self, *, seed: int | None = None, metadata: dict[str, Any] | None = None) -> None:
        self._end_episode()
        self._client.trial_id = str((metadata or {}).get("trial_id", uuid.uuid4().hex))
        self._call("prepare_case", {**(metadata or {}), "episode_seed": seed})
        self._call("reset")
        self._episode_active = True

    def act(self, observation: Any, info: dict[str, Any] | None = None) -> ActionChunk:
        mapped = observation_to_xpolicylab(observation)
        self._call("update_obs", mapped)
        return decode_chunk(
            self._call("get_action"), self.mode, expected_anchor=mapped["state"]["eef_encoder"]
        )

    def act_batch(self, observations: list[dict[str, Any]]) -> list[ActionChunk]:
        if not observations:
            raise ValueError("An observation batch cannot be empty")
        indices = [int(obs.get("env_idx", index)) for index, obs in enumerate(observations)]
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate environment index")
        mapped = [
            observation_to_xpolicylab(dict(obs, env_idx=index))
            for obs, index in zip(observations, indices)
        ]
        self._call("update_obs_batch", mapped)
        chunks = self._call("get_action_batch", indices)
        if len(chunks) != len(indices):
            raise ValueError("Policy returned the wrong batch size")
        return [
            decode_chunk(chunk, self.mode, expected_anchor=obs["state"]["eef_encoder"])
            for chunk, obs in zip(chunks, mapped)
        ]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._end_episode()
        finally:
            with contextlib.redirect_stdout(sys.stderr):
                self._client.close()
