"""Benchmark response contract over XPolicyLab's WebSocket transport."""

from __future__ import annotations

import base64
import copy
import io
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .xpolicylab import CAMERAS, XPolicyLabAdapter, decode_chunk, observation_to_xpolicylab


class _PolicyError(RuntimeError):
    """Connection, protocol, or policy input failure."""

    def __init__(self, message: str, *, body: Any = None, body_is_json: bool = True):
        super().__init__(message)
        self.body, self.body_is_json = body, body_is_json


class PolicyTransportError(_PolicyError):
    """Connection, protocol, or policy input failure."""


class PolicyOutputError(_PolicyError):
    """Invalid policy output or the frozen aperture observation rejection."""


def validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise ValueError("policy_url must use ws:// or wss:// with a hostname")


def map_error(exc: Exception) -> PolicyTransportError | PolicyOutputError:
    code = getattr(getattr(exc, "code", None), "value", None)
    details = getattr(exc, "details", {})
    category = details.get("exception_type") if isinstance(details, dict) else None
    message = str(exc)
    if code in {"call_failed", "infer_failed"}:
        if category == "PolicyOutputError":
            return PolicyOutputError(
                message, body={"code": "invalid_policy_output", "error": message}
            )
        if category == "ValueError" and message == "Negative gripper aperture":
            return PolicyOutputError(
                message, body={"code": "invalid_observation", "error": message}
            )
    return PolicyTransportError(message)


def reconstruct(rows: list[dict], mode: str, anchor: Any, full: dict | None) -> dict:
    """Validate every endpoint and retain all optional full-horizon evidence verbatim."""
    decode_chunk(rows, mode, expected_anchor=anchor)
    result = {
        k: copy.deepcopy(v)
        for k, v in rows[0].items()
        if k
        in {
            "mode",
            "coordinate_frame",
            "units",
            "control_hz",
            "execute_steps",
            "applied_j8",
            "j8_actuator_mask",
            "identity",
            "chunk_anchor_encoder",
        }
    }
    result.setdefault("mode", mode)
    result.setdefault("coordinate_frame", "encoder_arm_base")
    result.setdefault("control_hz", 10)
    result.setdefault("execute_steps", len(rows))
    result.setdefault("j8_actuator_mask", [False, False])
    result["gripper_width_m"] = [
        [r[f"{arm}_ee_joint_state"][0] for arm in ("left", "right")] for r in rows
    ]
    field = "joint_encoder_targets" if mode == "joint_absolute" else "eef_encoder_targets"
    suffix = "arm_joint_state" if mode == "joint_absolute" else "ee_transform"
    result[field] = [[r[f"{arm}_{suffix}"] for arm in ("left", "right")] for r in rows]
    if all("native_action" in r for r in rows):
        result["native_actions"] = [r["native_action"] for r in rows]
    arrays = {field, "gripper_width_m", "native_actions"}
    for key in arrays | {"chunk_anchor_encoder"}:
        if key in result:
            result[key] = np.asarray(result[key], dtype=np.float64).tolist()
    for row in rows:
        if row.get("identity", {}) != result.get("identity", {}):
            raise ValueError("Endpoint identities differ within the chunk")
    if full is None:
        result["missing_evidence"] = ["full_prediction", "unexecuted_horizon", "inference_seconds"]
        result["missing_evidence"].extend(
            key for key in ("native_actions", "applied_j8", "units") if key not in result
        )
        return result
    for key, value in result.items():
        expected = full[key][: len(rows)] if key in arrays else full[key]
        if key in arrays | {"chunk_anchor_encoder"}:
            a, b = np.asarray(value, dtype=np.float64), np.asarray(expected, dtype=np.float64)
            equal = a.shape == b.shape and a.tobytes() == b.tobytes()
        else:
            equal = value == expected
        if not equal:
            raise ValueError(f"Endpoint rows differ from full prediction: {key}")
    for row in rows:
        for key in (
            "identity",
            "mode",
            "coordinate_frame",
            "units",
            "control_hz",
            "execute_steps",
            "applied_j8",
            "j8_actuator_mask",
        ):
            if key in row and row[key] != full[key]:
                raise ValueError(f"Endpoint metadata differs from full prediction: {key}")
    # Replace only the executed rows with the checked endpoint values; retain the
    # exact key set and the untouched suffix and auxiliary evidence fields.
    response = copy.deepcopy(full)
    for key in arrays & result.keys():
        response[key] = result[key] + full[key][len(rows) :]
    return response


class RuntimePolicyAdapter:
    def __init__(self, url: str, *, expected_identity: dict[str, Any], bridge=None, root=None):
        validate_url(url)
        if not expected_identity.get("package_digest"):
            raise ValueError("Expected identity must include package_digest")
        self.url, self.package_digest = url, expected_identity["package_digest"]
        self.bridge = bridge
        self.root = root
        self._health_identity = self._latest_health_identity = None

    def _call(self, name, payload=None):
        if self.bridge is None:
            raise PolicyTransportError("Policy connection has not been authorized")
        try:
            return self.bridge._call(name, payload)
        except Exception as exc:
            raise map_error(exc) from exc

    def health(self):
        if self.bridge is None:
            try:
                self.bridge = XPolicyLabAdapter(self.url, root=self.root)
            except Exception as exc:
                raise map_error(exc) from exc
        identity = self._call("get_identity")
        if identity["package_digest"] != self.package_digest:
            raise ValueError("Policy is not ready for the authorized package")
        if self._health_identity is None:
            self._health_identity = copy.deepcopy(identity)
        self._latest_health_identity = copy.deepcopy(identity)
        return dict(ready=True, identity=identity)

    def reset(self, *, seed=None, metadata=None):
        if self.bridge is None:
            raise PolicyTransportError("Policy connection has not been authorized")
        try:
            self.bridge.reset(seed=seed, metadata=metadata)
        except Exception as exc:
            raise map_error(exc) from exc

    def predict(self, observation, *, headers=None, rpc_observer=None):
        from PIL import Image

        if self._health_identity is None:
            self.health()
        if self.bridge is None:
            raise PolicyTransportError("Policy connection has not been authorized")
        try:
            payload = dict(observation, images={})
            for name in CAMERAS:
                encoded = observation["images"][name]
                with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                    payload["images"][name] = np.asarray(image.convert("RGB")).copy()
            mapped = observation_to_xpolicylab(payload)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise PolicyTransportError(str(exc)) from exc
        if rpc_observer:
            rpc_observer("rpc_start")
        try:
            self._call("update_obs", mapped)
            rows = self._call("get_action")
            try:
                full = self.bridge._call("get_prediction")
            except Exception as exc:
                if (
                    getattr(getattr(exc, "code", None), "value", None) == "invalid_frame"
                    and str(exc) == "no model method named 'get_prediction'"
                ):
                    full = None
                else:
                    raise map_error(exc) from exc
        finally:
            if rpc_observer:
                rpc_observer("rpc_end", transport="websocket")
        try:
            result = reconstruct(
                rows, rows[0].get("mode", self.bridge.mode), mapped["state"]["eef_encoder"], full
            )
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise PolicyTransportError(str(exc)) from exc
        if (
            result["identity"] != self._health_identity
            or result["identity"] != self._latest_health_identity
        ):
            raise ValueError("Policy full identity differs from the startup health identity")
        return result

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
