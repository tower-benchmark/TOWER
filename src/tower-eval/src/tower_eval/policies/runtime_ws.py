"""Benchmark response contract over XPolicyLab's WebSocket transport.

Deploy one runtime process per policy server. INFER is atomic, but a server's
optional prediction cache is shared between clients.
"""

from __future__ import annotations

import base64
import copy
import io
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .xpolicylab import CAMERAS, XPolicyLabAdapter, observation_to_xpolicylab

SHARED_PREDICTION = (
    "policy server returned a prediction for a different request; is it shared by another client?"
)


class _PolicyError(RuntimeError):
    def __init__(self, message: str, *, body: Any = None, body_is_json: bool = True):
        super().__init__(message)
        self.body, self.body_is_json = body, body_is_json


class PolicyTransportError(_PolicyError):
    """Connection, protocol, structural response, or policy input failure."""


class PolicyOutputError(_PolicyError):
    """Invalid policy output or the frozen aperture observation rejection."""


def validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise ValueError("policy_url must use ws:// or wss:// with a hostname")


def transport_error(exc: Exception, request_id=None) -> PolicyTransportError:
    return PolicyTransportError(
        str(exc),
        body={
            "code": getattr(getattr(exc, "code", None), "value", "transport_error"),
            "message": str(exc),
            "exception_type": getattr(exc, "details", {}).get("exception_type", type(exc).__name__),
            "request_id": request_id,
        },
    )


def map_error(exc: Exception, request_id=None) -> PolicyTransportError | PolicyOutputError:
    code = getattr(getattr(exc, "code", None), "value", None)
    category = getattr(exc, "details", {}).get("exception_type")
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
    return transport_error(exc, request_id)


def normalize(value: Any) -> Any:
    """Retain scalar types and float64 bits; never silently widen floating arrays."""
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind == "f" and value.dtype != np.dtype("float64"):
            raise ValueError("Policy floating arrays must have float64 dtype")
        return normalize(value.tolist())
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def numeric_equal(a, b):
    if isinstance(a, list) or isinstance(b, list):
        return (
            isinstance(a, list)
            and isinstance(b, list)
            and len(a) == len(b)
            and all(numeric_equal(x, y) for x, y in zip(a, b))
        )
    return np.float64(a).tobytes() == np.float64(b).tobytes()


def reconstruct(rows, mode, full):
    """Check wire structure and audit consistency; control owns action semantics."""
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Policy rows must be a nonempty list of mappings")
    rows = normalize(rows)
    full = normalize(full)
    if full is not None and not isinstance(full, dict):
        raise ValueError("Policy prediction must be a mapping")
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
    is_eef = any("left_ee_transform" in row or row.get("mode") == "eef_delta" for row in rows)
    if is_eef:
        for row in rows:
            if "mode" not in row or "coordinate_frame" not in row:
                raise ValueError("EEF targets require an explicit encoder frame and mode")
        for key in ("control_hz", "execute_steps", "j8_actuator_mask"):
            if key not in result:
                raise ValueError(f"EEF targets require explicit {key}")
    else:
        result.setdefault("mode", mode)
        result.setdefault("coordinate_frame", "encoder_arm_base")
        result.setdefault("control_hz", 10)
        result.setdefault("execute_steps", len(rows))
        result.setdefault("j8_actuator_mask", [False, False])
    if "identity" not in result:
        raise ValueError("Policy response is missing identity")
    result["gripper_width_m"] = [
        [r[f"{arm}_ee_joint_state"][0] for arm in ("left", "right")] for r in rows
    ]
    field = "joint_encoder_targets" if result["mode"] == "joint_absolute" else "eef_encoder_targets"
    suffix = "arm_joint_state" if result["mode"] == "joint_absolute" else "ee_transform"
    result[field] = [[r[f"{arm}_{suffix}"] for arm in ("left", "right")] for r in rows]
    if all("native_action" in r for r in rows):
        result["native_actions"] = [r["native_action"] for r in rows]
    arrays = {field, "gripper_width_m", "native_actions", "chunk_anchor_encoder"}
    for row in rows:
        if row.get("identity") != result["identity"]:
            raise ValueError("Endpoint identities differ within the chunk")
    if full is None:
        result["missing_evidence"] = ["full_prediction", "unexecuted_horizon", "inference_seconds"]
        result["missing_evidence"].extend(
            k for k in ("native_actions", "applied_j8", "units") if k not in result
        )
        return result
    for key, value in result.items():
        expected = full[key][: len(rows)] if key in arrays - {"chunk_anchor_encoder"} else full[key]
        equal = numeric_equal(value, expected) if key in arrays else value == expected
        if not equal:
            raise ValueError(SHARED_PREDICTION)
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
                raise ValueError(SHARED_PREDICTION)
    response = copy.deepcopy(full)
    for key in (arrays - {"chunk_anchor_encoder"}) & result.keys():
        response[key] = result[key] + full[key][len(rows) :]
    return response


class RuntimePolicyAdapter:
    def __init__(self, url: str, *, expected_identity: dict[str, Any], bridge=None, root=None):
        validate_url(url)
        if not expected_identity.get("package_digest"):
            raise ValueError("Expected identity must include package_digest")
        self.url, self.package_digest = url, expected_identity["package_digest"]
        self.bridge, self.root = bridge, root
        self._health_identity = self._latest_health_identity = None
        self.cleanup_errors = []

    def health(self):
        try:
            if self.bridge is None:
                self.bridge = XPolicyLabAdapter(self.url, root=self.root, timeout_s=300.0)
            identity = normalize(self.bridge._call("get_identity"))
        except Exception as exc:
            raise transport_error(exc) from exc
        if identity["package_digest"] != self.package_digest:
            raise ValueError("Policy is not ready for the authorized package")
        if self._health_identity is None:
            self._health_identity = copy.deepcopy(identity)
        self._latest_health_identity = copy.deepcopy(identity)
        return dict(ready=True, identity=identity)

    def reset(self, *, seed=None, metadata=None):
        try:
            if self.bridge is None:
                raise ValueError("Policy connection has not been authorized")
            self.bridge.reset(seed=seed, metadata=metadata)
        except Exception as exc:
            raise transport_error(exc) from exc

    def predict(self, observation, *, headers=None, rpc_observer=None):
        from PIL import Image

        request_id = (headers or {}).get("X-Baseline-Request-ID")
        if self._health_identity is None:
            self.health()
        try:
            if self.bridge is None:
                raise ValueError("Policy connection has not been authorized")
            payload = dict(observation, images={})
            for name in CAMERAS:
                with Image.open(
                    io.BytesIO(base64.b64decode(observation["images"][name], validate=True))
                ) as image:
                    if image.format != "PNG":
                        raise ValueError("Policy archive image must be PNG")
                    payload["images"][name] = np.asarray(image.convert("RGB")).copy()
            mapped = observation_to_xpolicylab(payload)
            if rpc_observer:
                rpc_observer("rpc_start")
            try:
                try:
                    rows = self.bridge.infer(mapped)
                except Exception as exc:
                    raise map_error(exc, request_id) from exc
                try:
                    full = self.bridge._call("get_prediction", int(observation.get("env_idx", 0)))
                except Exception as exc:
                    if getattr(getattr(exc, "code", None), "value", None) == "invalid_frame":
                        full = None
                    elif getattr(exc, "details", {}).get("exception_type") == "KeyError":
                        raise ValueError(
                            f"missing prediction for env {observation.get('env_idx', 0)} "
                            "(shared server or policy bug)"
                        ) from exc
                    else:
                        raise transport_error(exc, request_id) from exc
            finally:
                if rpc_observer:
                    rpc_observer("rpc_end", transport="websocket")
            result = reconstruct(rows, self.bridge.mode, full)
            if (
                result["identity"] != self._health_identity
                or result["identity"] != self._latest_health_identity
            ):
                raise ValueError("Policy full identity differs from the startup health identity")
            return result
        except (PolicyOutputError, PolicyTransportError):
            raise
        except Exception as exc:
            raise transport_error(exc, request_id) from exc

    def close(self):
        if self.bridge is not None:
            try:
                self.bridge.close()  # closes the socket even if trial_end fails
            except Exception as exc:  # noqa: BLE001 -- cleanup must preserve recorded outcomes
                self.cleanup_errors.append(transport_error(exc).body)
