# SPDX-License-Identifier: Apache-2.0
"""Compatibility transport for the immutable benchmark-2026.11 /predict API."""

from __future__ import annotations

import base64
import copy
import http.client
import io
import json
import math
from collections.abc import Callable
from typing import Any
from urllib import error, request

import numpy as np

from .adapter import ActionChunk
from .xpolicylab import CAMERAS, decode_chunk


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value.tolist() if hasattr(value, "tolist") else value


class PolicyTransportError(RuntimeError):
    """HTTP/input/connection failure, distinct from invalid model output."""

    def __init__(self, message: str, *, body: Any = None, body_is_json: bool = True) -> None:
        super().__init__(message)
        self.body = body
        self.body_is_json = body_is_json


class PolicyOutputError(RuntimeError):
    """Frozen action rejection: HTTP 422 ``invalid_policy_output`` or the aperture gate."""

    def __init__(self, message: str, *, body: Any = None, body_is_json: bool = True) -> None:
        super().__init__(message)
        self.body = body
        self.body_is_json = body_is_json


class LegacyPredictAdapter:
    """No reset RPC or client seeding: the archived server owns inference seed 42.

    ``predict`` returns the unchanged response for the 2026.11 runtime's existing
    controller. Observation-history payload fields pass through verbatim.
    Startup authorizes only ``expected_identity["package_digest"]``, as the v4
    entry does; other keys are ignored. The formal-batch full-identity
    authorization is not represented. Predictions must equal the complete health
    identity. The transport preserves the existing controller interface.
    """

    def __init__(
        self,
        url: str,
        *,
        expected_identity: dict[str, Any],
        health_timeout_s: float = 30.0,
        predict_timeout_s: float = 300.0,
    ) -> None:
        if not url.strip():
            raise ValueError("Legacy policy server URL must not be empty")
        if not expected_identity.get("package_digest"):
            raise ValueError("Expected identity must include package_digest")
        self.url = url.rstrip("/")
        self.package_digest = expected_identity["package_digest"]
        for name, value in (
            ("health_timeout_s", health_timeout_s),
            ("predict_timeout_s", predict_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        self.health_timeout_s = health_timeout_s
        self.predict_timeout_s = predict_timeout_s
        self._health_identity: dict[str, Any] | None = None
        self._latest_health_identity: dict[str, Any] | None = None

    @classmethod
    def from_config(
        cls, policy: dict[str, Any], *, expected_identity: dict[str, Any]
    ) -> LegacyPredictAdapter:
        """Read the existing policy table; endpoint timeouts live in policy.config."""
        options = policy.get("config", {})
        return cls(
            policy.get("server", policy.get("url", "")),
            expected_identity=expected_identity,
            health_timeout_s=options.get("health_timeout_s", 30.0),
            predict_timeout_s=options.get("predict_timeout_s", 300.0),
        )

    def _request(self, req: Any, *, prediction: bool = False, rpc_observer=None) -> dict[str, Any]:
        status = result = None
        if rpc_observer is not None:
            rpc_observer("rpc_start")

        def update(s, r):
            nonlocal status, result
            status, result = s, r

        try:
            return self._read_request(req, prediction=prediction, observed=update)
        finally:
            if rpc_observer is not None:
                # Same expression as frozen baseline_metrics.http_json: a truthy
                # non-object JSON body raises here, replacing the HTTP error.
                rpc_observer(
                    "rpc_end",
                    http_status=status,
                    server=result.get("profiling") if result else None,
                )

    def _read_request(self, req: Any, *, prediction: bool, observed) -> dict[str, Any]:
        try:
            timeout = self.predict_timeout_s if prediction else self.health_timeout_s
            with request.urlopen(req, timeout=timeout) as response:
                observed(getattr(response, "status", None), None)
                result = json.load(response)
                observed(getattr(response, "status", None), result)
        except error.HTTPError as exc:
            raw = exc.read()
            body_is_json = True
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeError):
                body = {"message": raw.decode("utf-8", errors="replace")}
                body_is_json = False
            observed(exc.code, body if body_is_json else None)
            # baseline_metrics strips instrumentation before worker classification.
            if prediction and isinstance(body, dict):
                body = {key: value for key, value in body.items() if key != "profiling"}
            action_rejection = (
                prediction
                and exc.code == 422
                and isinstance(body, dict)
                and (
                    body.get("code") == "invalid_policy_output"
                    or body == {"code": "invalid_observation", "error": "Negative gripper aperture"}
                )
            )
            kind = PolicyOutputError if action_rejection else PolicyTransportError
            raise kind(
                f"Policy HTTP {exc.code}: {body}", body=body, body_is_json=body_is_json
            ) from exc
        except (error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            raise PolicyTransportError(f"Policy transport failure: {exc}") from exc
        return result

    def health(self) -> dict[str, Any]:
        result = self._request(self.url + "/health")
        identity = result["identity"]
        if not result.get("ready") or identity["package_digest"] != self.package_digest:
            raise ValueError("Legacy policy is not ready for the authorized package")
        # v4 startup and the batch worker each read health. Both authorize only
        # the digest; prediction must then match BOTH complete identity snapshots.
        if self._health_identity is None:
            self._health_identity = copy.deepcopy(identity)
        self._latest_health_identity = copy.deepcopy(identity)
        return result

    def reset(self, *, seed: int | None = None, metadata: dict[str, Any] | None = None) -> None:
        self.health()

    def predict(
        self,
        observation: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        rpc_observer: Callable[..., None] | None = None,
    ) -> dict[str, Any]:
        from PIL import Image

        if self._health_identity is None:
            self.health()
        payload = dict(observation)
        images = dict(payload["images"])
        for name in CAMERAS:
            if isinstance(images[name], str):
                continue  # Already encoded archived PNG; preserve its exact bytes.
            pixels = np.asarray(images[name])
            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
                raise ValueError(f"{name} must be RGB uint8 HxWx3")
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format="PNG")
            images[name] = base64.b64encode(stream.getvalue()).decode("ascii")
        payload.update(images=images, package_digest=self.package_digest)
        req = request.Request(
            self.url + "/predict",
            json.dumps(_jsonable(payload), allow_nan=False).encode(),
            # Frozen METRICS.headers order: Content-Type first, then baseline headers.
            headers={
                "Content-Type": "application/json",
                **{k: v for k, v in (headers or {}).items() if k.lower() != "content-type"},
            },
        )
        result = self._request(req, prediction=True, rpc_observer=rpc_observer)
        # Frozen-equivalent dict equality intentionally treats 1, 1.0 and True
        # as equal. Tightening identity types would change benchmark semantics.
        if (
            result["identity"] != self._health_identity
            or result["identity"] != self._latest_health_identity
        ):
            raise ValueError("Legacy policy full identity differs from the startup health identity")
        return result

    def act(self, observation: Any, info: dict[str, Any] | None = None) -> ActionChunk:
        result = self.predict(observation)
        if result.get("coordinate_frame") != "encoder_arm_base":
            raise ValueError("Legacy policy returned the wrong coordinate frame")
        rows = []
        for index in range(result["execute_steps"]):
            row = {
                key: result[key]
                for key in (
                    "mode",
                    "coordinate_frame",
                    "units",
                    "control_hz",
                    "execute_steps",
                    "j8_actuator_mask",
                    "identity",
                )
            }
            for arm, name in enumerate(("left", "right")):
                row[f"{name}_ee_joint_state"] = [result["gripper_width_m"][index][arm]]
                if result["mode"] == "joint_absolute":
                    row[f"{name}_arm_joint_state"] = result["joint_encoder_targets"][index][arm]
                else:
                    row[f"{name}_ee_transform"] = result["eef_encoder_targets"][index][arm]
                    row["chunk_anchor_encoder"] = result["chunk_anchor_encoder"]
            rows.append(row)
        return decode_chunk(rows, result["mode"], expected_anchor=observation.get("eef_encoder"))

    def close(self) -> None:
        pass
