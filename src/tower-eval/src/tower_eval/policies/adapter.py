# SPDX-License-Identifier: Apache-2.0
"""Stable policy interface shared by local and remote policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..constants import _SCHEMAS, ACTION_DT


class PolicyAdapter(Protocol):
    """Minimal policy lifecycle used by the evaluator and server template."""

    def reset(self, *, seed: int | None = None, metadata: dict[str, Any] | None = None) -> None:
        """Reset recurrent state before an episode."""

    def act(self, observation: Any, info: dict[str, Any] | None = None) -> Any:
        """Return one action valid for the evaluator action space."""

    def close(self) -> None:
        """Release model and transport resources."""


@dataclass(frozen=True)
class ActionChunk:
    """One or more actions returned by a policy with an explicit period."""

    actions: Any
    dt_s: float = ACTION_DT
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import math

        if not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        values = self.actions
        if hasattr(values, "__array__"):
            import numpy as np

            array = np.asarray(values)
            if array.ndim not in (1, 2) or not np.isfinite(array).all():
                raise ValueError("actions must be a finite one- or two-dimensional array")


def validate_action_frame(value: Any, accepted_frame: str = "physical") -> None:
    """Reject frame tags before normalization can discard their metadata.

    Untagged historical actions retain the backend's existing convention. This
    is validation only: no coordinate or J2 conversion is performed here.
    """
    if isinstance(value, ActionChunk) or (hasattr(value, "actions") and hasattr(value, "metadata")):
        validate_action_frame(
            {"coordinate_frame": value.metadata.get("coordinate_frame")}, accepted_frame
        )
        validate_action_frame(value.actions, accepted_frame)
    elif isinstance(value, dict):
        frame = value.get("coordinate_frame")
        if frame is not None and frame != accepted_frame:
            raise ValueError(
                f"Action coordinate_frame={frame!r} is not accepted by this backend "
                f"({accepted_frame!r}); an explicit environment frame converter is required"
            )
        if value.get("protocol") == _SCHEMAS["action_chunk"]:
            validate_action_frame(
                ActionChunk(value["actions"], value["dt_s"], value.get("metadata", {})),
                accepted_frame,
            )
    elif isinstance(value, (list, tuple)):
        for row in value:
            validate_action_frame(row, accepted_frame)


def normalize_actions(
    value: Any, env_dt: float | None = None, *, accepted_frame: str = "physical"
) -> list[Any]:
    """Normalize one action or an ActionChunk to the complete action sequence."""
    import numpy as np

    validate_action_frame(value, accepted_frame)
    if isinstance(value, dict) and value.get("protocol") == _SCHEMAS["action_chunk"]:
        value = ActionChunk(value["actions"], value["dt_s"], value.get("metadata", {}))
    if isinstance(value, ActionChunk):
        if env_dt is not None and not np.isclose(value.dt_s, env_dt, rtol=0, atol=1e-10):
            raise ValueError(
                f"action chunk dt_s={value.dt_s} does not match environment dt={env_dt}"
            )
        value = value.actions
    elif hasattr(value, "actions") and not isinstance(value, (np.ndarray, list, tuple, dict)):
        value = value.actions
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)) and any(isinstance(x, dict) for x in value):
        if not all(isinstance(x, dict) for x in value):
            raise ValueError("Mixed tagged/untagged chunks require explicit tags on every action")
        return list(value)
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError("policy returned an empty action chunk")
    return list(array) if array.ndim == 2 else [value]
