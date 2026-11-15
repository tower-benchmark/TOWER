# SPDX-License-Identifier: Apache-2.0
"""Adapters for policies running in the evaluator process."""

from __future__ import annotations

from typing import Any


class LocalPolicyAdapter:
    """Wrap an in-process policy or an XPolicyLab-style model."""

    def __init__(self, policy: Any, *, action_mode: str = "joint_absolute") -> None:
        self.policy = policy
        self.action_mode = action_mode
        self._model_style = callable(getattr(policy, "update_obs", None)) and callable(
            getattr(policy, "get_action", None)
        )

    def reset(self, *, seed: int | None = None, metadata: dict[str, Any] | None = None) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            if self._model_style:
                prepare = getattr(self.policy, "prepare_case", None)
                if callable(prepare):
                    prepare({**(metadata or {}), "episode_seed": seed})
                reset()
            else:
                reset(seed=seed, metadata=metadata)

    def act(self, observation: Any, info: dict[str, Any] | None = None) -> Any:
        if self._model_style:
            from .xpolicylab import decode_chunk, observation_to_xpolicylab

            mapped = observation_to_xpolicylab(observation)
            anchor = mapped["state"]["eef_encoder"].copy()
            self.policy.update_obs(mapped)
            return decode_chunk(self.policy.get_action(), self.action_mode, expected_anchor=anchor)
        return (
            self.policy(observation, info)
            if callable(self.policy)
            else self.policy.act(observation, info)
        )

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()


class ReferenceAdapter:
    """Deterministic zero-action adapter used for connectivity checks."""

    def __init__(self, action: Any = 0.0) -> None:
        self.action = action
        self.seed: int | None = None

    def reset(self, *, seed: int | None = None, metadata: dict[str, Any] | None = None) -> None:
        self.seed = seed

    def act(self, observation: Any, info: dict[str, Any] | None = None) -> Any:
        return self.action

    def close(self) -> None:
        return None
