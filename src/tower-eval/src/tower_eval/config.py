# SPDX-License-Identifier: Apache-2.0
"""Typed, strict TOML schemas for the public evaluation entry points."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import _SCHEMAS
from .tasks import resolve

__all__ = ["load_config"]


class _Section(Mapping[str, Any]):
    """Keep mapping access for existing adapter consumers of typed tables."""

    def __getitem__(self, key: str) -> Any:
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        value = getattr(self, key)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return (key for key in self.__dataclass_fields__ if getattr(self, key) is not None)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    __dataclass_fields__: dict[str, Any]
    _provided: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        keys = self if self._provided is None else self._provided
        return {
            key: value.to_dict() if isinstance(value := getattr(self, key), _Section) else value
            for key in keys
        }


def _positive(value: Any, name: str, *, integer: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int if integer else (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        qualifier = "a positive integer" if integer else "finite and positive"
        raise ValueError(f"{name} must be {qualifier}")


def _strings(section: _Section, *names: str) -> None:
    for name in names:
        value = getattr(section, name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string")


def _section(cls, value, label):
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a TOML table")
    unknown = set(value) - {item.name for item in fields(cls)}
    if unknown:
        hint = ""
        if label == "environment" and "max_steps" in unknown:
            hint = "; set the top-level duration_seconds instead of environment.max_steps"
        raise ValueError(f"unknown or unused {label} fields: {sorted(unknown)}; remove them{hint}")
    result = cls(**value)
    object.__setattr__(result, "_provided", tuple(value))
    return result


@dataclass(frozen=True)
class _EnvironmentConfig(_Section):
    backend: str = "cpu"
    task: str = "tower_transfer_2"
    asset_manifest: str | None = None

    def __post_init__(self):
        _strings(self, "backend", "task", "asset_manifest")
        object.__setattr__(self, "task", resolve(self.task).generator_id)


@dataclass(frozen=True)
class _PolicyOptions(_Section):
    # XPolicyLab constructor options read by the CLI.
    action_mode: str = "joint_absolute"
    client_root: str | None = None
    max_connect_attempts: int = 200
    connect_timeout_s: float = 10.0
    max_connect_seconds: float = 180.0
    # Read by LegacyPredictAdapter.from_config, including standalone users.
    health_timeout_s: float = 30.0
    predict_timeout_s: float = 300.0

    def __post_init__(self):
        _strings(self, "action_mode", "client_root")
        _positive(self.max_connect_attempts, "policy.config.max_connect_attempts", integer=True)
        for name in (
            "connect_timeout_s",
            "max_connect_seconds",
            "health_timeout_s",
            "predict_timeout_s",
        ):
            _positive(getattr(self, name), f"policy.config.{name}")


@dataclass(frozen=True)
class _PolicyConfig(_Section):
    adapter: str = "reference"
    server: str | None = None
    url: str | None = None
    entrypoint: str | None = None
    timeout_s: float = 120.0
    config: _PolicyOptions | dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _strings(self, "adapter", "server", "url", "entrypoint")
        _positive(self.timeout_s, "policy.timeout_s")
        object.__setattr__(self, "config", _section(_PolicyOptions, self.config, "policy.config"))


@dataclass(frozen=True)
class EvaluationConfig:
    schema: str = _SCHEMAS["config"]
    seed: int = 0
    episodes: int = 10
    duration_seconds: float = 600.0
    output: Path = Path("outputs/evaluation")
    environment: _EnvironmentConfig | dict[str, Any] = field(default_factory=dict)
    policy: _PolicyConfig | dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != _SCHEMAS["config"]:
            raise ValueError(f"expected schema {_SCHEMAS['config']!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        _positive(self.episodes, "episodes", integer=True)
        _positive(self.duration_seconds, "duration_seconds")
        object.__setattr__(
            self, "environment", _section(_EnvironmentConfig, self.environment, "environment")
        )
        object.__setattr__(self, "policy", _section(_PolicyConfig, self.policy, "policy"))
        if str(self.policy.get("adapter", "")).strip().lower() == "xpolicylab":
            if not str(self.policy.get("server", self.policy.get("url", ""))).strip():
                raise ValueError("XPolicyLab server URL must not be empty")
            if self.environment.get("backend", "cpu") in {
                "cpu",
                "cpu-reference",
                "cpu-smoke",
                "reference",
            }:
                raise ValueError(
                    "policy.adapter=xpolicylab requires encoder observations and an environment "
                    "frame converter; the CPU backend provides neither. Use tower smoke "
                    "--policy xpolicylab://demo_policy for the synthetic protocol fixture."
                )

    def to_dict(self) -> dict[str, Any]:
        """Preserve sparse v1 report/config-hash inputs despite typed defaults."""
        return {
            "schema": self.schema,
            "seed": self.seed,
            "episodes": self.episodes,
            "duration_seconds": self.duration_seconds,
            "output": self.output,
            "environment": self.environment.to_dict()
            if isinstance(self.environment, _Section)
            else self.environment,
            "policy": self.policy.to_dict() if isinstance(self.policy, _Section) else self.policy,
            # Historical report field; input controller tables are rejected.
            "controller": {},
        }


def _set_dotted(document: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"override must use key=value: {expression!r}")
    dotted, raw = expression.split("=", 1)
    value = tomllib.loads(f"value = {raw}")["value"]
    target = document
    parts = dotted.split(".")
    if not all(parts):
        raise ValueError(f"invalid override key: {dotted!r}")
    for part in parts[:-1]:
        nested = target.setdefault(part, {})
        if not isinstance(nested, dict):
            raise ValueError(f"override crosses a scalar: {dotted!r}")
        target = nested
    target[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> EvaluationConfig:
    """Load strict typed tables, resolve task aliases and apply dotted overrides."""
    config_path = Path(path).resolve()
    document = tomllib.loads(config_path.read_text())
    for expression in overrides or []:
        _set_dotted(document, expression)
    if "controller" in document:
        value = document["controller"]
        keys = [f"controller.{key}" for key in value] if isinstance(value, dict) else []
        raise ValueError(
            f"unused [controller] table {keys}: remove the [controller] table; "
            "the CPU runner never read it"
        )
    unknown = set(document) - {item.name for item in fields(EvaluationConfig)}
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    output_value = document.get("output")
    if output_value is None:
        now = datetime.now(UTC)
        output = Path("reports") / now.strftime("%Y_%m_%d") / now.strftime("%Y%m%dT%H%M%SZ")
    else:
        output = Path(output_value)
    if not output.is_absolute():
        output = (config_path.parent / output).resolve()
    if "duration_seconds" in document:
        _positive(document["duration_seconds"], "duration_seconds")
        document["duration_seconds"] = float(document["duration_seconds"])
    document["output"] = output
    return EvaluationConfig(**document)
