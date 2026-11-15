# SPDX-License-Identifier: Apache-2.0
"""TOWER's stable public API, resolved on demand without loading a simulator."""

# Preserve the established policy import closure; public attribute lookup below
# does not eagerly import any additional API implementation.
from typing import TYPE_CHECKING as _TYPE_CHECKING

from .policies import adapter as _adapter

if _TYPE_CHECKING:
    from .config import load_config as load_config
    from .controllers import DualModeController as DualModeController
    from .controllers import GripperSpec as GripperSpec
    from .controllers import RobotSpec as RobotSpec
    from .envs import TowerEnv as TowerEnv
    from .policies.adapter import ActionChunk as ActionChunk
    from .policies.adapter import PolicyAdapter as PolicyAdapter
    from .scoring import BenchmarkMetrics as BenchmarkMetrics
    from .scoring import EpisodeRecord as EpisodeRecord
    from .scoring import StackScorer as StackScorer
    from .scoring import score_run as score_run
    from .simulator import MeasuredState as MeasuredState
    from .simulator import SimulatorBackend as SimulatorBackend

    __version__: str

__all__ = [
    "PolicyAdapter",
    "ActionChunk",
    "TowerEnv",
    "SimulatorBackend",
    "MeasuredState",
    "RobotSpec",
    "GripperSpec",
    "DualModeController",
    "load_config",
    "score_run",
    "StackScorer",
    "BenchmarkMetrics",
    "EpisodeRecord",
    "__version__",
]

_PUBLIC = {
    "TowerEnv": "envs",
    "SimulatorBackend": "simulator",
    "MeasuredState": "simulator",
    "RobotSpec": "controllers",
    "GripperSpec": "controllers",
    "DualModeController": "controllers",
    "load_config": "config",
    "score_run": "scoring",
    "StackScorer": "scoring",
    "BenchmarkMetrics": "scoring",
    "EpisodeRecord": "scoring",
}


def __getattr__(name):
    if name in {"ActionChunk", "PolicyAdapter"}:
        return getattr(_adapter, name)
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("tower-eval")
        except PackageNotFoundError:
            # An uninstalled checkout still has a single version source.
            import tomllib
            from pathlib import Path

            project = Path(__file__).resolve().parents[2] / "pyproject.toml"
            try:
                metadata = tomllib.loads(project.read_text())["project"]
            except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
                return "0+unknown"
            if metadata.get("name") != "tower-eval":
                return "0+unknown"
            return metadata["version"]
    if name in _PUBLIC:
        from importlib import import_module

        return getattr(import_module(f".{_PUBLIC[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
