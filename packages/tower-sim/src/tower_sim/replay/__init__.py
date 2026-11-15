# SPDX-License-Identifier: Apache-2.0
"""Reusable Isaac Sim replay runner for recorded Nero Jenga episodes."""

__all__ = ["ReplayConfig", "default_output_dir", "replay_hdf5"]


def __getattr__(name):
    if name in __all__:
        from importlib import import_module

        return getattr(import_module(".replay_actions", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
