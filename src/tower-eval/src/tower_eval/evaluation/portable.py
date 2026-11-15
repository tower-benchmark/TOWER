# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the single episode runner."""

from ..runner.episode import run_episode as evaluate_policy

__all__ = ["evaluate_policy"]
