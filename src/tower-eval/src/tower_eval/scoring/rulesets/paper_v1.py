# SPDX-License-Identifier: Apache-2.0
"""Current package semantics, including historical Task5 rejection."""

from dataclasses import dataclass

from ..rules import Rules as BaseRules


@dataclass(frozen=True)
class Rules(BaseRules):
    """Paper-v1 thresholds and detector choices."""


DEFAULT_RULES = Rules()
