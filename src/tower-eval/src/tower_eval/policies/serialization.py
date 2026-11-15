# SPDX-License-Identifier: Apache-2.0
"""JSON-compatible values for persisted action chunks."""

from typing import Any

from ..constants import _SCHEMAS
from .adapter import ActionChunk


def jsonable(value: Any) -> Any:
    """Convert common array-like values without requiring NumPy."""
    if isinstance(value, ActionChunk):
        return dict(
            protocol=_SCHEMAS["action_chunk"],
            actions=jsonable(value.actions),
            dt_s=value.dt_s,
            metadata=jsonable(value.metadata),
        )
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
