"""Best-effort annotations, outside authoritative trial commitment."""

import json
from contextlib import suppress

from .resources import save_exclusive


def publish(states, score, output=None):
    try:
        from tower_eval.posture_flags import flags

        result = flags(states, score)
        json.dumps(result, allow_nan=False)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            save_exclusive(output, result)
        return result
    except Exception as error:  # noqa: BLE001 -- optional annotation only; interrupts propagate
        result = dict(
            schema="tower-posture-flags-v1",
            status="not_evaluated",
            operations=[],
            reason=f"Annotation failed: {type(error).__name__}: {error}",
        )
        if output is not None:
            with suppress(Exception):
                output.parent.mkdir(parents=True, exist_ok=True)
                save_exclusive(output, result)
        return result
