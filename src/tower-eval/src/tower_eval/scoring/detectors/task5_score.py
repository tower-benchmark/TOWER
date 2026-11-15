"""Task5 measured insertion gate derived from frozen generation evaluator.
Policy contact/release is inferred from observed hand clearance, never reference timing.
"""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np


def measured_gate(r):
    from ..geometry import inverse, rotate
    from .task5_dependencies.model import Block, Config, EpisodeSpec, Transfer
    from .task5_measured_evaluator import InsertionEvaluator

    bf = {f.name for f in fields(Block)}
    tf = {f.name for f in fields(Transfer)}
    layout = r.spec
    spec = EpisodeSpec(
        layout["logical_id"],
        layout["seed"],
        [Block(**{k: v for k, v in b.items() if k in bf}) for b in layout["blocks"]],
        [Transfer(**{k: v for k, v in t.items() if k in tf}) for t in layout["transfers"]],
        layout["layout_metadata"],
    )
    cfg = Config(task=layout["layout_metadata"]["task_variant"], max_seconds=float(r.times[-1] + 1))
    assert (
        cfg.placement_m,
        cfg.placement_degrees,
        cfg.settle_seconds,
        cfg.final_settle_seconds,
        cfg.collapse_seconds,
    ) == (0.005, 5.0, 0.5, 2.0, 0.25)
    ev = InsertionEvaluator(cfg, spec)
    releases = {}
    for t in layout["transfers"]:
        i = t["block_index"]
        dims = np.array(layout["blocks"][i]["dimensions_m"])
        clear = []
        for hand in (0, 1):
            local = rotate(inverse(r.poses[:, i, 3:]), r.pinch[:, hand] - r.poses[:, i, :3])
            distance = np.linalg.norm(np.maximum(np.abs(local) - dims / 2, 0), axis=1)
            clear.append((r.opening[:, hand] >= 0.075) | (distance >= 0.04))
        releases[i] = clear[0] & clear[1]
    for k, time in enumerate(r.times):
        released = {i for i, m in releases.items() if m[k]}
        active = next((i for i, m in releases.items() if not m[k]), None)
        state = SimpleNamespace(
            time=float(time), blocks=r.poses[k], block_velocities=r.velocities[k]
        )
        result = ev.update(
            state,
            active,
            released,
            controller_done=len(released) == len(layout["transfers"]),
        )
    return dict(
        passed=bool(result.success),
        failure_reason=result.failure_reason,
        **ev.measurements,
    )
