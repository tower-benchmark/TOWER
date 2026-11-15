"""Per-capture terminal guard with an explicit reference-environment context."""

import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class TerminalGuard:
    def __init__(self, original, env, output):
        self.original, self.env, self.output = original, env, Path(output)
        self.enabled = False
        self.reference_index = None
        self.rejections = {}
        self.consumed = set()

    def check(self, blocks, offset_m, camera_position_mm, camera_xyzw, intrinsic, near_mm, label):
        try:
            return self.original(
                blocks, offset_m, camera_position_mm, camera_xyzw, intrinsic, near_mm, label
            )
        except RuntimeError as exc:
            message = str(exc)
            match = re.fullmatch(
                r"Wrist near plane may clip a task block: (.+) \(([-+\d.eE]+) < ([-+\d.eE]+) mm\)",
                message,
            )
            if not self.enabled or not match or near_mm != 10.0:
                raise
            tick = round(float(self.env.elapsed[0]) * 400)
            if tick <= 0:
                raise
            indexed = re.fullmatch(r"(?:v3_probe|live)_env_(\d+):(?:left|back)_wrist", label)
            if indexed:
                index = int(indexed[1])
            elif label in ("native61_v2_2:left_wrist", "native61_v2_2:back_wrist"):
                index = 0 if self.reference_index is None else self.reference_index
            else:
                raise
            if not 0 <= index < self.env.n:
                raise
            if index not in self.rejections:
                row = dict(
                    schema="tower-camera-near-plane-terminal-v1",
                    environment_index=index,
                    failure_tick=tick,
                    time_s=tick / 400,
                    category="camera_near_plane_guard",
                    reason=message,
                    near_mm=near_mm,
                    label=label,
                    original_guard_rejected=True,
                    policy_observation_accepted=False,
                    renderer_parameters_changed=False,
                    terminal_image_purpose="diagnostic evidence only; never sent to policy",
                    blocks=np.asarray(blocks).tolist(),
                    offset_m=np.asarray(offset_m).tolist(),
                    camera_position_mm=np.asarray(camera_position_mm).tolist(),
                    camera_xyzw=np.asarray(camera_xyzw).tolist(),
                    intrinsic=np.asarray(intrinsic).tolist(),
                )
                self.output.mkdir(exist_ok=True)
                path = self.output / f"env-{index}-tick-{tick}.json"
                with path.open("x") as stream:
                    json.dump(row, stream, indent=2, allow_nan=False)
                row["evidence_path"] = str(path)
                self.rejections[index] = row
            # Preserve the rejected minimum in camera diagnostics. Never label
            # it a passed guard. The worker consumes the rejection immediately.
            return float(match[2])

    def consume(self, tick):
        pending = {i: row for i, row in self.rejections.items() if i not in self.consumed}
        assert all(row["failure_tick"] == tick for row in pending.values()), (
            "Unconsumed camera rejection"
        )
        self.consumed.update(pending)
        return pending

    @contextmanager
    def reference_environment(self, index):
        previous = self.reference_index
        self.reference_index = index
        try:
            yield
        finally:
            self.reference_index = previous
