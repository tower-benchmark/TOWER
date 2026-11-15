from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.maintainer.archive_root import ArchiveUnavailable
from scripts.maintainer.golden_support import GOLDEN, RUNTIME, read
from tests.characterization.golden_compare import assert_equivalent
from tests.characterization.runtime_scenarios import (
    command_queue_stream,
    encoder_observation_output,
    runtime_module,
)


@pytest.mark.archive
def test_command_queue_trace() -> None:
    assert_equivalent(read(GOLDEN / "command-queue.json"), command_queue_stream())


@pytest.mark.archive
def test_encoder_observation_coordinate_mapping() -> None:
    assert_equivalent(read(GOLDEN / "encoder-observation.json"), encoder_observation_output())


@pytest.mark.archive
def test_j2comp_real_cache() -> None:
    asset_root = os.environ.get("TOWER_ARCHIVE_ASSET_ROOT")
    if (
        not asset_root
        or not (Path(asset_root) / "station/station-dual-j2-wrist-shared-v3.usdz").is_file()
    ):
        raise ArchiveUnavailable(
            "SKIP: supply --archive-root and TOWER_ARCHIVE_ASSET_ROOT for the frozen station cache"
        )
    # The frozen module requires the original station bytes. Isolate its imports
    # and asset root so public tests keep using the re-authored distribution.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import j2comp
with open(sys.argv[2]) as handle:
    expected = json.load(handle)
for row in expected:
    actual = j2comp.physical_from_encoder(row["q"], row["width"], row["arm"])
    np.testing.assert_allclose(actual, row["physical"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        j2comp.encoder_from_physical(actual, row["width"], row["arm"]),
        row["roundtrip"], atol=1e-9, rtol=0,
    )
""",
            str(RUNTIME / "ik-capsule/runtime"),
            str(GOLDEN / "j2comp.json"),
        ],
        env={**os.environ, "TOWER_ASSET_ROOT": asset_root},
        check=True,
    )


@pytest.mark.archive
def test_command_bounds_and_observation_tolerance_are_distinct() -> None:
    control = runtime_module("control")
    profile = read(RUNTIME / "shared/shared-ik-spec.json")["limits"]
    lo = np.array(profile["base_soft_cap_urdf"]["lower_rad"]) + np.deg2rad(1.5)
    hi = np.array(profile["base_soft_cap_urdf"]["upper_rad"]) - np.deg2rad(1.5)
    lo[1], hi[1] = max(lo[1], -1.74 + np.deg2rad(5)), min(hi[1], 1.74 - np.deg2rad(5))
    np.testing.assert_allclose(control.GENERATION_LOWER, lo, atol=1e-15, rtol=0)
    np.testing.assert_allclose(control.GENERATION_UPPER, hi, atol=1e-15, rtol=0)
    assert np.deg2rad(0.5) == control.OBSERVED_TRACKING_TOLERANCE_RAD
    assert profile["ik_command"]["inset_deg"] != profile["observed_joint_tolerance"]["deg"]
    q = (lo + hi) / 2
    q[0] = hi[0] + np.deg2rad(0.25)
    with pytest.raises(control.PolicyActionError):
        control.validate_encoder_joints(q)
    control.validate_observed_encoder_joints(q)


@pytest.mark.archive
def test_rejected_ik_does_not_retry_a_reversed_target(monkeypatch: pytest.MonkeyPatch) -> None:
    control = runtime_module("control")
    calls = []
    poses = np.tile(np.eye(4), (1, 2, 1, 1))
    poses[:, :, :3, :3] = np.diag([1.0, -1.0, -1.0])

    def reject(targets: np.ndarray, seed: np.ndarray, **kwargs: object) -> dict:
        calls.append(targets.copy())
        return dict(accepted=False, diagnostics=[dict(reason="reversed target rejected")])

    monkeypatch.setattr(control, "deployed_solve", reject)
    response = dict(
        coordinate_frame="encoder_arm_base",
        j8_actuator_mask=[False, False],
        control_hz=10,
        execute_steps=1,
        mode="eef_delta",
        gripper_width_m=[[0.05, 0.05]],
        eef_encoder_targets=poses,
    )
    seed = np.tile((control.GENERATION_LOWER + control.GENERATION_UPPER) / 2, (2, 1))
    with pytest.raises(control.PolicyActionError, match="IK rejected policy target"):
        control.encoder_endpoints(response, {"joint_encoder": seed}, None, prefix=1)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], poses)
