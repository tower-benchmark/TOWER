# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.maintainer.check_tcp_fit import channel_identity, reject_key_changes

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / "benchmarks/2026.11/calibration"


def test_authentic_receipt_matches_migration_and_rejection_contract():
    cache = json.loads((CALIBRATION / "tcp-v2.json").read_text())
    receipt = json.loads((CALIBRATION / "fit-equivalence.json").read_text())
    assert receipt["status"] == "passed"
    assert receipt["key"] == cache["key"]
    assert receipt["lineage"] == cache["lineage"]
    assert receipt["backend_sha256"] == cache["release_provenance"]["backend_sha256"]
    assert receipt["comparison"]["atol"] == 1e-9
    assert receipt["comparison"]["rtol"] == 0
    assert all(0 <= v <= 1e-9 for v in receipt["comparison"]["max_abs_difference"].values())
    assert receipt["raw_fit_data_redistributed"] is False
    assert (
        reject_key_changes(CALIBRATION / "tcp-v2.json", cache["key"])
        == receipt["rejected_key_changes"]
    )


@pytest.mark.parametrize(
    "channel,profile_changes,fit_changes",
    [
        ("leader/qpos", True, False),
        ("observations/qpos", False, True),
        ("observations/eef", True, True),
        ("observations/gripper_status", True, False),
        ("timestamps/row_ros_ns", True, False),
        ("observations/images", False, False),
    ],
)
def test_channel_hash_scopes_include_actual_fit_inputs(
    tmp_path, channel, profile_changes, fit_changes
):
    path = tmp_path / "invented.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs["episode_id"] = "invented"
        for name, shape, dtype in [
            ("leader/qpos", (2, 16), "f8"),
            ("observations/qpos", (2, 16), "f4"),
            ("observations/eef", (2, 14), "f8"),
            ("observations/gripper_status", (2, 2), "f8"),
            ("timestamps/row_ros_ns", (2, 1), "i8"),
            ("observations/images", (2, 2), "u1"),
        ]:
            handle.create_dataset(name, data=np.zeros(shape, dtype=dtype))
    before = channel_identity(path)
    with h5py.File(path, "r+") as handle:
        handle[channel][0, 0] = 1
    after = channel_identity(path)
    assert before["episode_id"] == after["episode_id"] == "invented"
    assert (before["profile_channel_sha256"] != after["profile_channel_sha256"]) == profile_changes
    assert (before["fit_channel_sha256"] != after["fit_channel_sha256"]) == fit_changes
