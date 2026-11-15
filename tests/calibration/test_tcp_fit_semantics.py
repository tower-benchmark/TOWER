# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual pinned fitter with CPU FK and invented HDF5 channels."""

import importlib

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def fitter():
    return importlib.import_module("tower_sim.backend.backend").LulaModel.calibrate_tcp


def synthetic_episode(path, *, position_scale=1.0, rotation_scale=1.0):
    # Poison all unsampled frames. A stride, origin, channel or arm-column change
    # cannot silently keep passing. Leader joints deliberately disagree.
    qpos = np.full((46, 16), np.nan)
    eef = np.full((46, 2, 7), np.nan)
    displacement = np.array([0, 1, 2, 3, 4, 5, 6, 8]) * 0.001 * position_scale
    angles = np.array([0] * 7 + [2.0]) * rotation_scale
    translation = np.array([0.02, -0.03, 0.004])
    relative = Rotation.from_euler("xyz", [20, -30, 40], degrees=True)
    calls = []

    def fk(q):
        assert q.shape == (7,)
        calls.append(q.copy())
        return np.r_[q[:3], Rotation.from_rotvec(q[3:6]).as_quat()]

    for i, (frame, arm) in enumerate((f, a) for f in range(0, 46, 15) for a in range(2)):
        q = np.array([i / 10, -i / 20, 0.1, i / 8, -0.3, 0.5, i + 1])
        qpos[frame, arm * 8 : arm * 8 + 7] = q
        qpos[frame, arm * 8 + 7] = 999  # Grippers are excluded from FK.
        r = Rotation.from_rotvec(q[3:6])
        local = translation + [displacement[i], 0, 0]
        delta = Rotation.from_euler("z", angles[i], degrees=True) * relative
        eef[frame, arm] = np.r_[q[:3] + r.apply(local), (r * delta).as_quat()]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observations/qpos", data=qpos)
        handle.create_dataset("observations/eef", data=eef.reshape(46, 14))
        handle.create_dataset("leader/qpos", data=np.zeros_like(qpos))
    source = {
        "path": str(path),
        "split": "fit",
        "episode_id": "invented",
        "channel_sha256": "c" * 64,
    }
    profile = {"sources": [{"split": "heldout"}, source, {"split": "fit", "path": "must-not-open"}]}
    return fk, profile, calls, qpos, relative


def test_actual_fitter_stride_arms_frames_median_rotation_mean_and_quantiles(tmp_path, fitter):
    fk, profile, calls, qpos, relative = synthetic_episode(tmp_path / "synthetic.hdf5")

    class Model:
        pass

    model = Model()
    model.fk = fk
    fitter(model, profile)
    np.testing.assert_array_equal(calls, qpos[::15].reshape(-1, 8)[:, :7])
    np.testing.assert_allclose(model.tcp_offset, [0.0235, -0.03, 0.004], atol=1e-14, rtol=0)
    # An independent analytic chordal mean: seven 0-degree rotations and one
    # 2-degree rotation about z, followed by the same nontrivial fixed rotation.
    angle = np.arctan2(np.sin(np.deg2rad(2)), 7 + np.cos(np.deg2rad(2)))
    expected = Rotation.from_euler("z", angle) * relative
    np.testing.assert_allclose(
        model.tcp_rotation.as_matrix(), expected.as_matrix(), atol=1e-14, rtol=0
    )
    report = model.tcp_report
    np.testing.assert_allclose(
        report["position_residual_p50_p95_m"], [0.002, 0.00415], atol=1e-14, rtol=0
    )
    mean_deg = np.rad2deg(angle)
    np.testing.assert_allclose(
        report["rotation_residual_p50_p95_deg"],
        [mean_deg, mean_deg + 0.65 * (2 - 2 * mean_deg)],
        atol=1e-12,
        rtol=0,
    )
    assert report["fit_episode_id"] == "invented"
    assert report["source_channel_sha256"] == "c" * 64


@pytest.mark.parametrize("scales", [(2, 1), (1, 4)])
def test_actual_fitter_rejects_large_p95_residuals(tmp_path, fitter, scales):
    fk, profile, *_ = synthetic_episode(
        tmp_path / "synthetic.hdf5", position_scale=scales[0], rotation_scale=scales[1]
    )

    class Model:
        pass

    model = Model()
    model.fk = fk
    with pytest.raises(RuntimeError, match="verified fixed TCP transform"):
        fitter(model, profile)
