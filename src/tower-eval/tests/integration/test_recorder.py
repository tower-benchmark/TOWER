# SPDX-License-Identifier: Apache-2.0
"""Shared recording contracts retained after the Isaac 5.1 rollout retirement."""

import av
import h5py
import numpy as np
import pytest
from tower_eval.artifacts import Recorder
from tower_eval_fakes import FakeSim


@pytest.mark.parametrize("diagnostics", [False, True])
def test_recorder_preserves_optional_diagnostics(tmp_path, diagnostics):
    sim = FakeSim(diagnostics=diagnostics)
    recorder = Recorder(tmp_path, record_video=False)
    samples = [sim.read()]
    try:
        recorder.state(samples[0])
        for _ in range(3):
            samples.append(sim.step(np.full(16, 0.01)))
            recorder.state(samples[-1])
    finally:
        recorder.close()
    with h5py.File(tmp_path / "states.hdf5") as recorded:
        np.testing.assert_allclose(recorded["time"][:], [0, 1 / 30, 2 / 30, 0.1])
        np.testing.assert_array_equal(recorded["qpos"][:], [sample.qpos for sample in samples])
        for field in ("effort", "blocks", "block_velocities", "eef_world"):
            if diagnostics:
                np.testing.assert_array_equal(
                    recorded[field][:], [getattr(sample, field) for sample in samples]
                )
            else:
                assert field not in recorded


@pytest.mark.parametrize("fps", [10, 20, 30])
def test_recorder_encodes_requested_video_rate_and_frame_count(tmp_path, fps):
    recorder = Recorder(tmp_path, video_fps=fps)
    frames = fps + 1
    try:
        for index in range(frames):
            image = np.full((16, 16, 3), index, np.uint8)
            recorder.frame({"camera": image}, index / fps)
        assert recorder.frame_count == frames
    finally:
        recorder.close()
    with av.open(str(tmp_path / "camera.mp4")) as video:
        assert video.streams.video[0].average_rate == fps
        assert len(list(video.decode(video=0))) == frames
