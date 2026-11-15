# SPDX-License-Identifier: Apache-2.0
import zipfile

import numpy as np
import pytest
from tower_eval.artifacts import Recorder


def test_append_actions_preserves_request_arrays_and_original_on_duplicate(tmp_path):
    recorder = Recorder(tmp_path, record_video=False)
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    obs = {"state": np.arange(18), "images": {"camera": image}, "prompt": "stack"}
    recorder.request(0, obs)
    path = tmp_path / "requests/000000.npz"
    with zipfile.ZipFile(path) as archive:
        original = {name: archive.getinfo(name).CRC for name in archive.namelist()}
    actions = np.ones((16, 18))
    recorder.request_response(0, obs, {"actions": actions})
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["predicted_actions"], actions)
        np.testing.assert_array_equal(saved["camera"], image)
        np.testing.assert_array_equal(saved["state"], obs["state"])
        assert str(saved["prompt"]) == "stack"
    with zipfile.ZipFile(path) as archive:
        assert {name: archive.getinfo(name).CRC for name in original} == original
    intact = path.read_bytes()
    with pytest.raises(ValueError, match="already recorded"):
        recorder.request_response(0, obs, {"actions": actions})
    assert path.read_bytes() == intact
    recorder.close()
