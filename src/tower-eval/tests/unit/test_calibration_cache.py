# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest
from tower_eval.artifacts import sha256
from tower_eval.calibration import cached_tcp_calibration


def fixture(tmp_path):
    for name in ["profile", "backend", "nero.urdf", "nero.yaml"]:
        (tmp_path / name).write_text(name)
    record = {
        "schema": "tower-recorded-tcp-v1",
        "usd_sha256": "stage",
        "profile_sha256": sha256(tmp_path / "profile"),
        "backend_sha256": sha256(tmp_path / "backend"),
        "kinematics_sha256": {name: sha256(tmp_path / name) for name in ["nero.urdf", "nero.yaml"]},
        "tcp": {
            "translation_m": [0.0, 0.01, -0.03],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "position_residual_p50_p95_m": [0.0, 0.001],
            "rotation_residual_p50_p95_deg": [0.0, 0.2],
            "fit_episode_id": "episode",
            "source_channel_sha256": "channels",
        },
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(record))

    class Model:
        def calibrate_tcp(self, profile):
            raise AssertionError("Raw source must not be accessed")

    profile = {"sources": [{"split": "fit", "episode_id": "episode", "channel_sha256": "channels"}]}
    return Model, path, record, profile


def test_exact_cached_fit_and_method_restoration(tmp_path):
    model, path, record, profile = fixture(tmp_path)
    original = model.calibrate_tcp
    with cached_tcp_calibration(
        model, path, "stage", tmp_path / "profile", tmp_path, tmp_path / "backend"
    ):
        instance = model()
        instance.calibrate_tcp(profile)
        np.testing.assert_array_equal(instance.tcp_offset, record["tcp"]["translation_m"])
        assert instance.tcp_report == record["tcp"]
    assert model.calibrate_tcp is original


@pytest.mark.parametrize("kind", ["usd", "profile", "backend", "kinematics", "source", "nonfinite"])
def test_cache_rejects_incompatible_or_invalid_fit(tmp_path, kind):
    model, path, record, profile = fixture(tmp_path)
    original = model.calibrate_tcp
    stage = "wrong" if kind == "usd" else "stage"
    if kind in ["profile", "backend"]:
        (tmp_path / kind).write_text("changed")
    if kind == "kinematics":
        (tmp_path / "nero.urdf").write_text("changed")
    if kind == "source":
        profile["sources"][0]["channel_sha256"] = "changed"
    if kind == "nonfinite":
        record["tcp"]["translation_m"][0] = float("nan")
        path.write_text(json.dumps(record))
    with (
        pytest.raises(ValueError),
        cached_tcp_calibration(
            model, path, stage, tmp_path / "profile", tmp_path, tmp_path / "backend"
        ),
    ):
        model().calibrate_tcp(profile)
    assert model.calibrate_tcp is original
