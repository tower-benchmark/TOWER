# SPDX-License-Identifier: Apache-2.0
import copy
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from tower_eval.artifacts import sha256
from tower_eval.calibration import (
    cached_tcp_calibration,
    cached_tcp_calibration_v2,
    tcp_calibration_key,
)

from scripts.maintainer.archive_root import ArchivedPath

ROOT = Path(__file__).resolve().parents[2]
V1 = ArchivedPath("runtime/ik-capsule/tcp-cache.json")
V2 = ROOT / "benchmarks/2026.11/calibration/tcp-v2.json"
PROFILE = ROOT / "packages/tower-sim/src/tower_sim/data/motion_profile_v1.json"
FIT_CHANNELS = "54f417513366e0b3f52bbbe08f2cd7a64b797550fd748891d75f5f1ff0af6be1"
KEY_LEAVES = [
    ("fit_algorithm",),
    ("coordinate_convention",),
    ("stage_sha256",),
    ("kinematics_sha256", "nero.urdf"),
    ("kinematics_sha256", "nero.yaml"),
    ("kinematic_geometry_sha256",),
    ("fit_source", "episode_id"),
    ("fit_source", "profile_channel_sha256"),
    ("fit_source", "fit_channel_sha256"),
    ("profile_selection", "rule"),
    ("profile_selection", "profile_channel_scope"),
    ("profile_selection", "fit_channel_scope"),
]


@pytest.fixture
def inputs(tmp_path, monkeypatch, request):
    # The exact reference URDF is already distributed by nero-kinematics. The
    # tiny YAML is recreated using export_kinematics' original serializer.
    shutil.copyfile(
        ROOT / "packages/nero-kinematics/src/nero_kinematics/models/nero_sim_reference.urdf",
        tmp_path / "nero.urdf",
    )
    (tmp_path / "nero.yaml").write_text(
        yaml.safe_dump(
            {
                "api_version": 1.0,
                "cspace": [f"joint{i}" for i in range(1, 8)],
                "root_link": "base_link",
                "default_q": [0, -0.58, 0, 1.95, 0, -0.03, 1.25],
                "cspace_to_urdf_rules": [],
                "collision_spheres": [],
            }
        )
    )
    if request.node.get_closest_marker("archive"):
        record = json.loads(V1.read_text())
    else:
        public = json.loads(V2.read_text())
        record = dict(
            kinematics_sha256=public["key"]["kinematics_sha256"],
            usd_sha256=public["key"]["stage_sha256"],
            tcp=public["tcp"],
        )
    for name, digest in record["kinematics_sha256"].items():
        assert sha256(tmp_path / name) == digest
    backend = importlib.import_module("tower_sim.backend.backend")
    profile = json.loads(PROFILE.read_text())
    key = tcp_calibration_key(
        profile=profile,
        stage_sha256=record["usd_sha256"],
        kinematics=tmp_path,
        fit_channel_sha256=FIT_CHANNELS,
    )
    return backend.LulaModel, profile, key, record


@pytest.mark.archive
def test_frozen_capsule_rejects_in_tree_alias_after_backend_move(inputs, tmp_path):
    model, _, _, record = inputs
    original = model.calibrate_tcp
    backend = ROOT / "packages/tower-sim/src/tower_sim/backend/backend.py"
    assert sha256(backend) != record["backend_sha256"]
    with (
        pytest.raises(ValueError, match="backend"),
        cached_tcp_calibration(model, V1, record["usd_sha256"], PROFILE, tmp_path, backend),
    ):
        pytest.fail(
            "Frozen backend pin must reject the alias; reference reproduction uses the private archive"
        )
    assert model.calibrate_tcp is original


@pytest.mark.archive
@pytest.mark.parametrize("context", [cached_tcp_calibration, cached_tcp_calibration_v2])
def test_v2_resource_is_exact_v1_migration_and_restores_model(inputs, context):
    model, profile, key, record = inputs
    v2 = json.loads(V2.read_text())
    assert v2["tcp"] == record["tcp"]
    assert v2["lineage"] == {
        "path": "simulator/benchmark-202611/runtime/ik-capsule/tcp-cache.json",
        "sha256": sha256(V1),
    }
    assert v2["key"] == key
    original = model.calibrate_tcp
    with context(model, V2, key=key):
        a, b = model.__new__(model), model.__new__(model)
        a.calibrate_tcp(profile)
        a.tcp_offset[:] = 100
        a.tcp_report["translation_m"][:] = [100] * 3
        b.calibrate_tcp(profile)
        assert b.tcp_report == record["tcp"]
        np.testing.assert_array_equal(b.tcp_offset, record["tcp"]["translation_m"])
    assert model.calibrate_tcp is original


@pytest.mark.parametrize("path", KEY_LEAVES, ids=[".".join(p) for p in KEY_LEAVES])
@pytest.mark.parametrize("side", ["expected", "cache"])
def test_rejects_every_semantic_key_component(inputs, tmp_path, path, side):
    model, _, key, _ = inputs
    cache = json.loads(V2.read_text())
    node = key if side == "expected" else cache["key"]
    for name in path[:-1]:
        node = node[name]
    old = node[path[-1]]
    node[path[-1]] = ("0" if old[0] != "0" else "1") + old[1:]
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache))
    original = model.calibrate_tcp
    with pytest.raises(ValueError), cached_tcp_calibration(model, cache_path, key=key):
        pytest.fail("Changed semantic identity accepted")
    assert model.calibrate_tcp is original


@pytest.mark.parametrize("name", ["nero.urdf", "nero.yaml"])
def test_key_builder_hashes_actual_model_bytes(inputs, tmp_path, name):
    model, profile, key, _ = inputs
    with (tmp_path / name).open("a") as stream:
        stream.write("\n")
    changed = tcp_calibration_key(
        profile=profile,
        stage_sha256=key["stage_sha256"],
        kinematics=tmp_path,
        fit_channel_sha256=FIT_CHANNELS,
    )
    with (
        pytest.raises(ValueError, match="kinematics_sha256"),
        cached_tcp_calibration_v2(model, V2, key=changed),
    ):
        pass


def test_moved_backend_bytes_and_module_aliases_do_not_affect_v2(inputs, tmp_path, monkeypatch):
    _, profile, key, record = inputs
    path = tmp_path / "renamed_implementation.py"
    path.write_text(
        "class Model:\n    def calibrate_tcp(self, profile):\n        raise AssertionError('raw fit')\n"
    )
    spec = importlib.util.spec_from_file_location("new_runtime.renamed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "new_runtime.renamed", module)
    monkeypatch.setitem(sys.modules, "old_runtime.backend", module)
    actual_class = module.Model
    original = actual_class.calibrate_tcp
    with cached_tcp_calibration(actual_class, V2, key=key):
        assert sys.modules["old_runtime.backend"].Model is actual_class
        # Simulate a relocation and a harmless release-code change after import.
        moved = path.with_name("elsewhere.py")
        path.rename(moved)
        moved.write_text("# Changed unrelated release bytes\n")
        module.__file__ = str(moved)
        instance = sys.modules["old_runtime.backend"].Model()
        instance.calibrate_tcp(profile)
        assert instance.tcp_report == record["tcp"]
    assert actual_class.calibrate_tcp is original


def test_release_provenance_is_not_a_validity_key(inputs, tmp_path):
    model, profile, key, record = inputs
    cache = json.loads(V2.read_text())
    cache["release_provenance"] = {"commit": "different", "wheel_sha256": "0" * 64}
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(cache))
    with cached_tcp_calibration_v2(model, path, key=key):
        instance = model.__new__(model)
        instance.calibrate_tcp(profile)
        assert instance.tcp_report == record["tcp"]


def test_profile_only_binds_fit_selection_not_unrelated_metadata(inputs, tmp_path):
    model, profile, key, _ = inputs
    relocated = copy.deepcopy(profile)
    relocated["sources"][0]["path"] = "unavailable-and-unneeded.hdf5"
    relocated["fps"] = 999
    relocated["primitives"] = []
    assert (
        tcp_calibration_key(
            profile=relocated,
            stage_sha256=key["stage_sha256"],
            kinematics=tmp_path,
            fit_channel_sha256=FIT_CHANNELS,
        )
        == key
    )
    original = model.calibrate_tcp
    with (
        pytest.raises(ValueError, match="training source"),
        cached_tcp_calibration_v2(model, V2, key=key),
    ):
        selected = next(s for s in relocated["sources"] if s["split"] == "fit")
        selected["episode_id"] = "changed"
        model.__new__(model).calibrate_tcp(relocated)
    assert model.calibrate_tcp is original


def test_nested_context_and_exception_restore_exact_class_method(inputs):
    model, _, key, _ = inputs
    original = model.calibrate_tcp
    with (
        pytest.raises(RuntimeError, match="constructor failed"),
        cached_tcp_calibration_v2(model, V2, key=key),
    ):
        outer = model.calibrate_tcp
        with cached_tcp_calibration_v2(model, V2, key=key):
            assert model.calibrate_tcp is not outer
        assert model.calibrate_tcp is outer
        raise RuntimeError("constructor failed")
    assert model.calibrate_tcp is original


@pytest.mark.parametrize(
    "field,value",
    [
        ("translation_m", [float("nan"), 0, 0]),
        ("rotation_xyzw", [0, 0, 0, 0]),
        ("translation_m", [0, 0]),
        ("position_residual_p50_p95_m", [0, 0.006]),
        ("rotation_residual_p50_p95_deg", [0, 3.1]),
        ("position_residual_p50_p95_m", [0.002, 0.001]),
        ("rotation_residual_p50_p95_deg", [-0.1, 0]),
        ("fit_episode_id", "changed"),
        ("source_channel_sha256", "0" * 64),
    ],
)
def test_rejects_invalid_or_inconsistent_report(inputs, tmp_path, field, value):
    model, _, key, _ = inputs
    cache = json.loads(V2.read_text())
    cache["tcp"][field] = value
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(cache))
    original = model.calibrate_tcp
    with pytest.raises(ValueError), cached_tcp_calibration_v2(model, path, key=key):
        pass
    assert model.calibrate_tcp is original


@pytest.mark.archive
def test_schema_dispatch_errors_and_no_cache(inputs, tmp_path):
    model, _, key, _ = inputs
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"schema": "tower-recorded-tcp-v999"}))
    with (
        pytest.raises(ValueError, match="Unsupported TCP calibration cache schema.*v999"),
        cached_tcp_calibration(model, path, key=key),
    ):
        pass
    with (
        pytest.raises(ValueError, match="requires.*semantic key"),
        cached_tcp_calibration(model, V2),
    ):
        pass
    with (
        pytest.raises(ValueError, match="Expected tower-recorded-tcp-v2"),
        cached_tcp_calibration_v2(model, V1, key=key),
    ):
        pass
    original = model.calibrate_tcp
    with cached_tcp_calibration(model, None):
        assert model.calibrate_tcp is original


@pytest.mark.parametrize("context", [cached_tcp_calibration, cached_tcp_calibration_v2])
def test_v2_rechecks_kinematics_written_during_construction(inputs, tmp_path, context):
    model, profile, key, _ = inputs
    original = model.calibrate_tcp
    with context(model, V2, key=key, kinematics=tmp_path):
        model.__new__(model).calibrate_tcp(profile)
        # The model exports a different chain after the key was built.
        (tmp_path / "nero.urdf").write_text("<robot name='other'/>")
        with pytest.raises(ValueError, match="kinematics mismatch: nero.urdf"):
            model.__new__(model).calibrate_tcp(profile)
    assert model.calibrate_tcp is original


@pytest.mark.parametrize(
    "cache,kwargs,message",
    [
        (V2, {"stage_hash": "0" * 64}, "pass key= only"),
        (V2, {"backend_path": "backend.py"}, "pass key= only"),
        pytest.param(V1, {"key": {}}, "do not take a semantic key", marks=pytest.mark.archive),
    ],
)
def test_dispatcher_rejects_mixed_schema_arguments(inputs, cache, kwargs, message):
    model, _, key, _ = inputs
    kwargs = {"key": key, **kwargs} if cache == V2 else kwargs
    with pytest.raises(ValueError, match=message), cached_tcp_calibration(model, cache, **kwargs):
        pytest.fail("Mixed schema arguments accepted")


def test_semantic_key_is_identical_after_resource_relocation(inputs, tmp_path):
    _, profile, key, _ = inputs
    relocated = tmp_path / "different-checkout" / "kinematics"
    relocated.mkdir(parents=True)
    for name in ("nero.urdf", "nero.yaml"):
        shutil.copyfile(tmp_path / name, relocated / name)
    moved_profile = copy.deepcopy(profile)
    for source in moved_profile["sources"]:
        source["path"] = str(tmp_path / "elsewhere" / source["episode_id"])
    rebuilt = tcp_calibration_key(
        profile=moved_profile,
        stage_sha256=key["stage_sha256"],
        kinematics=relocated,
        fit_channel_sha256=FIT_CHANNELS,
    )
    assert rebuilt == key
    encoded = json.dumps(rebuilt)
    assert str(tmp_path) not in encoded
    assert "src/tower-datagenerator" not in encoded
    assert "tower_sim" not in encoded
