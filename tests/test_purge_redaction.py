"""Redacted metadata can prove identity but cannot supply runtime resources."""

import copy
import json
import sys
from types import SimpleNamespace as NS

import pytest
from tower_sim.backend.resources import DATA
from tower_sim.bench.drives import CalibratedDrives
from tower_sim.bench.redaction import matches_exact


@pytest.fixture
def configuration_pair():
    packaged = json.loads(
        (DATA / "benchmark/runtime/ik-capsule/runtime/scene-cache.json").read_bytes()
    )["configuration"]
    return copy.deepcopy(packaged), packaged


def test_real_redaction_shape(configuration_pair):
    assert matches_exact(*configuration_pair)
    assert matches_exact({"value": 1}, {"value": True})
    assert matches_exact({"source": "redacted"}, {"source": "redacted"})


@pytest.mark.parametrize(
    "original,packaged,pointer",
    [
        ({"n": 1}, {"n": 2}, "/n"),
        ({"s": "one"}, {"s": "redacted"}, "/s"),
        ({"s": 7}, {"s": "redacted"}, "/s"),
        ({"s": "one"}, {"s": "two"}, "/s"),
        ({"s": "one"}, {"s": "archive-resource:invalid"}, "/s"),
        ({"a": 1}, {"a": 1, "b": 2}, "/b"),
        ({"a": 1, "b": 2}, {"a": 1}, "/b"),
        ({"a": [1]}, {"a": [1, 2]}, "/a/1"),
        ({"a": [1, 2]}, {"a": [1]}, "/a/1"),
        ({"a/b~c": 1}, {"a/b~c": 2}, "/a~1b~0c"),
        ({"a": None}, {"a": {}}, "/a"),
        ({"a": None}, {"a": []}, "/a"),
    ],
)
def test_redaction_mismatch(original, packaged, pointer):
    with pytest.raises(ValueError, match="JSON pointer") as error:
        matches_exact(original, packaged)
    assert repr(pointer) in str(error.value)


def test_stage_checks_redaction_and_never_caches_failure(configuration_pair, monkeypatch):
    original, packaged = configuration_pair
    stage = NS(
        GetPrimAtPath=lambda path: NS(GetAttribute=lambda key: NS(Get=lambda: json.dumps(original)))
    )
    monkeypatch.setitem(sys.modules, "pxr", NS(Usd=NS(Stage=NS(Open=lambda path: stage))))
    drives = CalibratedDrives("neutral.usd", packaged)
    assert drives.stage() is stage
    changed = copy.deepcopy(packaged)
    changed["command_hz"] += 1
    drives = CalibratedDrives("neutral.usd", changed)
    for _ in range(2):
        with pytest.raises(ValueError, match="/command_hz"):
            drives.stage()
        assert drives._stage is None


class ProvenanceGuard(dict):
    """A consumer may carry metadata but must not read a redacted leaf."""

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, str) and value == "redacted":
            raise AssertionError(f"Runtime dereferenced redacted provenance: {key}")
        return value

    def get(self, key, default=None):
        return self[key] if key in self else default  # noqa: SIM401 -- trap direct provenance reads


def guarded(value):
    if isinstance(value, dict):
        return ProvenanceGuard({k: guarded(v) for k, v in value.items()})
    if isinstance(value, list):
        return [guarded(v) for v in value]
    return value


def test_runtime_readers_do_not_dereference_redacted_provenance(monkeypatch, tmp_path):
    from pathlib import Path

    import numpy as np
    from tower_sim.backend.runtime_profiles import load_runtime_profiles
    from tower_sim.bench import cameras, render_freshness
    from tower_sim.bench.j2comp import J2Compensation

    runtime = DATA / "benchmark/runtime"
    load = json.loads
    monkeypatch.setattr(json, "loads", lambda *a, **kw: guarded(load(*a, **kw)))
    scene = json.loads((runtime / "ik-capsule/runtime/scene-cache.json").read_bytes())
    comp = J2Compensation(scene["configuration"])
    assert np.isfinite(comp.physical_from_encoder(np.zeros(7), 0.05)).all()
    profiles = runtime / "ik-capsule/runtime-profiles"
    bundle = json.loads((profiles / "bundle.json").read_bytes())
    for stage_hash in bundle["compatible_runtime_stages"]:
        load_runtime_profiles(profiles, stage_hash)

    # Exercise the production camera constructor, observation and report with
    # the established CPU SDK boundary doubles and the installed camera data.
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "packages/tower-sim/tests")
    )
    import test_cameras

    monkeypatch.setattr(test_cameras, "RUNTIME", runtime)
    monkeypatch.delenv("TOWER_P706_RECORDED_ROOT", raising=False)
    sdk = test_cameras.CameraSDK(monkeypatch, 1, tmp_path)
    sdk.freshness(render_freshness, monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", NS(Tensor=type("Tensor", (), {})))
    output = tmp_path / "capture"
    output.mkdir()
    camera = cameras.make_camera(sdk.sim, output, resources=sdk.resources, near_guard=sdk.near)
    try:
        camera.observe()
        camera.report()
    finally:
        camera.close()


def test_provenance_guard_rejects_a_new_reader():
    value = guarded({"provenance": {"asset": "redacted"}})
    with pytest.raises(AssertionError, match="Runtime dereferenced"):
        value["provenance"].get("asset")
