"""Negative contracts at the actual SDK adapters, using CPU modules only."""

import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_cameras import CameraSDK
from test_construction import install
from tower_sim.bench import cameras, render_freshness
from tower_sim.bench.camera_guard import TerminalGuard
from tower_sim.bench.resources import sha
from tower_sim.bench.serial_camera_reference import SerialCameraReference

from scripts.maintainer.archive_root import ArchivedToolPath

# Shapes the frozen rule accepts or rejects. The pinned Replicator 1.13.36
# orchestrator.py assigns `_TIME_EPS = 1e-7` once as a direct statement of
# top-level step() and once in `async def step_async()`; PINNED_SDK_SHAPE mirrors
# that layout and must be accepted.
PINNED_SDK_SHAPE = (
    "def _get_time(n, d):\n    return n / d\n"
    "async def step_async():\n    _TIME_EPS = 1e-7\n"
    "def step():\n    _multitick = True\n    _TIME_EPS = 1e-7\n"
)
EPSILON_SOURCES = [
    ("def step():\n    _TIME_EPS = 1e-7\n", True),
    (PINNED_SDK_SHAPE, True),
    ("def step():\n    if condition:\n        _TIME_EPS = 1e-7\n", True),
    ("def step():\n    def nested():\n        _TIME_EPS = 1e-7\n", True),
    ("_TIME_EPS = 1e-7\ndef step():\n    _TIME_EPS = 1e-7\n", True),
    ("def step():\n    _TIME_EPS = 1e-6\n", False),
    ("def step():\n    _TIME_EPS = 1e-7\n    _TIME_EPS = 1e-7\n", False),
    ("def step():\n    _TIME_EPS = 1e-7\ndef step():\n    _TIME_EPS = 1e-7\n", False),
    ("_TIME_EPS = 1e-7\ndef step():\n    pass\n", False),
    ("class Camera:\n    def step(self):\n        _TIME_EPS = 1e-7\n", False),
    ("def step():\n    _TIME_EPS = '1e-7'\n", False),
]


@pytest.mark.archive
@pytest.mark.parametrize("source,valid", EPSILON_SOURCES)
def test_epsilon_rule_matches_frozen(frozen_loader, source, valid):
    frozen = frozen_loader("_p706_render_freshness", ArchivedToolPath("render_freshness.py"))
    outcomes = []
    for function in (frozen._source_time_epsilon, render_freshness._source_time_epsilon):
        try:
            outcomes.append(function(source))
        except RuntimeError:
            outcomes.append(None)
    assert outcomes[0] == outcomes[1]
    assert (outcomes[1] == 1e-7) is valid


@pytest.mark.parametrize("source,valid", EPSILON_SOURCES)
def test_sdk_step_epsilon_semantics(monkeypatch, tmp_path, source, valid):
    path = tmp_path / "orchestrator.py"
    path.write_text(source)
    install(
        monkeypatch,
        "omni.replicator.core.scripts.orchestrator",
        __file__=str(path),
        _orchestrator=NS(_sim_times_to_write=[(40, 400)]),
    )
    count = [0]
    node = NS(
        is_valid=lambda: True,
        get_compute_count=lambda: count[0],
        get_attribute=lambda name: NS(get=lambda: 40 if "Numerator" in name else 400),
    )
    install(monkeypatch, "omni.graph.core", get_node_by_path=lambda path: node)

    def step():
        count[0] += 1

    if valid:
        receipt = render_freshness.verified_step(step)
        assert receipt["verified"] is True
        assert receipt["scheduled"]["source_sha256"] == sha(path)
        assert receipt["scheduled"]["tolerance_s"] == 1e-7
    else:
        with pytest.raises(RuntimeError, match="tolerance changed or is ambiguous"):
            render_freshness.scheduled_target()
        with pytest.raises(RuntimeError, match="scheduled timestamp unavailable"):
            render_freshness.verified_step(step)
    assert count == [1]


def rejecting_guard(env, output):
    def reject(*args):
        raise RuntimeError(f"Wrist near plane may clip a task block: {args[-1]} (9.9 < 10.0 mm)")

    guard = TerminalGuard(reject, env, output)
    guard.enabled = True
    return guard


@pytest.mark.archive
def test_serial_reference_routes_native_guard_to_selected_environment(monkeypatch, tmp_path):
    sdk = CameraSDK(monkeypatch, 3, tmp_path)
    sdk.sim.env.elapsed = np.full(3, 0.1)
    guard = rejecting_guard(sdk.sim.env, tmp_path / "guard")
    camera = NS(sim=sdk.sim, names=cameras.Native61Capture.names)
    reference = SerialCameraReference(camera, 2, guard=guard)
    transforms = reference._pose()
    assert set(transforms) == set(camera.names)
    assert guard.reference_index is None
    rejected = guard.consume(40)
    assert set(rejected) == {2}
    assert rejected[2]["label"] == "native61_v2_2:left_wrist"
    assert rejected[2]["policy_observation_accepted"] is False
    for index in (-1, 3, True):
        with pytest.raises(ValueError, match="environment index"):
            SerialCameraReference(camera, index, guard=guard)


def test_guard_repeated_rejection_and_tick_mismatch(tmp_path):
    env = NS(n=2, elapsed=np.full(2, 0.1))
    guard = rejecting_guard(env, tmp_path / "guard")
    args = (
        [],
        np.zeros(3),
        np.zeros(3),
        [0, 0, 0, 1],
        np.eye(3),
        10.0,
        "v3_probe_env_1:back_wrist",
    )
    assert guard.check(*args) == 9.9
    first = copy.deepcopy(guard.rejections[1])
    env.elapsed[:] = 0.2
    assert guard.check(*args) == 9.9
    assert guard.rejections[1] == first
    assert len(list(guard.output.glob("*.json"))) == 1
    with pytest.raises(AssertionError, match="Unconsumed"):
        guard.consume(80)
    assert guard.consumed == set()
    assert set(guard.consume(40)) == {1}
    assert guard.consume(80) == {}
    with pytest.raises(RuntimeError):
        guard.check(*args[:-1], "v3_probe_env_2:back_wrist")


@pytest.fixture
def mount_receipt(tmp_path):
    roles = ("left_wrist", "back_wrist")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({role: {"adopted_usd": np.eye(4).tolist()} for role in roles}))
    env = NS(
        n=2,
        mounts=[{r: {"camera_to_gripper": np.eye(4)} for r in roles} for _ in range(2)],
        report={
            "wrist_camera_proxies": {
                r: {"camera_to_gripper_matrix": np.eye(4).tolist()} for r in roles
            }
        },
    )
    path = tmp_path / "receipt.json"
    digest = cameras.adopt_source_mounts(env, reference, path)
    report = dict(
        source_camera_calibration_verified=True,
        source_wrist_mount_adoption_path=str(path),
        source_wrist_mount_adoption_sha256=digest,
        environment_count=2,
    )
    value = json.loads(path.read_text())
    hashes = {value["source_function_path"]: value["source_function_sha256"]}
    return path, reference, report, value, hashes


@pytest.mark.parametrize(
    "change,match",
    [
        ("matrix", "mounts differ"),
        ("missing", "both source wrist mounts"),
        ("outside", "outside batch"),
        ("reference", "reference changed"),
        ("tolerance", "provenance/tolerance"),
        ("symlink", "non-symlink"),
        ("directory", "regular"),
        ("reference-symlink", "non-symlink"),
        ("one", "count differs"),
        ("eleven", "count differs"),
    ],
)
def test_c19_rejections(mount_receipt, tmp_path, change, match):
    path, reference, report, value, hashes = mount_receipt
    if change == "matrix":
        value["records"][0]["authored_matrix"][0][1] = 1.00001e-10
    elif change == "missing":
        value["records"].pop()
    elif change == "outside":
        outside = tmp_path.parent / (tmp_path.name + "-outside.json")
        outside.write_bytes(path.read_bytes())
        report["source_wrist_mount_adoption_path"] = str(outside)
    elif change == "reference":
        reference.write_text("{}")
    elif change == "tolerance":
        value["tolerance"] = 1e-9
    elif change in ("symlink", "directory"):
        other = tmp_path / "other"
        if change == "symlink":
            other.symlink_to(path)
        else:
            other.mkdir()
        report["source_wrist_mount_adoption_path"] = str(other)
    elif change == "reference-symlink":
        other = tmp_path / "reference-link"
        other.symlink_to(reference)
        value["serial_source_reference_path"] = str(other)
    else:
        value["environment_count"] = report["environment_count"] = 1 if change == "one" else 11
    path.write_text(json.dumps(value))
    report["source_wrist_mount_adoption_sha256"] = sha(path)
    with pytest.raises(ValueError, match=match):
        cameras.verify_mount_evidence(report, source_hashes=hashes, batch_root=tmp_path)


def test_c19_uses_recorded_source_manifest_and_optional_trusted_digest(mount_receipt, tmp_path):
    path, _, report, value, _ = mount_receipt
    # A receipt from a different install remains valid after the helper changes.
    recorded_path, recorded_sha = "/recorded-install/tower_sim/bench/cameras.py", "a" * 64
    value.update(source_function_path=recorded_path, source_function_sha256=recorded_sha)
    path.write_text(json.dumps(value))
    report["source_wrist_mount_adoption_sha256"] = sha(path)
    hashes = {recorded_path: recorded_sha}
    result = cameras.verify_mount_evidence(
        report, source_hashes=hashes, batch_root=tmp_path, expected_helper_sha256=recorded_sha
    )
    assert result["environment_count"] == 2
    for manifest, expected in (({recorded_path: "b" * 64}, None), (hashes, "b" * 64)):
        with pytest.raises(ValueError, match="provenance"):
            cameras.verify_mount_evidence(
                report, source_hashes=manifest, batch_root=tmp_path, expected_helper_sha256=expected
            )


@pytest.mark.archive
def test_serial_reference_delivery_matches_frozen(
    frozen_loader, monkeypatch, tmp_path, frozen_import_dependencies
):
    import sys

    from tests.characterization.runtime_scenarios import RUNTIME

    monkeypatch.syspath_prepend(str(RUNTIME / "support"))
    old_four = frozen_loader("_p706_serial_four", "ik-capsule/runtime/four_capture.py")
    old_fresh = frozen_loader("_p706_serial_fresh", ArchivedToolPath("render_freshness.py"))
    old = frozen_loader("_p706_serial", ArchivedToolPath("serial_camera_reference.py"))
    monkeypatch.setitem(sys.modules, "four_capture", old_four)
    monkeypatch.setitem(sys.modules, "render_freshness", old_fresh)
    results = []
    for label, cls in (("old", old.SerialCameraReference), ("new", SerialCameraReference)):
        sdk = CameraSDK(monkeypatch, 3, tmp_path)
        sdk.sim.env.elapsed = np.full(3, 0.1)
        sdk.freshness(render_freshness, monkeypatch)
        sdk.freshness(old_fresh, monkeypatch)
        output = tmp_path / label
        output.mkdir()
        camera = cameras.make_camera(sdk.sim, output, resources=sdk.resources, near_guard=sdk.near)
        guard = TerminalGuard(sdk.near, sdk.sim.env, output / "guard")
        install(
            monkeypatch, "camera_guard_terminal", reference_environment=guard.reference_environment
        )
        # USD returns an invalid prim for an uncreated independent namespace.
        original_get = sdk.stage.GetPrimAtPath
        root = "/__V3IndependentSerialSourceCameras_2"
        monkeypatch.setattr(
            sdk.stage,
            "GetPrimAtPath",
            lambda path, root=root, sdk=sdk, get=original_get: (
                None if path == root and path not in sdk.stage.prims else get(path)
            ),
        )
        monkeypatch.setattr(
            sys.modules["pxr"].UsdGeom.Xform,
            "Define",
            lambda stage, path, get=original_get: get(path),
        )
        reference = cls(camera, 2, **({"guard": guard} if label == "new" else {}))
        before = old.physical_snapshot(sdk.sim.env.read())
        sdk.events.clear()
        images, receipt = reference.observe()
        from tower_sim.bench.serial_camera_reference import physical_snapshot

        assert before == physical_snapshot(sdk.sim.env.read())
        assert len(receipt["render_deliveries"]) == 32
        assert all(item["verified"] for item in receipt["render_deliveries"])
        assert receipt["source_sha256"] == sha(receipt["source_path"])
        # Only the ported pose helper's source location/bytes differ.
        receipt.pop("source_path")
        receipt.pop("source_sha256")
        results.append((images, receipt, copy.deepcopy(sdk.events)))
        assert all(p.hydra_texture.get_updates_enabled() for p in camera.products.values())
        assert all(not p.hydra_texture.get_updates_enabled() for p in reference.products.values())
        camera.close()
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1:] == results[1][1:]
