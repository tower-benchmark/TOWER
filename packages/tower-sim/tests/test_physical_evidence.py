"""Physical/contact contracts at callback, array and evidence-file boundaries."""

import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_construction import install
from tower_sim.bench import evidence
from tower_sim.bench.camera_quality import CameraQuality

from scripts.maintainer.archive_root import ArchivedToolPath


@pytest.mark.archive
@pytest.mark.parametrize(
    "actors,expected",
    [
        (["/World/envs/env_0/Blocks/b", "/ground"], 0),
        (["/World/envs/env_1/a", "/World/envs/env_1/b"], 1),
        (["/World/envs/env_10/a", "/ground"], ValueError),
        (["/World/envs/env_0/a", "/World/envs/env_1/b"], ValueError),
        (["/ground", "/World/other"], None),
    ],
)
def test_contact_routing(frozen_loader, actors, expected):
    old = frozen_loader("_p706_routes", ArchivedToolPath("batch_reference_worker.py"))
    for route in (old.route_contact, evidence.route_contact):
        if expected is ValueError:
            with pytest.raises(ValueError):
                route(actors, 2)
        else:
            assert route(actors, 2) == expected


def contact_sdk(monkeypatch, tmp_path):
    paths = [
        {arm: f"/World/envs/env_{i}/Station/InstalledRobots/{arm}" for arm in ("left", "back")}
        for i in range(2)
    ]
    prims = []
    for row in paths:
        for root in row.values():
            for finger in ("gripper_link1", "gripper_link2"):
                prims.append(
                    NS(GetPath=lambda path=root + "/" + finger: path, HasAPI=lambda api: True)
                )
    threshold = [None]
    attr = NS(Set=lambda value: threshold.__setitem__(0, value), Get=lambda: threshold[0])
    api = NS(CreateThresholdAttr=lambda: attr, GetThresholdAttr=lambda: attr)
    install(
        monkeypatch,
        "pxr",
        Usd=NS(PrimRange=lambda root: prims),
        UsdPhysics=NS(RigidBodyAPI=object()),
        PhysxSchema=NS(PhysxContactReportAPI=NS(Apply=lambda prim: api)),
        PhysicsSchemaTools=NS(intToSdfPath=lambda path: path),
    )
    callbacks = []
    install(
        monkeypatch,
        "omni.physx",
        get_physx_simulation_interface=lambda: NS(
            subscribe_contact_report_events=lambda callback: callbacks.append(callback) or object()
        ),
    )
    env = NS(
        stage=NS(GetPseudoRoot=lambda: None),
        robot_paths=paths,
        elapsed=np.array([0.1, 0.2]),
        simulation_time=0.25,
    )
    rows = []
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "episode-params.json").write_text("{}")
        rows.append(dict(output=directory, plan=directory, ticks=40, parameters={}, candidate={}))
    return env, rows, callbacks, prims


def header(a, b, count=1):
    return NS(actor0=a, actor1=b, num_contact_data=count, contact_data_offset=0)


@pytest.mark.archive
@pytest.mark.parametrize("screen", [False, True])
def test_contact_callbacks_and_receipts_match_frozen(frozen_loader, monkeypatch, tmp_path, screen):
    old = frozen_loader("_p706_contacts", ArchivedToolPath("batch_reference_worker.py"))
    old_screen = frozen_loader("_p706_screen", ArchivedToolPath("batch_screen_worker.py"))
    env, rows, callbacks, _ = contact_sdk(monkeypatch, tmp_path)

    def windows(*args):
        return [
            dict(arm=i % 2, block_name=f"b{i}", observed=False, event_count=0) for i in range(4)
        ]

    def matches(pair, stamp, window):
        return pair[1].endswith("/" + window["block_name"])

    monkeypatch.setattr(old.runtime, "reference_contact_windows", windows)
    monkeypatch.setattr(old.runtime, "reference_contact_matches", matches)
    results = []
    for original in (True, False):
        if screen:
            audit = (
                old_screen.make_contacts(old, env, rows)
                if original
                else evidence.ScreenContacts(env, rows)
            )
        else:
            audit = (
                old.BatchContacts(env, rows)
                if original
                else evidence.BatchContacts(
                    env,
                    rows,
                    reference_windows=True,
                    contact_windows=windows,
                    contact_matches=matches,
                )
            )
        assert callbacks[-1] == audit.contact
        reporting = [(row["output"] / "contact-reporting.json").read_bytes() for row in rows]
        for raw in reporting:
            assert json.loads(raw)["verified"] is True
        points = [NS(separation=-0.3)]
        finger = env.robot_paths[1]["left"] + "/gripper_link1"
        audit.contact([header(finger, "/World/envs/env_1/Blocks/b0")], points)
        assert len(audit.settling[1]) == 1 and not audit.settling[0]
        audit.begin_motion()
        audit.contact([header("/ground", "/background")], [])
        audit.contact([header(finger, "/World/envs/env_1/Blocks/b0", 0)], [])
        for i in range(4):
            audit.contact([header(finger, f"/World/envs/env_1/Blocks/b{i}")], points)
        report = audit.finish(1, 40)
        assert report["passed"] is (not screen)
        assert report["coverage_complete"] is True
        assert report["expected_physics_steps"] == 40
        collision = (rows[1]["output"] / "collision-validation.json").read_bytes()
        prior = copy.deepcopy(audit.events)
        if not screen:
            audit.contact([header(finger, "/World/envs/env_1/Blocks/unplanned")], points)
            assert audit.events == prior  # finished rows no longer accept callbacks
        audit.contact([header(finger, "/World/envs/env_0/Blocks/b0")], points)
        assert "Cross-environment contact" in audit.error
        latched = audit.error
        audit.contact([], [])
        assert audit.error == latched
        failed = audit.finish(0, 39)
        assert not failed["passed"] and not failed["coverage_complete"]
        assert failed["isolation_error"] == latched
        results.append((reporting, collision, audit.events, audit.settling, audit.counts, failed))
        audit.close()
        assert audit.subscription is None
    assert results[0] == results[1]


def test_contact_reporting_requires_all_fingertips(monkeypatch, tmp_path):
    env, rows, _, prims = contact_sdk(monkeypatch, tmp_path)
    prims.pop()
    with pytest.raises(ValueError, match="Contact reporting readback"):
        evidence.BatchContacts(env, rows)
    assert (
        json.loads((rows[1]["output"] / "contact-reporting.json").read_text())["verified"] is False
    )
    with pytest.raises(ValueError, match="explicit window/match"):
        evidence.BatchContacts(env, rows, reference_windows=True)


@pytest.mark.archive
@pytest.mark.parametrize(
    "fault", [None, "missing", "duplicate", "seed", "digest", "material", "nonfinite", "failed"]
)
def test_scoped_readback(frozen_loader, fault):
    old = frozen_loader("_p706_scoped", ArchivedToolPath("batch_screen_worker.py"))
    row = dict(env_id=1, candidate=dict(seed=17, episode_spec=dict(blocks=[dict(name="b")])))
    record = dict(
        env_id=1,
        seed=17,
        candidate_sha256="digest",
        block="b",
        passed=True,
        material_path="/World/envs/env_1/Materials/b",
        material_properties=[0.5, 0.3, 0.0],
    )
    readback = dict(
        passed=True,
        records=[copy.deepcopy(record)],
        usd_readback=dict(passed=True, records=[copy.deepcopy(record)]),
    )
    usd = readback["usd_readback"]["records"]
    if fault == "missing":
        usd.clear()
    elif fault == "duplicate":
        usd.append(copy.deepcopy(record))
    elif fault == "seed":
        usd[0]["seed"] = 18
    elif fault == "digest":
        usd[0]["candidate_sha256"] = "wrong"
    elif fault == "material":
        usd[0]["material_path"] = "/World/envs/env_0/Materials/b"
    elif fault == "nonfinite":
        usd[0]["material_properties"][0] = float("nan")
    elif fault == "failed":
        readback["records"][0]["passed"] = False
    results = []
    for fn in (old.scoped_readback, evidence.scoped_readback):
        if fault:
            with pytest.raises(ValueError) as error:
                fn(readback, row, lambda candidate: "digest")
            results.append(str(error.value))
        else:
            results.append(fn(readback, row, lambda candidate: "digest"))
    assert results[0] == results[1]


@pytest.mark.archive
@pytest.mark.parametrize(
    "total,late,passed", [(4.999, 0.999, True), (5.0, 0.999, False), (4.0, 1.0, False)]
)
def test_settling_exclusive_limits(frozen_loader, total, late, passed):
    old = frozen_loader("_p706_settle", ArchivedToolPath("runtime_worker.py"))
    before = np.array([[0.0, 0, 0, 0, 0, 0, 1]])
    after = before.copy()
    after[0, 0] = total / 1000
    recent = after.copy()
    recent[0, 1] = late / 1000
    actual, expected = (
        evidence.settling_metrics(before, recent, after),
        old.settling_metrics(before, recent, after),
    )
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["passed"] is passed
    for invalid in (np.empty((0, 7)), np.full((1, 7), np.nan), np.zeros((1, 6))):
        with pytest.raises(ValueError):
            evidence.settling_metrics(invalid, recent, after)


@pytest.mark.archive
def test_batch_evidence_append_units_selection_and_ownership(frozen_loader):
    old = frozen_loader("_p706_append", ArchivedToolPath("batch_reference_worker.py"))
    state = NS(
        **{
            key: np.ones((2, 7))
            for key in ("qpos", "qvel", "eef", "eef_world", "blocks", "block_velocities")
        }
    )
    robots = [
        NS(
            body_names=["gripper_link2", "gripper_base", "gripper_link1"],
            dof_names=["gripper_joint2", "joint1", "gripper_joint1"],
        )
    ] * 2
    poses = np.arange(2 * 3 * 7, dtype=float).reshape(2, 3, 7)
    efforts = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    env = NS(
        read=lambda: [copy.deepcopy(state), copy.deepcopy(state)],
        robots=robots,
        link_poses=[poses.copy(), poses.copy()],
        offsets=np.array([[0, 0, 0], [1.0, 2.0, 3.0]]),
        raw_joint_efforts=[efforts.copy(), efforts.copy()],
        raw_joint_states=[(positions.copy(),), (positions.copy(),)],
    )
    writers = [cls(env, 2) for cls in (old.BatchEvidence, evidence.BatchEvidence)]
    for writer in writers:
        writer.append([0.1, 0.2], selected=[1])
        assert not writer.records[0]
        row = writer.records[1][0]
        assert row["time"] == 0.2
        np.testing.assert_array_equal(
            row["finger_poses"][0, :, :3], poses[1, [1, 2, 0], :3] * 0.001 - [1.0, 2.0, 3.0]
        )
        np.testing.assert_array_equal(row["finger_poses"][0, :, 3:], poses[1, [1, 2, 0], 3:])
        np.testing.assert_array_equal(row["finger_forces"], [[0.06, 0.04]] * 2)
        np.testing.assert_array_equal(row["finger_positions"], [[0.006, 0.004]] * 2)
    for key in writers[0].records[1][0]:
        np.testing.assert_array_equal(writers[0].records[1][0][key], writers[1].records[1][0][key])
    env.link_poses[0][:] = np.nan
    assert np.isfinite(writers[1].records[1][0]["finger_poses"]).all()
    with pytest.raises(ValueError, match="Nonfinite physical evidence"):
        writers[1].append([0.1, 0.2], selected=[1])


def test_camera_archives_share_quality_counters(tmp_path):
    pixels = np.random.default_rng(11).integers(0, 256, (1, 4, 240, 320, 3), dtype=np.uint8)
    quality = CameraQuality(audit_every=1)
    camera = NS(names=("a", "b", "c", "d"), observe=lambda: dict(rgb=pixels))
    for i in range(2):
        archive = evidence.CameraArchive(camera, tmp_path / f"{i}.hdf5", quality=quality)
        archive.append(0.0, encode_policy=False)
        archive.close(complete=True)
    assert quality.stats["calls"] == quality.stats["audited"] == 8
    with pytest.raises(TypeError):
        evidence.CameraArchive(camera, tmp_path / "missing.hdf5")
