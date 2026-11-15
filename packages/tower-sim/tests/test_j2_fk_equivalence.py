import json
import shutil

import numpy as np
import pytest
from nero_kinematics.factory import MODELS
from nero_kinematics.fk import urdf
from nero_kinematics.profiles import COMMAND_BOUNDS, load_profile
from scipy.spatial.transform import Rotation
from tower_sim.bench import control
from tower_sim.bench.exact_j2_cache import CachedFK
from tower_sim.bench.fast_observed_joints import observed_joint_validator, tau2_bound
from tower_sim.bench.j2comp import J2Compensation
from tower_sim.bench.kinematics import EefModel
from tower_sim.bench.observation import encoder_observation
from tower_sim.bench.options import ControlOptions

from scripts.maintainer.archive_root import ArchivedToolPath
from tests.characterization.runtime_scenarios import ROOT, RUNTIME


def exact(left, right):
    assert np.asarray(left).dtype == np.asarray(right).dtype
    assert np.asarray(left).shape == np.asarray(right).shape
    assert np.asarray(left).tobytes() == np.asarray(right).tobytes()


@pytest.mark.archive
def test_urdf_and_topology_cache_exact(frozen_loader):
    original = frozen_loader("_p705_fk", "ik-capsule/runtime/urdf_fk.py")
    cached = CachedFK(urdf)
    frozen_cache = frozen_loader("_p705_cache", ArchivedToolPath("exact_j2_cache.py")).CachedFK(
        original
    )
    exact(urdf.limits(), original.limits())
    for q in [np.zeros(7), *np.random.default_rng(19).uniform(-1, 1, (12, 7))]:
        for width in (0.0, 0.05, 0.1):
            expected = original.fk(q, width)
            for candidate in (urdf.fk(q, width), cached(q, width), frozen_cache(q, width)):
                assert candidate.keys() == expected.keys()
                for link in expected:
                    exact(candidate[link], expected[link])


@pytest.mark.archive
@pytest.mark.parametrize("cached", [False, True])
def test_j2_all_functions_and_golden(compensation, frozen_loader, cached):
    original, config, cache_file, scene = compensation
    new = J2Compensation.from_cache(
        cache_file, scene, options=ControlOptions(exact_j2_cache=cached)
    )
    assert new.cache_checks == (27 if cached else 0)
    if cached:
        frozen_loader("_p705_install", ArchivedToolPath("exact_j2_cache.py")).install(original)
    golden = json.loads((ROOT / "tests/golden/benchmark-2026.11/j2comp.json").read_text())
    for row in golden:
        q, width, arm = row["q"], row["width"], row["arm"]
        physical = new.physical_from_encoder(q, width, arm)
        exact(physical, original.physical_from_encoder(q, width, arm))
        np.testing.assert_allclose(physical, row["physical"], atol=1e-9, rtol=0)
        exact(new.potential(q, width), original.potential(q, width))
        exact(new.tau2(q, width), original.tau2(q, width))
        exact(new.delta2(q, width, arm), original.delta2(q, width, arm))
        for iterations in (0, 1, 6):
            exact(
                new.encoder_from_physical(physical, width, arm, iterations),
                original.encoder_from_physical(physical, width, arm, iterations),
            )
        np.testing.assert_allclose(
            new.encoder_from_physical(physical, width, arm), row["roundtrip"], atol=1e-9, rtol=0
        )
    home = np.tile(np.r_[(control.GENERATION_LOWER + control.GENERATION_UPPER) / 2, 0.05], 2)
    exact(new.physical_from_encoder16(home), original.physical_from_encoder16(home))
    # Repeated rows exercise the exact-result reuse in command_targets; changed
    # arm widths/joints exercise misses. Structural caching never rounds q.
    changed = home.copy()
    changed[[1, 7, 9, 15]] += [0.01, 0.01, -0.01, -0.01]
    commands = np.stack([home, home, changed, changed])
    for enabled in (False, True):
        actual = new.command_targets([0, 0.01, 0.02, 0.03], commands, np.zeros((2, 7)), 12, enabled)
        expected = original.command_targets(
            [0, 0.01, 0.02, 0.03], commands, np.zeros((2, 7)), 12, enabled
        )
        for a, b in zip(actual, expected):
            exact(a, b)
    scene.write_bytes(b"mismatched station")
    with pytest.raises(ValueError, match="calibrated station"):
        J2Compensation.from_cache(cache_file, scene)


@pytest.mark.archive
@pytest.mark.parametrize("mode", ["off", "fast", "shadow"])
def test_observed_joint_equivalence(compensation, frozen_control, frozen_loader, mode):
    original, config, _, _ = compensation
    new = J2Compensation(config)
    frozen_fast = frozen_loader("_p705_fast", ArchivedToolPath("fast_observed_joints.py"))
    assert tau2_bound(new) == frozen_fast.tau2_bound(original)
    if mode != "off":
        frozen_fast.install(frozen_control, original, mode=mode, audit_every=2)
    validate = observed_joint_validator(
        new, options=ControlOptions(observed_joints=mode, observation_audit_every=2)
    )
    limits = np.tile([-3.0, 3.0], (16, 1))
    limits[np.r_[0:7, 8:15]] = np.tile(urdf.limits(), (2, 1))
    home = np.tile(np.r_[(control.GENERATION_LOWER + control.GENERATION_UPPER) / 2, 0.05], 2)
    inputs = [home.copy() for _ in range(3)]
    for col in (0, 1, 6, 7, 8, 9, 14, 15):
        for value in (np.nan, np.inf, -3.0, 3.0, -0.011, 0.121):
            q = home.copy()
            q[col] = value
            inputs.append(q)
    for col in (0, 1, 9, 14):
        for value in (
            limits[col, 1] + np.radians(0.5),
            np.nextafter(limits[col, 1] + np.radians(0.5), np.inf),
        ):
            q = home.copy()
            q[col] = value
            inputs.append(q)
    for q in inputs:
        try:
            expected = frozen_control.validate_observed_joints(q, original, limits)
        except (ValueError, RuntimeError) as exc:
            with pytest.raises((ValueError, RuntimeError)) as actual:
                validate(q, new, limits)
            assert str(actual.value) == str(exc)
        else:
            actual = validate(q, new, limits)
            if expected is None:
                assert actual is None
            else:
                exact(actual, expected)
    if mode != "off":
        assert validate.stats == frozen_control._fast_observed_joints_stats
    # A different compensation instance must fall back, not use another calibration's certificate.
    if mode == "fast":
        assert validate(home, J2Compensation(config), limits) is not None


@pytest.mark.archive
def test_eef_model_and_real_observation_equivalence(
    compensation, frozen_loader, monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(RUNTIME / "support"))
    monkeypatch.syspath_prepend(str(ArchivedToolPath("")))
    import task2_fixed.eef_ik_movejs as legacy

    shutil.copyfile(MODELS / "nero_sim_reference.urdf", tmp_path / "nero.urdf")
    original = legacy.EefIkMoveJsModel(None, tmp_path)
    tcp = load_profile("tcp")
    original.tcp_offset = np.array(tcp["tcp_offset_m"])
    original.tcp_rotation = Rotation.from_quat(tcp["tcp_rotation_xyzw"])
    model = EefModel(
        scene_urdf=tmp_path / "nero.urdf",
        tcp_offset=original.tcp_offset,
        tcp_rotation=original.tcp_rotation,
    )
    exact(model.lower, original.lower)
    exact(model.upper, original.upper)
    exact(model.command_lower, COMMAND_BOUNDS.lower_rad)
    exact(model.command_upper, COMMAND_BOUNDS.upper_rad)
    exact(model.identity["sdk_scene_fk_max_error"], original.identity["sdk_scene_fk_max_error"])
    exact(model.tcp_transform(), original.tcp_transform())
    q = np.array([0.1, -0.3, 0.4, 0.6, -0.2, 0.15, 0.1])
    exact(model.fk(q), original.fk(q))
    for invalid in (np.zeros(6), np.full(7, np.nan), np.full(7, np.inf)):
        for candidate in (model, original):
            with pytest.raises(ValueError):
                candidate.fk(invalid)
    old_comp, config, _, _ = compensation
    new_comp = J2Compensation(config)
    qpos = np.tile(np.r_[q, 0.05], 2)
    from types import SimpleNamespace

    worker = frozen_loader("_p705_worker", ArchivedToolPath("runtime_worker.py"))
    sim = SimpleNamespace(
        read=lambda: SimpleNamespace(qpos=qpos), env=SimpleNamespace(lula=original)
    )
    expected = worker.encoder_observation(
        sim, SimpleNamespace(j2comp=old_comp, urdf_fk=old_comp.urdf_fk)
    )
    assert encoder_observation(qpos, new_comp, model.tcp_rotation, model.tcp_offset) == expected


def test_eef_requires_explicit_scene_urdf(tmp_path):
    with pytest.raises(TypeError, match="scene_urdf"):
        EefModel()
    with pytest.raises(ValueError, match="explicit scene URDF"):
        EefModel(scene_urdf=None)
    with pytest.raises(FileNotFoundError):
        EefModel(scene_urdf=tmp_path / "missing.urdf")
    with pytest.raises(ValueError, match="verified cache digest"):
        EefModel(
            scene_urdf=MODELS / "nero_sim_reference.urdf", expected_scene_urdf_sha256="unbound"
        )
