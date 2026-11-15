import copy
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from tower_sim.bench import camera_quality, cameras, render_freshness
from tower_sim.bench.camera_quality import CameraQuality
from tower_sim.bench.resources import Resources

from scripts.maintainer.archive_root import ArchivedToolPath
from tests.characterization.runtime_scenarios import RUNTIME


@pytest.fixture
def frozen_cameras(monkeypatch, frozen_loader, frozen_import_dependencies):
    monkeypatch.syspath_prepend(str(RUNTIME / "support"))
    monkeypatch.syspath_prepend(str(RUNTIME / "ik-capsule/runtime"))
    monkeypatch.syspath_prepend(str(ArchivedToolPath("")))
    # Make the frozen chosen implementation explicit, then let its factory load
    # its own ordinary/live classes exactly as the original bootstrap does.
    for name in ("ordinary_capture", "live_observation", "four_capture", "render_freshness"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    old = frozen_loader("_p706_camera_probe", ArchivedToolPath("probe_batch_cameras.py"))
    yield old
    for name in ("ordinary_capture", "live_observation", "four_capture", "render_freshness"):
        sys.modules.pop(name, None)


class Attribute:
    def __init__(self, path, value=None, kind=None):
        self.path, self.value, self.kind = path, value, kind

    def Get(self):
        return self.value

    def Set(self, value):
        self.value = value
        if hasattr(self, "record"):
            self.record("mask", self.path, value)

    def GetPath(self):
        return self.path

    def GetName(self):
        return self.path.rsplit(".", 1)[-1]

    def GetTypeName(self):
        return self.kind

    def IsCustom(self):
        return False

    def HasAuthoredValueOpinion(self):
        return True


class Prim:
    def __init__(self, path):
        self.path = path
        self.attrs = {}
        self.matrix = np.eye(4)

    def GetPath(self):
        return self.path

    def GetName(self):
        return self.path.rsplit("/", 1)[-1]

    def IsA(self, kind):
        return kind == "Mesh"

    def CreateAttribute(self, name, kind=None, **kwargs):
        return self.attrs.setdefault(name, Attribute(self.path + "." + name, kind=kind))

    def GetAttribute(self, name):
        return self.CreateAttribute(name)

    def GetAttributes(self):
        return list(self.attrs.values())

    def AddAppliedSchema(self, name):
        pass

    def GetRelationship(self, name):
        attr = self.CreateAttribute(name)
        return NS(SetTargets=attr.Set, GetTargets=attr.Get)


class Stage:
    def __init__(self):
        self.prims = {}

    def GetPrimAtPath(self, path):
        return self.prims.setdefault(path, Prim(path))

    def Flatten(self):
        return copy.deepcopy(self)

    def GetRootLayer(self):
        return self

    def GetSessionLayer(self):
        return self


class CameraSDK:
    def __init__(self, monkeypatch, count, tmp_path):
        # Imported helper is in the same pytest directory, not production closure.
        from test_construction import install, normalized

        self.events = []
        self.counter = 0
        self.stage = Stage()
        self.product_index = 0
        self.count = count
        profile = json.loads((RUNTIME / "shared/camera-profile-near10.json").read_text())
        self.controls = dict(
            profile["renderer_controls"],
            **{
                "/omni/replicator/captureOnPlay": True,
                "/exts/omni.replicator.core/maxAssetLoadingTime": 30.0,
                "/renderer/activeGpu": 0,
            },
        )
        self.controls.setdefault("/rtx/post/tonemap/cm2Factor", 1.0)

        def record(*args):
            self.events.append(normalized(args))

        def setting(key, value):
            self.controls[key] = value
            record("setting", key, value)

        controls = NS(
            get=self.controls.get,
            set=setting,
            set_int=setting,
            get_as_float=lambda key: float(self.controls[key]),
        )
        install(monkeypatch, "carb", settings=NS(get_settings=lambda: controls))
        sdk = self

        def texture(path):
            state = NS(enabled=True)

            def enable(v):
                state.enabled = v
                record("texture", path, v)

            return NS(set_updates_enabled=enable, get_updates_enabled=lambda: state.enabled)

        self.texture = texture

        def product(path, resolution, name, tiled=False):
            sdk.product_index += 1
            prod_path = "/Render/" + name
            prim = sdk.stage.GetPrimAtPath(prod_path)
            prim.GetRelationship("camera").SetTargets(path if tiled else [path])
            if tiled:
                side = int(round(len(path) ** 0.5))
                resolution = (resolution[0] * side, resolution[1] * side)
            prim.CreateAttribute("resolution").Set(resolution)
            record("product", path, resolution, name, tiled)
            return NS(
                path=prod_path,
                hydra_texture=texture(prod_path),
                destroy=lambda: record("destroy", prod_path),
            )

        class Annotator:
            def __init__(self, **kwargs):
                record("annotator", kwargs)

            def attach(self, path):
                self.path = path
                record("attach", path)

            def detach(self, path):
                record("detach", path)

            def get_data(self, **kwargs):
                width, height = sdk.stage.GetPrimAtPath(self.path).GetAttribute("resolution").Get()
                return np.tile(np.arange(width, dtype=np.uint8)[None, :, None], (height, 1, 4))

        def step(**kwargs):
            sdk.counter += 1
            record("step", kwargs)

        rep = install(
            monkeypatch,
            "omni.replicator.core",
            create=NS(
                render_product=lambda path, resolution, name: product(path, resolution, name),
                render_product_tiled=lambda paths, tile_resolution, name: product(
                    paths, tile_resolution, name, True
                ),
            ),
            AnnotatorRegistry=NS(get_annotator=lambda name, **kw: Annotator(**kw)),
            orchestrator=NS(
                set_capture_on_play=lambda v: setting("/omni/replicator/captureOnPlay", v),
                step=step,
            ),
        )
        install(monkeypatch, "omni.graph.core", get_node_by_path=lambda path: None)
        install(
            monkeypatch,
            "isaacsim.core.utils.render_product",
            get_resolution=lambda path: (
                sdk.stage.GetPrimAtPath(path).GetAttribute("resolution").Get()
            ),
        )
        install(
            monkeypatch,
            "isaacsim.core.simulation_manager",
            SimulationManager=NS(get_simulation_time=lambda: 0.0),
        )

        def copy_spec(src, source, target, destination):
            prim = copy.deepcopy(src.GetPrimAtPath(source))
            prim.path = destination
            target.prims[destination] = prim
            record("copy_spec", source, destination)
            return True

        class Transform:
            def __init__(self, prim):
                self.prim = prim

            def MakeMatrixXform(self):
                return NS(
                    Set=lambda value: setattr(self.prim, "matrix", np.asarray(value).copy()),
                    Get=lambda: self.prim.matrix,
                )

        class Cache:
            def GetLocalToWorldTransform(self, prim):
                return prim.matrix

        import contextlib

        def primvars(prim):
            def create(name, kind):
                attr = prim.CreateAttribute(name, kind)
                attr.record = record
                return NS(GetAttr=lambda: attr)

            return NS(CreatePrimvar=create)

        source_stage = Stage()
        source_stage.GetPrimAtPath(
            "/__DualReplayCalibrationRender/DefaultPreviewDome"
        ).CreateAttribute("inputs:intensity").Set(123.0)
        install(
            monkeypatch,
            "pxr",
            Gf=NS(Vec3f=lambda *v: list(v), Matrix4d=lambda v: np.asarray(v)),
            Sdf=NS(
                CopySpec=copy_spec,
                ValueTypeNames=NS(Bool="bool", Float="float", Color3f="color3f"),
                ChangeBlock=contextlib.nullcontext,
                Path=lambda p: p,
            ),
            Usd=NS(
                Stage=NS(Open=lambda path: source_stage),
                PrimRange=lambda prim: [sdk.stage.GetPrimAtPath(prim.path + "/visual")],
                EditContext=lambda *args: contextlib.nullcontext(),
            ),
            UsdGeom=NS(
                Xform=NS(Define=lambda stage, path: stage.GetPrimAtPath(path)),
                Xformable=Transform,
                XformCache=Cache,
                Imageable=lambda prim: NS(MakeVisible=lambda: None),
                Mesh="Mesh",
                PrimvarsAPI=primvars,
            ),
        )

        def near(*args):
            record("near_guard", *args)
            return 100.0

        self.near = near
        install(
            monkeypatch, "jenga_" + "generator.render_validation", check_wrist_near_plane=self.near
        )
        install(
            monkeypatch, "tower_sim.backend.render_validation", check_wrist_near_plane=self.near
        )

        # Installed package parents are needed for frozen direct imports.
        class Camera:
            def __init__(self, path):
                self.prim_path = path
                self.paused = False
                self._render_product = NS(hydra_texture=texture(path))
                sdk.stage.GetPrimAtPath(path).CreateAttribute("clippingRange").Set((10.0, 3000.0))

            def get_clipping_range(self):
                return sdk.stage.GetPrimAtPath(self.prim_path).GetAttribute("clippingRange").Get()

            def set_clipping_range(self, *values):
                sdk.stage.GetPrimAtPath(self.prim_path).GetAttribute("clippingRange").Set(values)
                record("clip", self.prim_path, values)

            def is_paused(self):
                return self.paused

            def pause(self):
                self.paused = True
                record("pause", self.prim_path)

            def resume(self):
                self.paused = False
                record("resume", self.prim_path)

            def set_world_pose(self, xyz, quat, **kwargs):
                record("pose", self.prim_path, xyz, quat, kwargs)

        roles = tuple(cameras.Native61Capture.names.values())
        mounts = {
            role: {"camera_to_gripper": np.diag([0.001, 0.001, 0.001, 1.0]).tolist()}
            for role in ("left_wrist", "back_wrist")
        }
        state = NS(
            time=0.0,
            qpos=np.zeros(16),
            qvel=np.zeros(16),
            effort=np.zeros(16),
            gripper=np.zeros(2),
            blocks=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            block_velocities=np.zeros((1, 6)),
            eef=np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (2, 1)),
            eef_world=np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (2, 1)),
        )
        env = NS(
            n=count,
            stage=self.stage,
            config=NS(wrist_near_clip_mm=10.0),
            cameras=[{r: Camera(f"/env{i}/{r}") for r in roles} for i in range(count)],
            mounts=[copy.deepcopy(mounts) for _ in range(count)],
            report={
                "wrist_camera_proxies": {
                    r: {"camera_to_gripper_matrix": v["camera_to_gripper"]}
                    for r, v in mounts.items()
                }
            },
            robot_paths=[{r: f"/env{i}/{r}" for r in ("left", "back")} for i in range(count)],
            camera_report=[
                {r: {"intrinsic_matrix_k": np.eye(3).tolist()} for r in roles} for _ in range(count)
            ],
            robots=[NS(body_names=["gripper_base"]) for _ in range(2)],
            physics_output=NS(verify_capture=lambda: None),
            link_poses=[
                np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (count, 1, 1)) for _ in range(2)
            ],
            _latest_block_poses=np.stack([state.blocks] * count),
            offsets=np.zeros((count, 3)),
            read=lambda: [copy.deepcopy(state) for _ in range(count)],
        )
        # Effective reference template settings; resource paths are supplied below.
        settings = dict(
            instanceable_blocks=True,
            coupled_gripper_fingers=True,
            raytracing_cache_enabled=None,
            camera_rotation_corrections={"right_front_camera": 180, "left_back_camera": 180},
            gripper_friction={"static": 2.4, "dynamic": 2.0},
            render_warmup_captures=1,
            record_metric_evidence=True,
            initial_robot_state=None,
            layout_source="frozen",
            logical_episode_id=1,
            block_physics_material={
                "static_friction": 0.551,
                "dynamic_friction": 0.263,
                "restitution": 0.02,
            },
        )
        if recorded_root := os.environ.get("TOWER_P706_RECORDED_ROOT"):
            template = json.loads(
                (Path(recorded_root) / "task1-2/template/settings.json").read_text()
            )
            assert {key: template[key] for key in settings} == settings
            settings = template
        self.sim = NS(
            env=env, settings=settings, read=lambda: copy.deepcopy(state), read_batch=env.read
        )
        reference = tmp_path / "mounts.json"
        reference.write_text(
            json.dumps({r: {"adopted_usd": v["camera_to_gripper"]} for r, v in mounts.items()})
        )
        self.resources = Resources(
            dict(
                camera_profile=RUNTIME / "shared/camera-profile-near10.json",
                scene=reference,
                wrist_mounts=reference,
            )
        )
        self.reference = reference
        self.rep = rep
        self.record = record

    def freshness(self, module, monkeypatch):
        monkeypatch.setattr(
            module,
            "snapshot_dispatcher",
            lambda: dict(
                exists=True,
                compute_count=self.counter,
                reference_numerator=self.counter,
                reference_denominator=1,
            ),
        )
        monkeypatch.setattr(
            module,
            "scheduled_target",
            lambda: dict(available=True, numerator=self.counter, denominator=1, tolerance_s=1e-7),
        )


@pytest.mark.archive
@pytest.mark.parametrize("count", [1, 3, 16])
def test_camera_constructor_and_capture_equal(
    frozen_cameras, frozen_loader, monkeypatch, tmp_path, count
):
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    # Frozen constructor's data path is replaced only at its existing env boundary.
    monkeypatch.setenv("TOWER_ASSET_ROOT", str(tmp_path))
    sdk = CameraSDK(monkeypatch, count, tmp_path)
    old_fresh = importlib.import_module("render_freshness")
    sdk.freshness(old_fresh, monkeypatch)
    monkeypatch.setitem(sys.modules, "j2comp", NS(SCENE=sdk.reference, CONFIG={}))
    replay = frozen_loader("_p706_adoption", "ik-capsule/runtime/adapter.py")
    original_env, candidate_env = copy.deepcopy(sdk.sim.env), copy.deepcopy(sdk.sim.env)
    assert replay.adopt_usd_wrist_mounts(original_env) == cameras.adopt_usd_wrist_mounts(
        candidate_env
    )
    assert original_env.report == candidate_env.report
    old_output = tmp_path / "old"
    old_output.mkdir()
    sdk.events.clear()
    old = frozen_cameras.make_camera(
        sdk.sim, old_output, replay=replay, source_mount_reference=sdk.reference
    )
    # CPU source capture imports torch even on its CPU branch; supply an empty
    # module because no torch operation is executed in this effective branch.
    monkeypatch.setitem(sys.modules, "torch", NS(Tensor=type("Tensor", (), {})))
    first = old.observe()
    second = old.observe()
    old_report = old.report()
    old.close()
    expected = copy.deepcopy(sdk.events)
    expected_reports = {
        name: (old_output / name).read_bytes()
        for name in ("camera-near-plane-readback.json", "final-sensor-response-readback.json")
    }
    sdk = CameraSDK(monkeypatch, count, tmp_path)
    sdk.freshness(render_freshness, monkeypatch)
    new_output = tmp_path / "new"
    new_output.mkdir()
    sdk.events.clear()
    new = cameras.make_camera(sdk.sim, new_output, resources=sdk.resources, near_guard=sdk.near)
    a, b = new.observe(), new.observe()
    new_report = new.report()
    new.close()
    # Receipts use different versioned schemas/helpers; their paths and hashes
    # are the only intentional report differences. The mount-evidence tests verify them separately.
    for key in ("source_wrist_mount_adoption_path", "source_wrist_mount_adoption_sha256"):
        assert old_report.pop(key) != new_report.pop(key)
    assert new_report == old_report
    assert any(event[0] == "near_guard" for event in sdk.events)
    assert any(event[0] == "mask" for event in sdk.events)
    assert sdk.events == expected
    for left, right in ((a, first), (b, second)):
        np.testing.assert_array_equal(left["rgb"], right["rgb"])
        np.testing.assert_array_equal(left["qpos"], right["qpos"])
    assert new.schedule == old.schedule
    assert new.verified_capture_steps == old.verified_capture_steps
    for name, value in expected_reports.items():
        assert (new_output / name).read_bytes() == value


@pytest.mark.archive
@pytest.mark.parametrize("kind", ["random", "black", "white", "strided", "float", "wrong"])
def test_camera_quality_exact_audit(frozen_loader, kind):
    old = frozen_loader("_p706_quality", ArchivedToolPath("runtime_worker.py")).camera_quality
    value = np.random.default_rng(721).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    if kind == "black":
        value[:] = 0
    if kind == "white":
        value[:] = 255
    if kind == "strided":
        value = value[:, ::-1]
    if kind == "float":
        value = value.astype(float)
    if kind == "wrong":
        value = value[:3]
    new = CameraQuality(audit_every=1)
    assert json.dumps(old(value)) == json.dumps(new(value))
    assert new.stats["calls"] == 1
    assert new.stats["audited"] == (1 if kind in ("random", "black", "white") else 0)


@pytest.mark.parametrize("interval", [2, 3])
def test_camera_quality_audits_only_every_nth_fast_call(monkeypatch, interval):
    quality = CameraQuality(audit_every=interval)
    value = np.random.default_rng(721).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    reference = camera_quality.camera_quality
    checked = []

    def audited(rgb):
        checked.append(quality.stats["fast"])
        return reference(rgb)

    monkeypatch.setattr(camera_quality, "camera_quality", audited)
    for call in range(1, 2 * interval + 2):
        quality(value)
        assert checked == list(range(interval, call + 1, interval))
        assert quality.stats == dict(calls=call, fast=call, fallback=0, audited=call // interval)
    # Fallback invokes the reference but does not consume a fast-path audit slot.
    quality(value[:, ::-1])
    assert quality.stats == dict(
        calls=2 * interval + 2, fast=2 * interval + 1, fallback=1, audited=2
    )
    assert checked[-1] == 2 * interval + 1
    quality(value)
    assert quality.stats["audited"] == (2 * interval + 2) // interval
    assert len(checked) == quality.stats["audited"] + 1


def test_camera_quality_audit_mismatch_raises_on_scheduled_call(monkeypatch):
    quality = CameraQuality(audit_every=3)
    value = np.random.default_rng(721).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    fast = camera_quality._fast

    def drifted(rgb):
        result = fast(rgb)
        result["valid"] = not result["valid"]
        return result

    monkeypatch.setattr(camera_quality, "_fast", drifted)
    quality(value)
    quality(value)
    assert quality.stats["audited"] == 0
    with pytest.raises(RuntimeError, match="differs from the original"):
        quality(value)
    assert quality.stats == dict(calls=3, fast=3, fallback=0, audited=1)
