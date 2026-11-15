"""One synchronous near10 capture implementation, with explicit resource/guard inputs."""

import inspect
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .four_capture import Native61Capture
from .live_observation import BatchedLiveCameraObservation, LiveFourCameraObservation
from .render_freshness import verified_step
from .resources import save, sha


def assert_unchanged(before, after):
    if len(before) != len(after):
        raise ValueError("Environment count changed during rendering")
    for index, (old, new) in enumerate(zip(before, after, strict=True)):
        for field in ("time", "qpos", "qvel", "blocks", "block_velocities", "eef_world"):
            if not np.array_equal(getattr(old, field), getattr(new, field)):
                raise ValueError(f"Rendering changed env{index} physical {field}")


def compare_assignment(tiled, ordinary, *, maximum_mae=8.0, maximum_channel_bias=6.0):
    """All four views jointly distinguish row identity; ambiguous rows fail closed."""
    tiled, ordinary = np.asarray(tiled), np.asarray(ordinary)
    if (
        tiled.dtype != np.uint8
        or ordinary.dtype != np.uint8
        or tiled.shape != ordinary.shape
        or tiled.ndim != 5
        or tiled.shape[1:] != (4, 240, 320, 3)
        or not 1 <= len(tiled) <= 16
    ):
        raise ValueError("Expected matching uint8 [N,4,240,320,3] image sets")
    costs = np.empty((len(tiled), len(tiled)))
    per_camera, channel_bias = [], []
    for i in range(len(tiled)):
        for j in range(len(tiled)):
            costs[i, j] = np.abs(tiled[i].astype(float) - ordinary[j]).mean()
        delta = tiled[i].astype(float) - ordinary[i]
        per_camera.append(np.abs(delta).mean(axis=(1, 2, 3)))
        channel_bias.append(np.abs(delta.mean(axis=(1, 2))))
    per_camera, channel_bias = np.asarray(per_camera), np.asarray(channel_bias)
    unique = len(tiled) == 1 or all(
        costs[i, i] < np.delete(costs[i], i).min() for i in range(len(tiled))
    )
    passed = bool(
        unique
        and np.max(per_camera) <= maximum_mae
        and np.max(channel_bias) <= maximum_channel_bias
    )
    return dict(
        passed=passed,
        environment_count=len(tiled),
        assignment_scope="single_environment" if len(tiled) == 1 else "cross_environment",
        unique_diagonal_assignment=bool(unique),
        cross_environment_mae=costs.tolist(),
        per_camera_mae=per_camera.tolist(),
        absolute_channel_bias=channel_bias.tolist(),
        maximum_mae=maximum_mae,
        maximum_channel_bias=maximum_channel_bias,
        threshold_source="Native61 independent-reference image fidelity thresholds",
    )


def verify_source_mounts(mounts, reference):
    """Every environment must use the serial source's authored local mounts."""
    records = []
    for index, row in enumerate(mounts):
        for role in ("left_wrist", "back_wrist"):
            expected = np.asarray(reference[role]["adopted_usd"], float)
            actual = np.asarray(row[role]["camera_to_gripper"], float)
            if (
                expected.shape != (4, 4)
                or actual.shape != (4, 4)
                or not np.isfinite(expected).all()
                or not np.isfinite(actual).all()
                or not np.allclose(actual, expected, atol=1e-10, rtol=0)
            ):
                raise ValueError(f"Authored source wrist mount mismatch: env{index} {role}")
            records.append(
                dict(
                    env_id=index,
                    role=role,
                    expected_source_matrix=expected.tolist(),
                    authored_matrix=actual.tolist(),
                    passed=True,
                )
            )
    if not records:
        raise ValueError("Missing authored source wrist mounts")
    return records


def ordinary_references(camera):
    """Four reused ordinary products verify tiled row identity independently.

    Keep annotators attached until process shutdown: destroying them can delete
    the source SDK's shared articulation handles. Only rendering changes here.
    """
    import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]
    from pxr import Gf, Sdf, Usd  # pyright: ignore[reportMissingImports]

    env = camera.sim.env
    products, annotators = {}, {}
    for name, role in camera.names.items():
        product = rep.create.render_product(
            camera.batch_camera_paths[role][0], (320, 240), name="v3_assignment_reference_" + role
        )
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu", do_array_copy=True)
        annotator.attach(product.path)
        product.hydra_texture.set_updates_enabled(False)
        products[name], annotators[name] = product, annotator
    for product in camera.products.values():
        product.hydra_texture.set_updates_enabled(False)
    result, mappings = [], []
    camera._assignment_products, camera._assignment_annotators = products, annotators
    for index in range(env.n):
        mapping = {}
        for name, role in camera.names.items():
            product = products[name]
            prim = env.stage.GetPrimAtPath(product.path)
            path = camera.batch_camera_paths[role][index]
            with Usd.EditContext(env.stage, env.stage.GetSessionLayer()):
                prim.GetRelationship("camera").SetTargets([Sdf.Path(path)])
            if list(map(str, prim.GetRelationship("camera").GetTargets())) != [path]:
                raise ValueError("Ordinary reference camera relationship mismatch")
            whitepoint = camera.profile["cameras"][name]["whitepoint"]
            prim.CreateAttribute(
                "omni:rtx:post:tonemap:whitepoint", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(*whitepoint))
            product.hydra_texture.set_updates_enabled(True)
            mapping[name] = path
        # Product creation can migrate global exposure to its camera. Repair
        # the exact already-qualified attributes before reference acquisition.
        verified_step(
            lambda: rep.orchestrator.step(
                rt_subframes=2, delta_time=0.0, pause_timeline=True, wait_for_render=True
            ),
            initializing=True,
        )
        for attr, value in camera.expected_attributes:
            attr.Set(
                Gf.Vec3f(*value) if attr.GetTypeName() == Sdf.ValueTypeNames.Color3f else value
            )
        camera._verify_response()
        camera._apply_masks(tuple(camera.names.values()))
        for _ in range(32):
            receipt = verified_step(
                lambda: rep.orchestrator.step(
                    rt_subframes=6, delta_time=0.0, pause_timeline=True, wait_for_render=True
                )
            )
            with (Path(camera.output) / "assignment-render-delivery.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(dict(receipt, environment_index=index), allow_nan=False) + "\n"
                )
        views = []
        for name in camera.names:
            rgba = np.asarray(annotators[name].get_data(device="cpu", do_array_copy=True))
            if rgba.shape != (240, 320, 4):
                raise ValueError("Invalid ordinary reference shape")
            rgb = rgba[..., :3].copy()
            if camera.sim.settings.get("camera_rotation_corrections", {}).get(name) == 180:
                rgb = np.ascontiguousarray(np.rot90(rgb, 2))
            views.append(rgb)
        result.append(np.stack(views))
        mappings.append(mapping)
    for product in products.values():
        product.hydra_texture.set_updates_enabled(False)
    for product in camera.products.values():
        product.hydra_texture.set_updates_enabled(True)
    return np.stack(result), mappings


def adopt_usd_wrist_mounts(env):
    env.report = deepcopy(env.report)
    result = {}
    for name in ("left_wrist", "back_wrist"):
        authored = np.asarray(env.mounts[0][name]["camera_to_gripper"])
        result[name] = {
            "old": env.report["wrist_camera_proxies"][name]["camera_to_gripper_matrix"],
            "adopted_usd": authored.tolist(),
        }
        env.report["wrist_camera_proxies"][name]["camera_to_gripper_matrix"] = authored.tolist()
    return result


class SynchronousBatchCapture(BatchedLiveCameraObservation):
    def __init__(self, *args, adoption_path, adoption_sha, **kwargs):
        self.adoption_path, self.adoption_sha = adoption_path, adoption_sha
        self.pose_checks_passed = 0
        self.verified_capture_steps = 0
        self.last_render_delivery = None
        super().__init__(*args, **kwargs)
        assert self.profile["capture"]["wrist_near_clip_mm"] == 10.0
        readbacks = []
        for index, cameras in enumerate(self.sim.env.cameras):
            for role in ("left_wrist", "back_wrist"):
                near, far = cameras[role].get_clipping_range()
                assert abs(near - 10.0) < 1e-6, (index, role, near)
                paths = (
                    [self.reference_paths[role]]
                    if self.environment_count == 1
                    else [self.batch_camera_paths[role][index]]
                )
                for path in paths:
                    actual = (
                        self.sim.env.stage.GetPrimAtPath(path).GetAttribute("clippingRange").Get()
                    )
                    assert abs(float(actual[0]) - 10.0) < 1e-6, (path, actual)
                    readbacks.append(
                        dict(
                            environment=index,
                            role=role,
                            camera_near_mm=near,
                            rendered_prim=path,
                            rendered_near_mm=float(actual[0]),
                        )
                    )
        Path(self.output, "camera-near-plane-readback.json").write_text(
            json.dumps(
                dict(passed=True, near_mm=10.0, strict_block_guard=True, records=readbacks),
                indent=2,
            )
            + "\n"
        )
        # Native capture returns early for N=1 and retains its ordinary
        # products. Expose their real paths through the batch contract;
        # never create a second physical environment or tiled padding.
        if self.environment_count == 1:
            self.batch_camera_paths = {
                role: [self.reference_paths[role]] for role in self.names.values()
            }
            self.batch_camera_xforms = {
                role: [self.reference_xforms[role]] for role in self.names.values()
            }
            self.pose_checks_passed = 0
        # Sensor-schema installation follows the ordinary-camera constructor.
        # Drain that startup work before restoring the same authored responses.
        before_state = self.sim.env.read()
        self.initializing = True
        try:
            for _ in range(2):
                self._step(2)
        finally:
            self.initializing = False
        assert_unchanged(before_state, self.sim.env.read())
        globals_before = {key: self.controls.get(key) for key in self.profile["renderer_controls"]}
        for key, expected in self.profile["renderer_controls"].items():
            actual = self.controls.get(key)
            same = (
                np.allclose(actual, expected, rtol=1e-6, atol=1e-8)
                if isinstance(expected, (int, float, list)) and not isinstance(expected, bool)
                else actual == expected
            )
            if not same:
                self.controls.set(key, expected)
        from pxr import Gf  # pyright: ignore[reportMissingImports]

        responses = []
        for attr, expected in self.expected_attributes:
            prior = attr.Get()
            if not np.allclose(prior, expected, rtol=1e-6, atol=1e-8):
                attr.Set(Gf.Vec3f(*expected) if isinstance(expected, list) else expected)
            responses.append(
                dict(
                    path=str(attr.GetPath()),
                    before=np.asarray(prior).tolist(),
                    expected=expected,
                    after=np.asarray(attr.Get()).tolist(),
                )
            )
        self._verify_response()
        Path(self.output, "final-sensor-response-readback.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    physics_unchanged=True,
                    globals_before=globals_before,
                    expected_globals=self.profile["renderer_controls"],
                    globals_after={
                        key: self.controls.get(key) for key in self.profile["renderer_controls"]
                    },
                    camera_responses=responses,
                    scope="Startup only; every subsequent frame retains the original strict response check",
                ),
                indent=2,
                default=str,
            )
        )

    def _step(self, subframes):
        receipt = verified_step(
            lambda: Native61Capture._step(self, subframes), initializing=self.initializing
        )
        if not self.initializing:
            self.verified_capture_steps += 1
            self.last_render_delivery = receipt
            with (Path(self.output) / "render-delivery.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt, allow_nan=False) + "\n")
        return receipt

    def _pose(self):
        # Same calibrated reference camera path used in the generation
        # operation render_tiled_episodes.py; no approximate batched FK.
        LiveFourCameraObservation._pose(self)
        if self.environment_count == 1:
            self.pose_checks_passed += 1
            return
        from pxr import Gf, Sdf  # pyright: ignore[reportMissingImports]
        from scipy.spatial.transform import Rotation

        reference = self._legacy_batch_pose()
        env = self.sim.env
        with Sdf.ChangeBlock():
            for role in self.names.values():
                for index in range(1, self.environment_count):
                    matrix = reference[role][index]
                    if role.endswith("_wrist"):
                        self.near_guard(
                            env._latest_block_poses[index],
                            env.offsets[index],
                            matrix[3, :3],
                            Rotation.from_matrix(matrix[:3, :3].T).as_quat(),
                            self._wrist_intrinsics[index][role],
                            self._wrist_near_mm[index][role],
                            f"v3_probe_env_{index}:{role}",
                        )
                    self.batch_camera_xforms[role][index].Set(Gf.Matrix4d(matrix.tolist()))
        for role in self.names.values():
            for index in range(1, self.environment_count):
                if not np.array_equal(
                    self.batch_camera_xforms[role][index].Get(), reference[role][index]
                ):
                    raise ValueError("Per-environment reference camera transform changed")
        self.pose_checks_passed += 1

    def report(self):
        result = super().report()
        result.update(
            mode=(
                "synchronous_single_environment_capture"
                if self.environment_count == 1
                else "experimental_synchronous_tiled_capture"
            ),
            steady_replicator_steps=None,
            total_replicator_steps=self.steps,
            delivery_verified=True,
            dispatcher_stamp_verification=False,
            delivery_verification="synchronous_replicator_dispatcher_and_scheduled_timestamp",
            verified_capture_steps=self.verified_capture_steps,
            last_render_delivery=self.last_render_delivery,
            camera_pose_path="calibrated reference on every frame",
            rgb_transfer_mode="owned_cpu_array",
            pinned_host_reuse_protected_by_cuda_events=False,
            source_wrist_mount_adoption_path=str(self.adoption_path.resolve()),
            source_wrist_mount_adoption_sha256=self.adoption_sha,
            source_camera_calibration_verified=True,
        )
        if self.environment_count == 1:
            result["camera_batching"] = (
                "One ordinary product per camera role; one physical environment."
            )
        return result


def adopt_source_mounts(env, reference_path, output):
    reference_path = Path(reference_path)
    reference = json.loads(reference_path.read_text())
    if len(env.mounts) != env.n:
        raise ValueError("Missing per-environment wrist mount metadata")
    records = verify_source_mounts(env.mounts, reference)
    adopted = adopt_usd_wrist_mounts(env)
    for role in ("left_wrist", "back_wrist"):
        if not np.allclose(
            env.report["wrist_camera_proxies"][role]["camera_to_gripper_matrix"],
            reference[role]["adopted_usd"],
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError("Source wrist adoption did not install the expected calibration")
    source_file = inspect.getsourcefile(adopt_usd_wrist_mounts)
    if source_file is None:
        raise RuntimeError("Capture helper source unavailable")
    source = Path(source_file).resolve()
    result = dict(
        schema="tower-bench-source-wrist-mount-adoption-v1",
        passed=True,
        source_function="tower_sim.bench.cameras.adopt_usd_wrist_mounts",
        source_function_path=str(source),
        source_function_sha256=sha(source),
        serial_source_reference_path=str(reference_path.resolve()),
        serial_source_reference_sha256=sha(reference_path),
        environment_count=env.n,
        tolerance=1e-10,
        relative_tolerance=0.0,
        records=records,
        source_adoption=adopted,
    )
    save(output, result)
    return sha(output)


def make_camera(sim, output, *, resources, near_guard):
    adoption_path = Path(output) / "source-wrist-mount-adoption.json"
    adoption_sha = adopt_source_mounts(sim.env, resources.path("wrist_mounts"), adoption_path)
    return SynchronousBatchCapture(
        sim,
        output,
        device="cpu",
        subframes=6,
        adaptive_min_subframes=1,
        mask_policy="global_secondary",
        persistent_views=True,
        resolution=(320, 240),
        renderer_overrides={"/rtx/post/aa/op": 4},
        resources=resources,
        near_guard=near_guard,
        adoption_path=adoption_path,
        adoption_sha=adoption_sha,
    )


def verify_mount_evidence(camera_report, *, source_hashes, batch_root, expected_helper_sha256=None):
    """Bind the recorded helper to its source manifest, independent of install path.

    An optional trusted digest pins the expected implementation. The currently
    imported module is not historical evidence and may have moved or changed.

    Mount evidence validates only batches of 2..10 environments. All 12
    reference tasks have N=1 and cannot pass this batch verifier. Their
    serial reference checks remain separate from batch validation.
    """
    if camera_report.get("source_camera_calibration_verified") is not True:
        raise ValueError("Missing exact source wrist adoption")
    path = Path(camera_report["source_wrist_mount_adoption_path"])
    if not path.is_absolute() or not path.resolve().is_relative_to(Path(batch_root).resolve()):
        raise ValueError("Source mount receipt is outside batch")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Source mount receipt must be a regular non-symlink file")
    if sha(path) != camera_report["source_wrist_mount_adoption_sha256"]:
        raise ValueError("Source mount receipt changed")
    value = json.loads(path.read_text())
    helper = value.get("source_function_path")
    recorded_sha = value.get("source_function_sha256")
    if (
        value.get("schema") != "tower-bench-source-wrist-mount-adoption-v1"
        or value.get("passed") is not True
        or value.get("source_function") != "tower_sim.bench.cameras.adopt_usd_wrist_mounts"
        or not isinstance(helper, str)
        or not isinstance(recorded_sha, str)
        or len(recorded_sha) != 64
        or any(c not in "0123456789abcdef" for c in recorded_sha)
        or source_hashes.get(helper) != recorded_sha
        or (expected_helper_sha256 is not None and recorded_sha != expected_helper_sha256)
        or value.get("tolerance") != 1e-10
        or value.get("relative_tolerance") != 0.0
    ):
        raise ValueError("Source wrist adoption provenance/tolerance differs")
    reference_path = Path(value["serial_source_reference_path"])
    if reference_path.is_symlink() or not reference_path.is_file():
        raise ValueError("Serial reference must be a regular non-symlink file")
    if sha(reference_path) != value["serial_source_reference_sha256"]:
        raise ValueError("Original serial mount reference changed")
    reference = json.loads(reference_path.read_text())
    count = camera_report["environment_count"]
    if value.get("environment_count") != count or not 2 <= count <= 10:
        raise ValueError("Mount environment count differs")
    expected_keys = {(i, role) for i in range(count) for role in ("left_wrist", "back_wrist")}
    observed = set()
    for row in value["records"]:
        key = (row["env_id"], row["role"])
        if key not in expected_keys or key in observed:
            raise ValueError("Duplicate/missing source mount slot")
        observed.add(key)
        expected = np.asarray(reference[row["role"]]["adopted_usd"], float)
        arrays = [
            np.asarray(item, float)
            for item in (
                row["authored_matrix"],
                row["expected_source_matrix"],
                value["source_adoption"][row["role"]]["adopted_usd"],
            )
        ]
        if (
            row.get("passed") is not True
            or expected.shape != (4, 4)
            or not np.isfinite(expected).all()
            or any(
                a.shape != (4, 4)
                or not np.isfinite(a).all()
                or not np.allclose(a, expected, atol=1e-10, rtol=0)
                for a in arrays
            )
        ):
            raise ValueError(
                "Authored/installed source wrist mounts differ from original serial calibration"
            )
    if observed != expected_keys:
        raise ValueError("Not every environment has both source wrist mounts")
    return dict(
        receipt_sha256=sha(path),
        serial_source_reference_sha256=value["serial_source_reference_sha256"],
        environment_count=count,
    )
