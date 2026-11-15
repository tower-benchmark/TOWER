"""Recorded physical arrays and camera archives; no loop or scoring ownership."""

import base64
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .camera_quality import CameraQuality
from .resources import save, sha


def environment_for_actor(path, count):
    match = re.search(r"^/World/envs/env_(\d+)(?:/|$)", path)
    if not match:
        return None
    index = int(match[1])
    if index >= count:
        raise ValueError("Contact references an environment outside the batch")
    return index


def route_contact(actors, count):
    ids = {environment_for_actor(path, count) for path in actors} - {None}
    if len(ids) > 1:
        raise ValueError("Cross-environment contact: isolation failed")
    return next(iter(ids)) if ids else None


class BatchEvidence:
    def __init__(self, env, count):
        self.env, self.records = env, [[] for _ in range(count)]

    def append(self, stamps, selected=None):
        states = self.env.read()
        selected = range(len(states)) if selected is None else selected
        for index in selected:
            state = states[index]
            row: dict[str, Any] = {
                key: np.asarray(getattr(state, key)).copy()
                for key in ("qpos", "qvel", "eef", "eef_world", "blocks", "block_velocities")
            }
            poses, force, positions = [], [], []
            for arm, robot in enumerate(self.env.robots):
                links = [
                    robot.body_names.index(n)
                    for n in ("gripper_base", "gripper_link1", "gripper_link2")
                ]
                pose = self.env.link_poses[arm][index, links].copy()
                pose[:, :3] = pose[:, :3] * 0.001 - self.env.offsets[index]
                fingers = [robot.dof_names.index(n) for n in ("gripper_joint1", "gripper_joint2")]
                poses.append(pose)
                force.append(self.env.raw_joint_efforts[arm][index, fingers] * 0.001)
                positions.append(self.env.raw_joint_states[arm][0][index, fingers] * 0.001)
            row.update(
                time=float(stamps[index]),
                finger_poses=np.array(poses),
                finger_forces=np.array(force),
                finger_positions=np.array(positions),
            )
            if not all(np.isfinite(v).all() for v in row.values()):
                raise ValueError(f"Nonfinite physical evidence in env{index}")
            self.records[index].append(row)
        return states

    def write(self, index, directory):
        import h5py

        directory.mkdir(parents=True, exist_ok=True)
        (directory / "episode").mkdir(exist_ok=True)
        rows = self.records[index]
        if not rows:
            raise ValueError("Missing per-environment evidence")
        values = {key: np.array([r[key] for r in rows]) for key in rows[0]}
        with h5py.File(directory / "episode/states.hdf5", "x") as handle:
            handle.attrs.update(
                schema="tower-v3-physical-evidence-v1", units="SI", quaternion="xyzw", env_id=index
            )
            for key, value in values.items():
                if not key.startswith("finger_"):
                    handle[key] = value
        np.savez_compressed(
            directory / "finger-link-poses.npz", time=values["time"], poses_m=values["finger_poses"]
        )
        np.savez_compressed(
            directory / "finger-feedback.npz",
            time=values["time"],
            forces_n=values["finger_forces"],
            positions_m=values["finger_positions"],
        )
        save(directory / "finger_geometry.json", self.env.finger_geometry)
        return values


def scoped_readback(readback, row, digest):
    """Reject missing, duplicate or cross-environment asset/material evidence."""
    index, candidate = row["env_id"], row["candidate"]
    usd = readback["usd_readback"]
    if readback.get("passed") is not True or usd.get("passed") is not True:
        raise ValueError("Heterogeneous construction/readback did not pass")
    wanted = {b["name"] for b in candidate["episode_spec"]["blocks"]}
    result = dict(env_id=index, seed=candidate["seed"], candidate_sha256=digest(candidate))
    for label, records in (("physics", readback["records"]), ("usd", usd["records"])):
        selected = [r for r in records if r["env_id"] == index]
        if len(selected) != len(wanted) or {r["block"] for r in selected} != wanted:
            raise ValueError("Incomplete or duplicated per-environment block readback")
        for record in selected:
            if record["seed"] != candidate["seed"] or record["candidate_sha256"] != digest(
                candidate
            ):
                raise ValueError("Readback belongs to another candidate")
            if label == "physics" and record.get("passed") is not True:
                raise ValueError("Physical readback failed")
            if label == "usd":
                if not record["material_path"].startswith(f"/World/envs/env_{index}/Materials/"):
                    raise ValueError("Material readback belongs to another environment")
                material = np.asarray(record["material_properties"])
                if material.shape != (3,) or not np.isfinite(material).all():
                    raise ValueError("Invalid material properties")
        result[label + "_records"] = selected
    result.update(heterogeneous_asset_readback_valid=True, materials_readback_valid=True)
    return result


def settling_metrics(before, late, after):
    """Exact generator replay acceptance: total <5mm and final-quarter drift <1mm."""
    arrays = [np.asarray(v, float) for v in (before, late, after)]
    if (
        arrays[0].ndim != 2
        or arrays[0].shape[1] != 7
        or not len(arrays[0])
        or any(v.shape != arrays[0].shape or not np.isfinite(v).all() for v in arrays)
    ):
        raise ValueError("Invalid physical block poses for source settling checks")
    before, late, after = arrays
    displacement = np.linalg.norm(after[:, :3] - before[:, :3], axis=1) * 1000
    drift = np.linalg.norm(after[:, :3] - late[:, :3], axis=1) * 1000
    return dict(
        displacement_mm=displacement,
        final_quarter_second_drift_mm=drift,
        max_displacement_mm=float(displacement.max()),
        max_final_quarter_second_drift_mm=float(drift.max()),
        maximum_displacement_mm_exclusive=5.0,
        maximum_late_drift_mm_exclusive=1.0,
        passed=bool(displacement.max() < 5.0 and drift.max() < 1.0),
    )


class CameraArchive:
    def __init__(self, camera, path, *, quality: CameraQuality):
        if not isinstance(quality, CameraQuality):
            raise TypeError("Pass the process-wide shared CameraQuality instance")
        self.quality = quality
        import h5py

        self.camera = camera
        self.file = h5py.File(path, "x")
        self.times = self.file.create_dataset("time", (0,), maxshape=(None,), dtype="f8")
        self.frames = {
            name: self.file.create_dataset(
                name, (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
            )
            for name in camera.names
        }
        self.file.attrs.update(fps=30.0, resolution="320x240", complete=False)
        self._latest_pixels = None
        self._latest_encoded = None
        self.all_observations_valid = True
        self.low_visibility = {}

    @property
    def latest(self):
        """Lossless transport for the latest validated frame, encoded on demand.

        Replay capture runs much faster than policy requests. Keep owned raw
        pixels so unused replay frames never pay for PNG/base64 transport.
        """
        if self._latest_pixels is None:
            return None
        if self._latest_encoded is None:
            from PIL import Image

            encoded = {}
            for index, name in enumerate(self.camera.names):
                stream = io.BytesIO()
                Image.fromarray(self._latest_pixels[index]).save(stream, format="PNG")
                encoded[name] = base64.b64encode(stream.getvalue()).decode("ascii")
            self._latest_encoded = encoded
        return self._latest_encoded

    def append(self, stamp, *, encode_policy=True):
        # A failed capture must never leave a stale model observation available.
        self._latest_pixels = self._latest_encoded = None
        observed = self.camera.observe()["rgb"]
        pixels = (
            observed.detach().cpu().numpy()[0].copy()
            if hasattr(observed, "detach")
            else np.array(observed[0], copy=True)
        )
        format_valid = pixels.shape == (4, 240, 320, 3) and pixels.dtype == np.uint8
        quality = (
            {name: self.quality(pixels[index]) for index, name in enumerate(self.camera.names)}
            if format_valid
            else {"format": dict(valid=False, shape=list(pixels.shape), dtype=str(pixels.dtype))}
        )
        self.file.attrs["last_frame_quality_json"] = json.dumps(dict(time_s=stamp, cameras=quality))
        validator = getattr(self.camera, "validate_observation", None)
        try:
            delivery = (
                validator(pixels, stamp, quality)
                if validator is not None
                else dict(
                    valid=all(q["valid"] for q in quality.values()),
                    reason="Legacy capture without independent delivery validator",
                )
            )
        except Exception as exc:  # noqa: BLE001 -- latch native callback/delivery failures
            delivery = dict(
                valid=False, reason=f"Capture validation failed: {type(exc).__name__}: {exc}"
            )
        valid = format_valid and delivery.get("valid") is True
        self.file.attrs["last_delivery_json"] = json.dumps(delivery, allow_nan=False)
        if not valid or not all(q["valid"] for q in quality.values()):
            # Retain exact questionable pixels before accepting physical occlusion
            # or rejecting incomplete delivery. No rejected frame reaches the model.
            directory = Path(str(self.file.filename) + ".diagnostics")
            directory.mkdir(exist_ok=True)
            raw_path = directory / f"{len(self.times):08d}.npz"
            with raw_path.open("xb") as stream:
                np.savez_compressed(stream, pixels=pixels)
            detail = dict(
                time_s=stamp,
                cameras=quality,
                delivery=delivery,
                camera_order=list(self.camera.names),
                raw_pixels=dict(path=str(raw_path.resolve()), sha256=sha(raw_path)),
            )
            if not valid:
                with raw_path.with_suffix(".json").open("x") as stream:
                    json.dump(detail, stream, allow_nan=False)
            if valid:
                self.low_visibility[str(len(self.times))] = detail
                self.file.attrs["verified_low_visibility_json"] = json.dumps(
                    self.low_visibility, allow_nan=False
                )
        if validator is not None:
            self.file.attrs["delivery_validator"] = "tower-v3-observation-delivery-v1"
        if not valid:
            self.all_observations_valid = False
            self.file.flush()
            raise RuntimeError(f"Invalid camera observation at {stamp}s: {delivery}")
        from PIL import Image

        count = len(self.times)
        self.times.resize((count + 1,))
        self.times[count] = stamp
        jpeg_hashes = {}
        for index, name in enumerate(self.camera.names):
            stream = io.BytesIO()
            Image.fromarray(pixels[index]).save(stream, format="JPEG", quality=95)
            raw = stream.getvalue()
            jpeg_hashes[name] = hashlib.sha256(raw).hexdigest()
            self.frames[name].resize((count + 1,))
            self.frames[name][count] = np.frombuffer(raw, np.uint8)
        if str(count) in self.low_visibility:
            detail = self.low_visibility[str(count)]
            detail["archived_jpeg_sha256"] = jpeg_hashes
            with Path(detail["raw_pixels"]["path"]).with_suffix(".json").open("x") as stream:
                json.dump(detail, stream, allow_nan=False)
            self.file.attrs["verified_low_visibility_json"] = json.dumps(
                self.low_visibility, allow_nan=False
            )
        self.file.flush()
        self._latest_pixels = pixels
        return self.latest if encode_policy else None

    def close(self, *, complete=False):
        times = np.asarray(self.times)
        consistent = (
            len(times) > 0
            and np.isfinite(times).all()
            and np.all(np.diff(times) > 0)
            and all(len(stream) == len(times) for stream in self.frames.values())
        )
        consistent = bool(consistent and self.all_observations_valid)
        self.file.attrs["complete"] = bool(complete and consistent)
        self.file.attrs["streams_consistent"] = bool(consistent)
        self.file.close()
        if complete and not consistent:
            raise RuntimeError("Cannot complete a camera archive with inconsistent streams")


class BatchContacts:
    """Contact routing with caller-owned reference semantics.

    reference_windows=False is the screen-only default. Reference validation requires True
    and the frozen contact_windows/contact_matches for reference validation;
    an empty-window audit cannot pass the reference collision gate.
    """

    def __init__(
        self, env, rows, *, reference_windows=False, contact_windows=None, contact_matches=None
    ):
        if reference_windows and (contact_windows is None or contact_matches is None):
            raise ValueError("Reference contacts require explicit window/match functions")
        self.contact_matches = contact_matches or (lambda *args: False)
        import omni.physx  # pyright: ignore[reportMissingImports]
        from pxr import PhysxSchema, Usd, UsdPhysics  # pyright: ignore[reportMissingImports]

        self.env, self.rows, self.phase = env, rows, "settling"
        self.error = None
        self.events = [[] for _ in rows]
        self.settling = [[] for _ in rows]
        self.counts = [
            dict(callback_invocations=0, raw_header_count=0, raw_contact_point_count=0)
            for _ in rows
        ]
        if reference_windows:
            assert contact_windows is not None  # validated above, before native construction
            self.windows = [contact_windows(r["parameters"], r["candidate"]) for r in rows]
        else:
            self.windows = [[] for _ in rows]
        self.active = [True] * len(rows)
        configured = [[] for _ in rows]
        for prim in Usd.PrimRange(env.stage.GetPseudoRoot()):
            path = str(prim.GetPath())
            index = environment_for_actor(path, len(rows))
            if (
                index is not None
                and "InstalledRobots" in path
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                api.CreateThresholdAttr().Set(0.0)
                configured[index].append(
                    dict(
                        path=path,
                        threshold=float(api.GetThresholdAttr().Get()),
                        api_present=bool(prim.HasAPI(PhysxSchema.PhysxContactReportAPI)),
                    )
                )
        for index, row in enumerate(rows):
            required = [
                env.robot_paths[index][arm] + "/" + finger
                for arm in ("left", "back")
                for finger in ("gripper_link1", "gripper_link2")
            ]
            valid = bool(
                set(required) <= {p["path"] for p in configured[index]}
                and all(p["api_present"] and p["threshold"] == 0 for p in configured[index])
            )
            save(
                row["output"] / "contact-reporting.json",
                dict(
                    schema="tower-v3-contact-reporting-configuration-v1",
                    env_id=index,
                    configured_paths=configured[index],
                    required_fingertip_paths=required,
                    verified=valid,
                    threshold_stage_units=0.0,
                    sampling="PhysX contact callback",
                    authoring_phase="after initialization, before settling/reference",
                ),
            )
            if not valid:
                raise ValueError(f"Contact reporting readback failed in env{index}")
        self.subscription = (
            omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
                self.contact
            )
        )

    def contact(self, headers, points):
        from pxr import PhysicsSchemaTools  # pyright: ignore[reportMissingImports]

        seen = set()
        try:
            for header in headers:
                pair = [
                    str(PhysicsSchemaTools.intToSdfPath(p)) for p in (header.actor0, header.actor1)
                ]
                index = route_contact(pair, len(self.rows))
                if index is None or not self.active[index]:
                    continue
                seen.add(index)
                self.counts[index]["raw_header_count"] += 1
                self.counts[index]["raw_contact_point_count"] += int(header.num_contact_data)
                if not header.num_contact_data:
                    continue
                robots = [p for p in pair if "InstalledRobots" in p]
                if not robots or all(p.endswith(("/base_link", "/world")) for p in robots):
                    continue
                stamp = float(self.env.elapsed[index])
                if self.phase == "settling":
                    if any("/Blocks/" in p for p in pair):
                        self.settling[index].append(dict(time_s=stamp, actors=pair))
                    continue
                matches = [w for w in self.windows[index] if self.contact_matches(pair, stamp, w)]
                for window in matches:
                    window["event_count"] += 1
                    window["observed"] = True
                self.events[index].append(
                    dict(
                        time_s=stamp,
                        actors=pair,
                        planned_fingertip_contact=bool(matches),
                        allowed_block_names=[w["block_name"] for w in matches],
                        separation_stage_units=min(
                            float(points[k].separation)
                            for k in range(
                                header.contact_data_offset,
                                header.contact_data_offset + header.num_contact_data,
                            )
                        ),
                    )
                )
            for index in seen:
                self.counts[index]["callback_invocations"] += 1
        except Exception as exc:  # noqa: BLE001 -- latch native callback/delivery failures
            # Callback exceptions can be swallowed by PhysX; the driver checks this
            # latched error after every fetch_results and invalidates the batch.
            self.error = str(exc)

    def begin_motion(self):
        self.phase = "motion"
        self.counts = [
            dict(callback_invocations=0, raw_header_count=0, raw_contact_point_count=0)
            for _ in self.rows
        ]

    def finish(self, index, checked):
        self.active[index] = False
        row = self.rows[index]
        events, windows = self.events[index], self.windows[index]
        missing = [
            dict(arm=w["arm"], block_name=w["block_name"]) for w in windows if not w["observed"]
        ]
        forbidden = sum(not e["planned_fingertip_contact"] for e in events)
        valid = bool(
            not self.error
            and all(self.counts[index].values())
            and len(windows) == 4
            and not missing
        )
        complete = checked == row["ticks"]
        passed = valid and complete and forbidden == 0
        report = dict(
            schema="tower-v3-reference-executed-collision-v2",
            env_id=index,
            passed=passed,
            status="passed" if passed else "inconclusive_or_collision",
            physics_hz=400,
            physics_steps_checked=checked,
            expected_physics_steps=row["ticks"],
            full_duration_s=checked / 400,
            coverage_complete=complete,
            tick_coverage_complete=complete,
            settling_contacts_checked=True,
            instrumentation_valid=valid,
            events=events,
            event_count=len(events),
            forbidden_count=forbidden,
            expected_fingertip_contact_coverage=windows,
            missing_expected_contacts=missing,
            reporting_configuration_sha256=sha(row["output"] / "contact-reporting.json"),
            source_episode_params_sha256=sha(row["plan"] / "episode-params.json"),
            contact_window_definition="source grasp arrival through paired release departure, inclusive",
            isolation_error=self.error,
            **self.counts[index],
        )
        save(row["output"] / "collision-validation.json", report)
        return report

    def close(self):
        self.subscription = None


class ScreenContacts(BatchContacts):
    def contact(self, headers, points):
        from pxr import PhysicsSchemaTools  # pyright: ignore[reportMissingImports]

        try:
            for header in headers:
                pair = [
                    str(PhysicsSchemaTools.intToSdfPath(p)) for p in (header.actor0, header.actor1)
                ]
                index = route_contact(pair, len(self.rows))
                if index is None:
                    continue
                self.counts[index]["raw_header_count"] += 1
                self.counts[index]["raw_contact_point_count"] += int(header.num_contact_data)
                if (
                    header.num_contact_data
                    and any("/Blocks/" in p for p in pair)
                    and any("InstalledRobots" in p for p in pair)
                ):
                    self.settling[index].append(
                        dict(
                            actors=pair,
                            time_s=float(self.env.simulation_time),
                            count=int(header.num_contact_data),
                        )
                    )
        except Exception as exc:  # noqa: BLE001 -- latch native callback/delivery failures
            self.error = str(exc)
