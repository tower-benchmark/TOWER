"""Shared command/policy lifecycle, extracted from the frozen batch loop.

Explicit policy hooks; no module patches or environment-selected behavior. Construction
and preflight are owned by the maintainer adapter; teardown follows collection.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import cameras as probe
from . import control, environment
from . import observation_validity as validity
from . import serial_camera_reference as source_reference
from .camera_guard import TerminalGuard
from .camera_quality import CameraQuality
from .contacts import make_contacts
from .evidence import BatchEvidence, CameraArchive, settling_metrics
from .fast_early_stop import CachedComputation
from .observation import encoder_observation
from .resources import save, sha
from .scoring import BenchScorer


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass
class ReplayFactories:
    contacts: Any = make_contacts
    guard: Any = lambda env, output: TerminalGuard(environment.wrist_guard, env, output)
    camera: Any = probe.make_camera
    serial: Any = source_reference.SerialCameraReference
    observation: Any = encoder_observation
    scorer: Any = BenchScorer


def validate_trajectory(context, rows):
    """Validate each environment's frozen bootstrap extent after delay/J2 conversion.

    Initial-only replay rows cover 20 ticks; CommandQueue.target checks the rest.
    """
    for index, row in enumerate(rows):
        encoder, physical = context.compensation.command_targets(
            row["times"], row["actions"], context.delays, row["ticks"]
        )
        control.validate_physical_targets(physical, context.physical_command_limits(index))
        row.update(encoder=encoder, physical=physical)


def settle(rows, env, contacts):
    evidence = BatchEvidence(env, len(rows))
    env.elapsed[:] = 0.0
    evidence.append(env.elapsed)
    for tick in range(1, 401):
        env.physics.simulate(1 / 400, (tick - 1) / 400)
        env.physics.fetch_results()
        if contacts.error:
            raise ValueError("Contact instrumentation/isolation failed: " + contacts.error)
        env.simulation_time = tick / 400
        env.elapsed[:] = tick / 400
        if tick % 20 == 0:
            evidence.append(env.elapsed)
    for index, row in enumerate(rows):
        directory = row["output"]
        values = evidence.write(index, directory)
        before, late, after = values["blocks"][0], values["blocks"][15], values["blocks"][-1]
        save(
            directory / "settling.json",
            dict(
                schema="tower-v3-generator-settling-v1",
                physics_hz=400,
                required_ticks=400,
                ticks_completed=400,
                duration_s=1.0,
                late_sample_time_s=0.75,
                source_initial_state_sha256=sha(row["plan"] / "reference-initial-state.json"),
                trace_sha256=sha(directory / "episode/states.hdf5"),
                before_block_poses=before,
                late_block_poses=late,
                after_block_poses=after,
                **settling_metrics(before, late, after),
            ),
        )
        save(directory / "initial-contacts.json", contacts.settling[index])
        save(
            directory / "settling-contact-coverage.json",
            dict(
                physics_steps_checked=400,
                physics_hz=400,
                duration_s=1.0,
                callback_error=contacts.error,
                coverage_complete=True,
                reporting_configuration_sha256=sha(directory / "contact-reporting.json"),
                **contacts.counts[index],
            ),
        )


def failure_terminal_tick(tick):
    if type(tick) is not int or tick < 0:
        raise ValueError("Failure clock must be a nonnegative integer physics tick")
    return 20 if tick == 0 else tick


def initial_image_check(pixels):
    """The known illuminated starting scene cannot have four entirely zero views."""
    pixels = np.asarray(pixels)
    if pixels.ndim != 5 or pixels.shape[1:] != (4, 240, 320, 3) or pixels.dtype != np.uint8:
        raise ValueError("Invalid initial batch camera shape/dtype")
    blank = [i for i, views in enumerate(pixels) if not np.any(views)]
    return dict(
        passed=not blank,
        all_zero_environment_indices=blank,
        scope="Known initial scene before policy inference; later physical occlusion remains separately verified",
    )


def terminal_evidence_recorded(*, ended, terminal_tick, state_time, camera_time):
    """A planned failure tail is never evidence that physics reached that time."""
    return bool(
        ended
        and type(terminal_tick) is int
        and camera_time is not None
        and state_time == camera_time == terminal_tick / 400
    )


class SelectedFrame:
    """Expose one owned batch row to the existing lossless model transport."""

    def __init__(self, names):
        self.names = names
        self.pixels: Any = None
        self.validator: Any = None

    def validate_observation(self, pixels, stamp, quality):
        if self.validator is None:
            return dict(valid=False, reason="No actual synchronized delivery validator installed")
        return self.validator(pixels, stamp, quality)

    def observe(self):
        if self.pixels is None:
            raise ValueError("No independently assigned frame available")
        return dict(rgb=self.pixels[None])


def execute(
    context,
    rows,
    task,
    *,
    metrics,
    source_collector,
    observed_validator,
    expert_sha256=None,
    factories=None,
    policy=None,
    evaluation=None,
):
    if not 1 <= context.env.n <= 10 or len(rows) != context.env.n:
        raise ValueError("Replay requires 1..10 environments and one row per environment")
    from nero_kinematics.profiles import ROOT as profile_root
    from PIL import Image

    command_provenance = dict(
        profile_sha256=sha(profile_root / "command.json"),
        ik_bounds=None,
        ik_bounds_scope="joint_absolute replay uses packaged command bounds; no IK service/client "
        "or TOWER_EVAL_IK_BOUNDS override is read",
    )
    factories = factories or ReplayFactories()
    env, app, compensation = context.env, context.app, context.compensation
    validate_trajectory(context, rows)
    quality = CameraQuality()
    package = dict(package_digest=expert_sha256, checkpoint_sha256=expert_sha256)
    identity = dict(
        kind="scripted_reference_command_replay",
        package_digest=expert_sha256,
        neural_inference=False,
    )
    if policy is None:
        if expert_sha256 is None or evaluation is not None:
            raise ValueError("Command replay requires an expert digest and cannot be formal")
        save(context.output / "policy-health.json", dict(ready=True, identity=identity))
    else:
        package = policy.package
        try:
            health = policy.start(context.output)
        except BaseException:
            policy.close()
            raise
        command_provenance.update(
            ik_bounds=policy.solver.identity,
            ik_bounds_scope="native policy IK; command bounds separate from observation tolerance",
        )
    contacts = factories.contacts(env, rows)
    try:
        # These traces are preserved separately from all policy-time evidence.
        original = [row["output"] for row in rows]
        for row in rows:
            row["output"] = row["output"] / "settling"
            row["output"].mkdir()
            # Contact configuration was actually authored and read back at root.
            shutil.copy2(
                row["output"].parent / "contact-reporting.json",
                row["output"] / "contact-reporting.json",
            )
        settle(rows, env, contacts)
        for row, output in zip(rows, original, strict=True):
            row["output"] = output
            report = json.loads((output / "settling/settling.json").read_text())
            if not report["passed"] or contacts.settling[row["env_id"]]:
                raise ValueError("Policy scene failed source settling/contact checks")
    finally:
        contacts.close()
    env.elapsed[:] = 0.0
    env.simulation_time = 1.0
    settings = context.resources.read("settings")
    sim = SimpleNamespace(
        app=app, env=env, settings=settings, read=lambda: env.read()[0], read_batch=env.read
    )
    capture_directory = context.output / "capture"
    capture_directory.mkdir()
    camera_guard = factories.guard(env, capture_directory / "near-plane-terminals")
    before_creation = env.read()
    camera = factories.camera(
        sim, capture_directory, resources=context.resources, near_guard=camera_guard.check
    )
    probe.assert_unchanged(before_creation, env.read())
    maps = {}
    for name, role in camera.names.items():
        targets = list(
            map(
                str,
                env.stage.GetPrimAtPath(camera.products[role].path)
                .GetRelationship("camera")
                .GetTargets(),
            )
        )
        if targets[: env.n] != camera.batch_camera_paths[role]:
            raise ValueError("Tiled camera environment order changed")
        maps[name] = targets
    evidence = BatchEvidence(env, len(rows))
    evidence.append(env.elapsed)
    if policy is not None:
        policy.initialize(rows, camera.names, health)
    frames, archives = [], []
    queues, commands, last_encoder, last_physical = [], [], [], []
    last_velocity = np.zeros((len(rows), 16))
    observed_joint_checks = [0] * len(rows)
    live_limits = [context.physical_command_limits(i) for i in range(len(rows))]
    delays = context.delays
    for row in rows:
        encoder, physical = (
            np.array(row["initial"]["qpos"]),
            np.array(row["initial"]["physical_qpos"]),
        )
        queues.append(
            control.CommandQueue(
                encoder,
                delays,
                compensation.physical_from_encoder16,
                context.physical_command_limits(row["env_id"]),
            )
        )
        commands.append(
            [dict(time=0.0, encoder=encoder, physical=physical, arm_velocity_target=np.zeros(16))]
        )
        last_encoder.append(encoder.copy())
        last_physical.append(physical.copy())
        frame = SelectedFrame(camera.names)
        frames.append(frame)
        archives.append(CameraArchive(frame, row["output"] / "camera-frames.hdf5", quality=quality))
        (row["output"] / "chunks").mkdir()
    tick, frame_number, render_checks = 0, 1, 0
    terminal_ticks: list[Any] = [None] * len(rows)
    failures: list[Any] = [None] * len(rows)
    successes: list[Any] = [None] * len(rows)
    shared_cache = CachedComputation()
    scorers = [factories.scorer(task, mode="fast", computation=shared_cache) for _ in rows]
    early_stop = scorers if task.early_success_stop else None
    ended = [False] * len(rows)
    duration_ticks = task.duration_ticks
    assignment = {}
    source_parity = {}
    serial_camera = factories.serial(camera, guard=camera_guard)
    guards = [validity.DeliveryGuard() for _ in rows]
    visibility_references = {0: serial_camera}
    timings = dict(
        camera_observe_s=0.0,
        camera_archive_s=0.0,
        policy_requests_s=0.0,
        controller_s=0.0,
        physics_s=0.0,
    )

    def validate_delivery(index, pixels, stamp, quality):
        def independent():
            if index not in visibility_references:
                visibility_references[index] = factories.serial(camera, index, guard=camera_guard)
            before = env.read()
            reference, receipt = visibility_references[index].observe()
            after = env.read()
            probe.assert_unchanged(before, after)
            directory = rows[index]["output"] / "visibility"
            directory.mkdir(exist_ok=True)
            stem = directory / f"{round(stamp * 400):08d}"
            np.savez_compressed(str(stem) + ".npz", observed=pixels, reference=reference)
            receipt.update(
                physical_state_preserved=True,
                evidence_path=str(stem) + ".npz",
                evidence_sha256=sha(str(stem) + ".npz"),
            )
            state_path = Path(str(stem) + ".json")
            save(
                state_path,
                dict(
                    schema="tower-v3-camera-state-preservation-v1",
                    time_s=stamp,
                    time_coordinate="episode_relative_seconds",
                    environment_index=index,
                    before=source_reference.physical_snapshot(before),
                    after=source_reference.physical_snapshot(after),
                    pixels=dict(path=receipt["evidence_path"], sha256=receipt["evidence_sha256"]),
                ),
            )
            receipt["state_evidence"] = dict(path=str(state_path.resolve()), sha256=sha(state_path))
            return reference, receipt

        return guards[index].validate(
            pixels,
            stamp,
            quality,
            render_step=camera.verified_capture_steps,
            synchronized=True,
            paused=True,
            delta_time_s=0.0,
            independent_reference=independent,
        ) | dict(renderer_delivery=camera.last_render_delivery)

    for index, frame in enumerate(frames):
        frame.validator = lambda pixels, stamp, quality, i=index: validate_delivery(
            i, pixels, stamp, quality
        )

    def observe():
        nonlocal render_checks
        before = env.read()
        previous_steps = camera.verified_capture_steps
        packed = np.asarray(camera.observe()["rgb"]).copy()
        if camera.verified_capture_steps <= previous_steps:
            raise ValueError("No new synchronous render completed for observation")
        probe.assert_unchanged(before, env.read())
        if packed.shape != (env.n, 4, 240, 320, 3) or packed.dtype != np.uint8:
            raise ValueError("Invalid batch observation shape/dtype")
        render_checks += 1
        return packed

    def store_camera(selected, at_tick):
        started = time.perf_counter()
        packed = metrics.call("camera.observe", observe)
        timings["camera_observe_s"] += time.perf_counter() - started
        started = time.perf_counter()
        selected = [i for i in selected if not ended[i]]
        rejected = camera_guard.consume(at_tick)
        newly_rejected = [i for i in rejected if not ended[i]]
        for index in newly_rejected:
            detail = rejected[index]
            if failures[index] is None:
                failures[index] = dict(
                    failure_tick=at_tick,
                    reason=detail["reason"],
                    category="camera_near_plane_guard",
                    rejected_action_applied=False,
                    rejected_observation_sent_to_policy=False,
                    camera_guard=detail,
                )
                successes[index] = None
            else:
                failures[index]["camera_guard"] = detail
            terminal_ticks[index] = at_tick
        selected = sorted(set(selected) | set(newly_rejected))
        record_state(newly_rejected, at_tick)
        for index in selected:
            if (
                len(archives[index].times)
                and round(float(archives[index].times[-1]) * 400) == at_tick
            ):
                continue
            frames[index].pixels = packed[index].copy()
            archives[index].append(at_tick / 400, encode_policy=False)
            if policy is not None:
                policy.append_camera(
                    index,
                    archives[index],
                    at_tick,
                    rejected=index in newly_rejected,
                    terminal=failures[index] is not None or terminal_ticks[index] == at_tick,
                )
        timings["camera_archive_s"] += time.perf_counter() - started
        for index in newly_rejected:
            # Synchronized terminal RGB is evidence, not an eligible model input.
            archives[index]._latest_pixels = archives[index]._latest_encoded = None
            archives[index].file.attrs["terminal_policy_observation_rejected"] = True
            archives[index].file.attrs["terminal_near_plane_mm"] = 10.0
            finalize_environment(index, at_tick)

    @metrics.decorated("evidence.record_state")
    def record_state(selected, at_tick):
        selected = [
            i
            for i in selected
            if not ended[i] and round(evidence.records[i][-1]["time"] * 400) < at_tick
        ]
        if selected:
            evidence.append(np.full(env.n, at_tick / 400), selected)
            for index in selected:
                commands[index].append(
                    dict(
                        time=at_tick / 400,
                        encoder=last_encoder[index].copy(),
                        physical=last_physical[index].copy(),
                        arm_velocity_target=last_velocity[index].copy(),
                    )
                )

    @metrics.decorated("camera.assignment_proof")
    def prove_assignment(label):
        packed = metrics.call("camera.observe", observe)
        before = env.read()
        ordinary, reference_maps = probe.ordinary_references(camera)
        probe.assert_unchanged(before, env.read())
        comparison = probe.compare_assignment(packed, ordinary)
        for index, row in enumerate(rows):
            for c, name in enumerate(camera.names):
                Image.fromarray(packed[index, c]).save(row["output"] / f"{label}-tiled-{name}.png")
                Image.fromarray(ordinary[index, c]).save(
                    row["output"] / f"{label}-ordinary-{name}.png"
                )
        independent, source_receipt = serial_camera.observe()
        after = env.read()
        probe.assert_unchanged(before, after)
        source_comparison = source_reference.compare_pixels(packed[0], independent)
        for c, name in enumerate(camera.names):
            Image.fromarray(independent[c]).save(
                rows[0]["output"] / f"{label}-serial-source-{name}.png"
            )
        state_path = context.output / f"{label}-camera-physical-state.json"
        save(
            state_path,
            dict(
                before=source_reference.physical_snapshot(before),
                after=source_reference.physical_snapshot(after),
                exactly_unchanged=True,
            ),
        )
        seeds = [row["candidate"]["seed"] for row in rows]
        candidate_hashes = [digest(row["candidate"]) for row in rows]
        assignment[label] = dict(
            comparison=comparison,
            reference_maps=reference_maps,
            time_s=tick / 400,
            seeds=seeds,
            candidate_sha256=candidate_hashes,
            physical_state_path=str(state_path.resolve()),
            physical_state_sha256=sha(state_path),
        )
        source_parity[label] = dict(
            **source_receipt,
            comparison=source_comparison,
            seeds=seeds,
            time_s=tick / 400,
            representative_seed=seeds[0],
            physical_state_path=str(state_path.resolve()),
            physical_state_sha256=sha(state_path),
        )
        save(context.output / "camera-assignment.json", assignment)
        save(context.output / "source-camera-parity.json", source_parity)
        if label == "initial":
            initial_check = initial_image_check(packed)
            save(context.output / "initial-image-check.json", initial_check)
            if not initial_check["passed"]:
                raise ValueError(
                    "Initial scene has four all-zero camera views; renderer startup is invalid: "
                    + str(initial_check["all_zero_environment_indices"])
                )
        if not comparison["passed"] or not source_comparison["passed"]:
            raise ValueError("Moving camera assignment/source fidelity failed: " + label)

    def actual_sources():
        return context.collect_sources(source_collector)

    @metrics.decorated("evidence.finalization")
    def finalize_environment(index, at_tick):
        ended[index] = True
        if policy is not None:
            policy.finalize(index)
        save(
            rows[index]["output"] / "observed-joint-bounds.json",
            dict(
                checked_physics_ticks=observed_joint_checks[index],
                frequency_hz=400,
                observed_tracking_tolerance_rad=control.OBSERVED_TRACKING_TOLERANCE_RAD,
                **command_provenance,
                passed=not (
                    failures[index]
                    and failures[index].get("category") == "observed_joint_limit_violation"
                ),
            ),
        )
        # Per-environment terminal collection preserves the frozen formal path.
        sources = actual_sources()
        save(rows[index]["output"] / "runtime-sources.json", sources)
        if evaluation is not None:
            evaluation.finalize_trial(
                index,
                evidence,
                commands[index],
                archives[index],
                camera.report(),
                sources,
                tick=at_tick,
                failure=failures[index],
                success=successes[index],
            )

    from collections import deque

    velocity_history = {}
    velocity_columns = {}
    for arm, robot in enumerate(env.robots):
        ids, _ = env._commands(np.stack(last_physical), arm)
        gripper_columns = [k for k, i in enumerate(ids) if robot.dof_names[i].startswith("gripper")]
        arm_columns = [list(ids).index(robot.dof_names.index(f"joint{j + 1}")) for j in range(7)]
        velocity_columns[arm] = (ids.copy(), gripper_columns, arm_columns)
    complete = False
    original_error = None
    try:
        initial_pixels = observe()
        # Preserve actual startup failures before a bounded renderer-only drain.
        # No physical step, camera parameter, controller or acceptance threshold
        # is changed; the initial clock and every physical array must match.
        startup_rows = []
        for attempt in range(3):
            check = initial_image_check(initial_pixels)
            from pxr import UsdGeom  # pyright: ignore[reportMissingImports]

            paths = set(camera.reference_paths.values())
            paths.update(path for values in camera.batch_camera_paths.values() for path in values)
            paths.update(c.prim_path for group in env.cameras for c in group.values())

            def attributes(path):
                prim = env.stage.GetPrimAtPath(path)
                return dict(
                    path=str(path),
                    schemas=list(prim.GetAppliedSchemas()),
                    attributes={a.GetName(): str(a.Get()) for a in prim.GetAttributes()},
                    world_transform=np.asarray(
                        UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                    ).tolist(),
                )

            startup_rows.append(
                dict(
                    attempt=attempt,
                    check=check,
                    cameras=[attributes(p) for p in sorted(paths)],
                    controls={
                        k: camera.controls.get(k) for k in camera.profile["renderer_controls"]
                    },
                )
            )
            save(context.output / f"startup-camera-readback-{attempt}.json", startup_rows[-1])
            if check["passed"]:
                break
            np.savez_compressed(
                context.output / f"startup-camera-failed-{attempt}.npz", rgb=initial_pixels
            )
            if attempt == 2:
                break
            before_drain = env.read()
            for _ in range(32):
                camera._step(6)
            initial_pixels = observe()
            probe.assert_unchanged(before_drain, env.read())
        save(
            context.output / "startup-camera-drain.json",
            dict(
                attempts=len(startup_rows),
                passed=check["passed"],
                physics_unchanged=True,
                parameters_unchanged=True,
            ),
        )
        prove_assignment("initial")
        store_camera(range(env.n), 0)
        metrics.barrier(env.read, probe.assert_unchanged)
        camera_guard.enabled = True
        while tick < duration_ticks and not all(ended):
            metrics.tick(tick)
            if tick % 640 == 0:
                with metrics.span("policy.total"):
                    inference_started = time.perf_counter()
                    observations = {
                        index: factories.observation(
                            env.read()[index].qpos,
                            compensation,
                            env.lula.tcp_rotation,
                            env.lula.tcp_offset,
                        )
                        for index in range(len(rows))
                        if not ended[index] and failures[index] is None
                    }
                    pending = policy.prefetch(observations, archives, tick) if policy else None
                    for index, row in enumerate(rows):
                        if index not in observations:
                            continue
                        observation = observations[index]
                        if policy is None:
                            metrics.emit(
                                "command_replay_request",
                                request_id=metrics.request_id(row["candidate"]["seed"], tick),
                            )
                        try:
                            if policy is not None:
                                predicted = policy.prediction(pending, index, observation, tick)
                                if evaluation is not None:
                                    evaluation.verify_prediction(predicted["identity"])
                            else:
                                stamp = tick / 400 + 0.1 * np.arange(1, 33)
                                commanded = np.stack(
                                    [
                                        np.interp(stamp, row["times"], row["actions"][:, j])
                                        for j in range(16)
                                    ],
                                    axis=1,
                                )
                                predicted = dict(
                                    mode="joint_absolute",
                                    coordinate_frame="encoder_arm_base",
                                    j8_actuator_mask=[False, False],
                                    control_hz=10,
                                    execute_steps=16,
                                    joint_encoder_targets=np.stack(
                                        (commanded[:, :7], commanded[:, 8:15]), axis=1
                                    ).tolist(),
                                    gripper_width_m=commanded[:, [7, 15]].tolist(),
                                    identity=identity,
                                    reference_source_sha256=expert_sha256,
                                    neural_inference=False,
                                )
                            save(
                                row["output"] / "chunks" / f"{tick // 640:05d}.json",
                                dict(
                                    observation_time_s=tick / 400,
                                    observation=observation,
                                    response=predicted,
                                ),
                            )
                            endpoints, diagnostics = metrics.call(
                                "policy.ik",
                                control.encoder_endpoints,
                                predicted,
                                observation,
                                solver=policy.solver if policy else None,
                            )
                            save(
                                row["output"] / "chunks" / f"{tick // 640:05d}-ik.json", diagnostics
                            )
                            metrics.call(
                                "policy.enqueue", queues[index].enqueue, tick / 400, endpoints
                            )
                        except control.PolicyActionError as exc:
                            if hasattr(exc, "diagnostics"):
                                save(
                                    row["output"]
                                    / "chunks"
                                    / f"{tick // 640:05d}-ik-rejected.json",
                                    exc.diagnostics,
                                )
                            failures[index] = dict(
                                failure_tick=tick, reason=str(exc), rejected_action_applied=False
                            )
                            terminal_ticks[index] = failure_terminal_tick(tick)
                    timings["policy_requests_s"] += time.perf_counter() - inference_started
            controller_started = time.perf_counter()
            for index in range(env.n):
                if ended[index] or failures[index] is not None:
                    continue
                try:
                    encoder, physical = metrics.call(
                        "controller.queue_target", queues[index].target, tick
                    )
                    last_encoder[index], last_physical[index] = encoder, physical
                except control.PolicyActionError as exc:
                    failures[index] = dict(
                        failure_tick=tick, reason=str(exc), rejected_action_applied=False
                    )
                    terminal_ticks[index] = failure_terminal_tick(tick)
            timings["controller_s"] += time.perf_counter() - controller_started
            finished_now = [i for i in range(env.n) if not ended[i] and terminal_ticks[i] == tick]
            if finished_now:
                record_state(finished_now, tick)
                store_camera(finished_now, tick)
                for index in finished_now:
                    if not ended[index]:
                        finalize_environment(index, tick)
            if all(ended):
                break
            targets = np.stack(last_physical)
            for arm, robot in enumerate(env.robots):
                ids, target = metrics.call(
                    "controller.command_mapping", env._commands, targets, arm
                )
                metrics.call(
                    "controller.position_targets",
                    robot.set_joint_position_targets,
                    target,
                    joint_indices=ids,
                )
                history = velocity_history.setdefault(arm, deque(maxlen=17))
                history.append(target.copy())
                velocity = (history[-1] - history[0]) * (400.0 / max(len(history) - 1, 1))
                expected_ids, gripper_columns, arm_columns = velocity_columns[arm]
                if not np.array_equal(ids, expected_ids):
                    raise ValueError("Articulation command order changed")
                velocity[:, gripper_columns] = 0.0
                metrics.call(
                    "controller.velocity_targets",
                    robot.set_joint_velocity_targets,
                    velocity,
                    joint_indices=ids,
                )
                last_velocity[:, arm * 8 : arm * 8 + 7] = velocity[:, arm_columns]
            physics_started = time.perf_counter()
            metrics.call("physics.simulate", env.physics.simulate, 1 / 400, 1.0 + tick / 400)
            metrics.call("physics.fetch_results", env.physics.fetch_results)
            timings["physics_s"] += time.perf_counter() - physics_started
            tick += 1
            env.elapsed[:] = tick / 400
            env.simulation_time = 1.0 + tick / 400
            env.previous[:] = targets
            selected = [i for i in range(env.n) if not ended[i]]
            for index, state in enumerate(metrics.call("state.readback", env.read)):
                if ended[index] or failures[index] is not None:
                    continue
                try:
                    metrics.call(
                        "controller.observed_joint_validation",
                        observed_validator,
                        state.qpos,
                        compensation,
                        live_limits[index],
                    )
                    observed_joint_checks[index] += 1
                except control.PolicyActionError as exc:
                    failures[index] = dict(
                        failure_tick=tick,
                        reason=str(exc),
                        rejected_action_applied=False,
                        category="observed_joint_limit_violation",
                        observed_physical_qpos=state.qpos.tolist(),
                    )
                    terminal_ticks[index] = tick
            if tick % 20 == 0 or tick == duration_ticks:
                record_state(selected, tick)
            if early_stop is not None and tick >= 4000 and tick % 400 == 0:
                for index in selected:
                    if failures[index] is None:
                        recording = scorers[index].recording(
                            rows[index]["candidate"]["episode_spec"],
                            evidence.records[index],
                            env.finger_geometry,
                        )
                        successes[index] = metrics.call(
                            "scoring.early_stop", early_stop[index].check, recording, tick
                        )
                        if successes[index] is not None:
                            terminal_ticks[index] = tick
            terminal = [i for i in selected if terminal_ticks[i] == tick or tick == duration_ticks]
            if (
                tick >= int(np.ceil(frame_number * 400 / 30 - 1e-10))
                or tick % 640 == 0
                or tick == duration_ticks
            ):
                store_camera(selected, tick)
                frame_number += 1
            if terminal:
                record_state(terminal, tick)
                store_camera(terminal, tick)
                for index in terminal:
                    if not ended[index]:
                        terminal_ticks[index] = tick
                        finalize_environment(index, tick)
            if tick == 640:
                prove_assignment("moving")
            if tick % 400 == 0:
                save(
                    context.output / "worker-status.json",
                    dict(
                        stage=policy.stage if policy else "reference_command_replay",
                        simulated_seconds=tick / 400,
                        ended=ended,
                        serialized_neural_inference=policy is not None,
                        concurrent_request_limit=1 if policy else min(16, env.n),
                    ),
                )
        metrics.end_loop()
        prove_assignment("batch_end")
        complete = True
    except BaseException as exc:  # noqa: BLE001 -- preserve evidence and primary errors
        # Only this loop's own failure: an exception the caller is merely handling
        # must not suppress cleanup errors below.
        original_error = exc
        raise
    finally:
        cleanup_errors = []
        sources = {}
        if policy is not None:
            try:
                policy.close()
            except BaseException as exc:  # noqa: BLE001 -- preserve evidence and primary errors
                cleanup_errors.append(f"policy close: {type(exc).__name__}: {exc}")
        try:
            sources = actual_sources()
            save(context.output / "runtime-sources-final.json", sources)
            if complete:
                qualify_capture(
                    env.n, camera.report(), assignment, source_parity, sources, context.output
                )
        except BaseException as exc:  # noqa: BLE001 -- close capture even on rejected evidence
            cleanup_errors.append(f"source finalization: {type(exc).__name__}: {exc}")
        for index, row in enumerate(rows):
            output = row["output"]
            own_complete = False
            if evaluation is not None:
                try:
                    evaluation.invalidate_unfinished(
                        index,
                        evidence,
                        commands[index],
                        archives[index],
                        camera.report(),
                        sources,
                        reason="Shared batch stopped before this environment reached a committed terminal",
                    )
                except BaseException as exc:  # noqa: BLE001 -- preserve evidence and primary errors
                    cleanup_errors.append(
                        f"env{index} formal finalization: {type(exc).__name__}: {exc}"
                    )
                continue
            try:
                evidence.write(index, output)
                save(output / "commands.json", commands[index])
                save(
                    output / "camera-readback.json",
                    dict(
                        **camera.report(),
                        env_id=index,
                        same_physics_state_checks=render_checks,
                        assignment_receipt_sha256=(
                            sha(context.output / "camera-assignment.json")
                            if (context.output / "camera-assignment.json").exists()
                            else None
                        ),
                    ),
                )
                failure = failures[index]
                actual_time = evidence.records[index][-1]["time"]
                terminal_recorded = terminal_evidence_recorded(
                    ended=ended[index],
                    terminal_tick=terminal_ticks[index],
                    state_time=actual_time,
                    camera_time=float(archives[index].times[-1])
                    if len(archives[index].times)
                    else None,
                )
                if failure is not None:
                    save(
                        output / "policy-action-failure.json",
                        dict(
                            **failure,
                            failure_time_s=failure["failure_tick"] / 400,
                            terminal_evidence_complete=terminal_recorded,
                            terminal_time_s=actual_time if terminal_recorded else None,
                            actual_last_state_time_s=actual_time,
                            held_evidence_tail_s=(actual_time - failure["failure_tick"] / 400)
                            if terminal_recorded
                            else None,
                        ),
                    )
                own_complete = bool(
                    complete
                    and terminal_recorded
                    and len(evidence.records[index]) >= 2
                    and len(archives[index].times) >= 2
                )
                success_sha = None
                if successes[index] is not None:
                    names = (
                        "episode_spec.json",
                        "episode/states.hdf5",
                        "finger-link-poses.npz",
                        "finger-feedback.npz",
                        "finger_geometry.json",
                    )
                    save(
                        output / "task-success.json",
                        dict(
                            successes[index],
                            diagnostic_only=True,
                            seed=row["candidate"]["seed"],
                            source_hashes={n: sha(output / n) for n in names},
                        ),
                    )
                    success_sha = sha(output / "task-success.json")
                save(
                    output / "status.json",
                    dict(
                        status="diagnostic_prefix_complete"
                        if own_complete
                        else "infrastructure_failure",
                        valid_trial=False,
                        **getattr(context, "public_provenance", {}),
                        diagnostic_only=True,
                        evidence_complete=own_complete,
                        task_success=successes[index] is not None,
                        success_evidence_sha256=success_sha,
                        policy_action_failure=failure is not None,
                        checkpoint_package_digest=package["package_digest"],
                        checkpoint_sha256=package["checkpoint_sha256"],
                        simulated_seconds=actual_time,
                        returncode=0 if own_complete else 1,
                        env_id=index,
                        seed=row["candidate"]["seed"],
                    ),
                )
            except BaseException as exc:  # noqa: BLE001 -- preserve evidence, then fail cleanup
                own_complete = False
                cleanup_errors.append(f"env{index} evidence: {type(exc).__name__}: {exc}")
            finally:
                try:
                    archives[index].close(complete=own_complete)
                except BaseException as exc:  # noqa: BLE001 -- preserve evidence, then fail cleanup
                    cleanup_errors.append(f"env{index} archive close: {type(exc).__name__}: {exc}")
        # Detachment can invalidate SDK tensor handles: ALL physical reads and
        # evidence writes must finish before releasing capture resources.
        try:
            camera.close()
        except BaseException as exc:  # noqa: BLE001 -- preserve evidence, then fail cleanup
            cleanup_errors.append(f"capture close: {type(exc).__name__}: {exc}")
        complete = bool(complete and not cleanup_errors)
        try:
            save_batch_results(
                context,
                complete,
                env,
                tick,
                package,
                assignment,
                source_parity,
                cleanup_errors,
                timings,
                policy=policy,
                evaluation=evaluation,
            )
        except BaseException as exc:  # noqa: BLE001 -- retain the primary loop failure
            cleanup_errors.append(f"batch results: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            detail = "Batch diagnostic evidence/cleanup failed: " + str(cleanup_errors)
            if original_error is not None:
                original_error.add_note(detail)
            else:
                raise RuntimeError(detail)
    # Like the frozen wrapper, publish a passing receipt only after execute cleanup
    # succeeds. All counted physical readbacks still precede camera teardown.
    if complete:
        context.finalize_drive_receipt()
    return 0 if complete else 1


def save_batch_results(
    context,
    complete,
    env,
    tick,
    package,
    assignment,
    source_parity,
    cleanup_errors,
    timings,
    *,
    policy=None,
    evaluation=None,
):
    save(
        context.output / "batch-policy-results.json",
        dict(
            completed=complete,
            **getattr(context, "public_provenance", {}),
            diagnostic_only=evaluation is None,
            formal_eligible=False,
            environment_count=env.n,
            simulated_seconds=tick / 400,
            checkpoint_package_digest=package["package_digest"],
            neural_batching=False,
            inference=policy.inference
            if policy
            else "scripted reference replay; no neural inference",
            camera_assignment=assignment,
            source_camera_parity=source_parity,
            cleanup_errors=cleanup_errors,
            committed_formal_trials=evaluation.completed if evaluation else {},
            source_camera_settings="AA4/global_secondary/persistent/adaptive1/full6/320x240",
            timing_seconds=timings,
        ),
    )


def qualify_capture(count, report, assignment, serial, sources, output):
    """Serial proof is mandatory; the unchanged batch range is explicit."""
    if not 1 <= count <= 10:
        raise ValueError("Camera qualification supports 1 serial or 2..10 batch environments")
    for label in ("initial", "batch_end"):
        if (
            assignment[label]["comparison"]["passed"] is not True
            or serial[label]["comparison"]["passed"] is not True
        ):
            raise ValueError("Serial camera evidence failed")
    if count >= 2:
        probe.verify_mount_evidence(report, source_hashes=sources, batch_root=output)
    save(
        output / "camera-qualification.json",
        dict(
            serial_passed=True,
            environment_count=count,
            batch_qualified=count >= 2,
            c19_range=[2, 10],
            c19_applied=count >= 2,
            reason="N=1 serial replay; outside C19 batch range"
            if count == 1
            else "batch mount evidence verified",
        ),
    )
