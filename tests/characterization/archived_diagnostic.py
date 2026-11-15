"""Historical diagnostic scorer oracle; no public entry point."""

import importlib
from pathlib import Path
from typing import Any, cast

from scripts.maintainer import archived_benchmark as benchmark


def score_physical_diagnostic(run, task):
    import h5py
    import numpy as np

    run = Path(run)
    status = benchmark.read(run / "status.json")
    if (
        status.get("diagnostic_only") is not True
        or status.get("valid_trial") is not False
        or status.get("evidence_complete") is not True
        or status.get("status") != "diagnostic_prefix_complete"
    ):
        raise ValueError("A complete, explicitly unqualified diagnostic is required")
    settings = benchmark.read(benchmark.BUNDLE / "protocol.json")["tasks"][task]
    scorer = benchmark.load_scorer()
    trajectory = importlib.import_module(scorer.__package__ + ".offline.trajectory")
    # Validate the original artifact set and retain its original status hash.
    loaded = trajectory.load_recording(run, trajectory.DEFAULT_RULES)
    if loaded.times[-1] > settings["duration_s"] + 1e-8:
        raise ValueError("Diagnostic exceeds the frozen task deadline")
    with h5py.File(run / "camera-frames.hdf5") as camera_file:
        cameras = cast(Any, camera_file)
        if not cameras.attrs.get("complete") or not cameras.attrs.get("streams_consistent"):
            raise ValueError("Diagnostic camera archive is incomplete")
        camera_times = cameras["time"][:]
        if abs(float(camera_times[-1]) - float(loaded.times[-1])) > 1e-8:
            raise ValueError("Diagnostic terminal camera/state timestamps differ")
    with h5py.File(run / "episode/states.hdf5") as state_file:
        states = cast(Any, state_file)
        values = {name: states[name][:] for name in ("time", "blocks", "block_velocities", "qpos")}
    with np.load(run / "finger-link-poses.npz", allow_pickle=False) as fingers:
        values["finger_poses"] = fingers["poses_m"]
    with np.load(run / "finger-feedback.npz", allow_pickle=False) as feedback:
        values["finger_forces"] = feedback["forces_n"]
    rows = [{key: value[i] for key, value in values.items()} for i in range(len(values["time"]))]
    # Use the same physical-row conversion as EvaluationContext.finalize_trial
    # and the online detector. Its temporary status is an algorithm input;
    # no original status file or qualification receipt is changed or created.
    recording = scorer.from_physical_rows(
        loaded.spec, rows, benchmark.read(run / "finger_geometry.json")
    )
    if status.get("policy_action_failure") and task != "task1-2":
        recording.status["status"] = "task_failure"
    report = scorer.score_recording(recording, task, success_rule=settings["success_rule"])
    if report.get("valid") is not True:
        raise ValueError("Scorer rejected the diagnostic physical recording")
    if status.get("task_success") and report["scores"]["s"] != 1:
        raise ValueError("Offline physical score disagrees with online success")
    report["physical_recording_valid"] = report.pop("valid")
    report.update(
        diagnostic_only=True,
        formal_eligible=False,
        pilot_approved=False,
        benchmark_deadline_s=settings["duration_s"],
        recorded_trial_status=status,
        source_hashes=loaded.hashes,
        scoring_scope="Physical outcome only, using campaign detectors; no trial or posture qualification",
    )
    return report
