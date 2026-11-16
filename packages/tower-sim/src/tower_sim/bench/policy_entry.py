"""Policy entry with explicit public release verification for formal runs."""

import json
from pathlib import Path
from typing import Any, cast

from . import policy, replay
from .resources import save_exclusive
from .tasks import validate_spec


def validate_entry(request, rows, task):
    if request.get("command_replay", False):
        raise ValueError("Policy entrypoint cannot execute reference-replay requests")
    from tower_eval.policies.runtime_ws import validate_url

    validate_url(request["policy_url"])
    if not 1 <= len(rows) <= 10:
        raise ValueError("Candidate camera qualification supports 1..10 environments")
    if [row["candidate"]["seed"] for row in rows] != request["seeds"]:
        raise ValueError("Manifest seeds differ from policy request")
    if request["task"] != task.identity.legacy_id:
        raise ValueError("Explicit task differs from policy request")
    if request["success_rule"] != task.success_rule:
        raise ValueError("Policy success rule differs from benchmark")
    duration = request["duration"]
    if duration != task.deadline_s:
        raise ValueError("Policy duration differs from resolved execution task")
    for row in rows:
        validate_spec(row["candidate"]["episode_spec"], request["task"])


def prepare(
    rows,
    task,
    request,
    *,
    metrics,
    solver,
    diagnostic=False,
    benchmark_protocol=None,
    verification=None,
    client_root=None,
):
    from .verification import ReleaseVerification

    if verification is not None and type(verification) is not ReleaseVerification:
        raise ValueError("Formal verification must be a public ReleaseVerification instance")
    if not diagnostic and type(verification) is not ReleaseVerification:
        raise ValueError(
            "Formal runs require a ReleaseVerification; use diagnostic=True otherwise."
        )
    if verification is not None:
        verification.verify()
    validate_entry(request, rows, task)
    package = json.loads(Path(request["package"]).read_text())
    if diagnostic:
        if benchmark_protocol is None:
            raise ValueError("Diagnostic requires the frozen benchmark protocol")
        if request["prompt"] != benchmark_protocol["tasks"][request["task"]]["prompt"]:
            raise ValueError("Diagnostic prompt differs from frozen protocol")
    worker = policy.from_package(
        package=package,
        url=request["policy_url"],
        prompt=request["prompt"],
        task=request["task"],
        metrics=metrics,
        solver=solver,
        client_root=client_root,
    )
    try:
        health = metrics.call("startup.context_validation", worker.authorize)
        if diagnostic:
            validate_diagnostic_health(health, package)
        evaluation = None
        if not diagnostic:
            from .evaluation import EvaluationContext

            evaluation = EvaluationContext(request, health, rows, task, verification=verification)
        return worker, evaluation
    except BaseException:
        worker.close()
        raise


def validate_diagnostic_health(health, package):
    """Frozen benchmark_diagnostic health contract, before simulator startup."""
    identity = health.get("identity", {})
    if not health.get("ready"):
        raise ValueError("Policy service is not ready")
    for field in ("package_digest", "checkpoint_sha256"):
        if identity.get(field) != package[field]:
            raise ValueError(f"Policy service has a different {field}")
    for field, expected in (
        ("horizon", 32),
        ("execute_steps", 16),
        ("control_hz", 10),
        ("denoise_steps", 10),
        ("seed", 42),
    ):
        if identity.get(field) != expected:
            raise ValueError(f"Policy service changes the frozen {field}")


def execute(
    context,
    rows,
    task,
    *,
    worker,
    evaluation,
    metrics,
    source_collector,
    observed_validator,
    factories=None,
):
    """Run a session authorized by prepare BEFORE simulator construction."""
    try:
        return replay.execute(
            context,
            rows,
            task,
            metrics=metrics,
            source_collector=source_collector,
            observed_validator=observed_validator,
            factories=factories,
            policy=worker,
            evaluation=evaluation,
        )
    finally:
        worker.close()


def worker_exit(returncode, *, boundary):
    """Preserve process failure separately from verified episode evidence."""
    return dict(
        returncode=returncode,
        signal=-returncode if returncode < 0 else None,
        phase=("teardown" if boundary else "execution") if returncode else None,
    )


def collect_diagnostic(output, task, seed, returncode, *, bench_task=None, duration=None):
    """Kit exit zero alone never proves completion (benchmark_diagnostic:174)."""
    output = Path(output)
    run = output / "simulation" / f"seed-{seed}"
    paths = [
        output / "worker-complete.json",
        run / "status.json",
        output / "simulation/batch-policy-results.json",
    ]
    complete, status, batch = [json.loads(p.read_text()) if p.is_file() else {} for p in paths]
    boundary = complete.get("completed") is True and complete.get("phase") == "teardown"
    result = dict(
        diagnostic_only=True,
        formal_eligible=False,
        pilot_approved=False,
        benchmark_deadline_s=bench_task.identity.deadline_s if bench_task else None,
        prefix_only=bool(
            bench_task and duration is not None and duration < bench_task.identity.deadline_s
        ),
        task=task,
        seed=seed,
        worker_returncode=returncode,
        status=status,
        evidence_complete=bool(
            boundary
            and status.get("evidence_complete")
            and status.get("diagnostic_only")
            and not status.get("valid_trial")
            and batch.get("completed")
        ),
        posture_review="pending; no compliance claim",
    )
    if result["evidence_complete"]:
        try:
            if bench_task is None:
                raise ValueError("Resolved benchmark task required for diagnostic scoring")
            report = score_diagnostic(run, bench_task, duration=duration)
            save_exclusive(run / "diagnostic-score.json", report)
            result["score"] = report
        except Exception:  # noqa: BLE001 -- preserve incomplete diagnostic evidence
            import traceback

            result["scoring_error"] = traceback.format_exc()
            result["evidence_complete"] = False
    result["worker_exit"] = worker_exit(returncode, boundary=boundary)
    save_exclusive(
        run / "diagnostic-result.json"
        if run.is_dir()
        else output / f"diagnostic-result-seed-{seed}.json",
        result,
    )
    return result


def score_diagnostic(run, task, *, duration=None):
    """Physical outcome only; preserve status and never qualify a trial or pilot."""
    import h5py
    import numpy as np
    from tower_eval.scoring.recording import from_files

    from .scoring import BenchScorer

    run = Path(run)
    status = json.loads((run / "status.json").read_text())
    if (
        status.get("diagnostic_only") is not True
        or status.get("valid_trial") is not False
        or status.get("evidence_complete") is not True
        or status.get("status") != "diagnostic_prefix_complete"
    ):
        raise ValueError("A complete, explicitly unqualified diagnostic is required")
    loaded = from_files(run)
    if loaded.times[-1] > task.identity.deadline_s + 1e-8:
        raise ValueError("Diagnostic exceeds the frozen task deadline")
    with h5py.File(run / "camera-frames.hdf5", "r") as cameras:
        if not cameras.attrs.get("complete") or not cameras.attrs.get("streams_consistent"):
            raise ValueError("Diagnostic camera archive is incomplete")
        if abs(float(cast(h5py.Dataset, cameras["time"])[-1]) - float(loaded.times[-1])) > 1e-8:
            raise ValueError("Diagnostic terminal camera/state timestamps differ")
    with h5py.File(run / "episode/states.hdf5", "r") as states:
        values: dict[str, Any] = {
            name: cast(h5py.Dataset, states[name])[:]
            for name in ("time", "blocks", "block_velocities", "qpos")
        }
    with np.load(run / "finger-link-poses.npz", allow_pickle=False) as fingers:
        values["finger_poses"] = fingers["poses_m"]
    with np.load(run / "finger-feedback.npz", allow_pickle=False) as feedback:
        values["finger_forces"] = feedback["forces_n"]
    rows = [{key: value[i] for key, value in values.items()} for i in range(len(values["time"]))]
    scorer = BenchScorer(task)
    recording = scorer.recording(
        loaded.spec,
        rows,
        json.loads((run / "finger_geometry.json").read_text()),
        failed=status.get("policy_action_failure", False),
    )
    report = scorer.score(recording)
    if report.get("valid") is not True:
        raise ValueError("Scorer rejected the diagnostic physical recording")
    if status.get("task_success") and report["scores"]["s"] != 1:
        raise ValueError("Offline physical score disagrees with online success")
    report["physical_recording_valid"] = report.pop("valid")
    report.update(
        diagnostic_only=True,
        formal_eligible=False,
        pilot_approved=False,
        benchmark_deadline_s=task.identity.deadline_s,
        prefix_only=bool(duration is not None and duration < task.identity.deadline_s),
        recorded_trial_status=status,
        source_hashes=loaded.hashes,
        scoring_scope="Physical outcome only, using campaign detectors; no trial or posture qualification",
    )
    return report
