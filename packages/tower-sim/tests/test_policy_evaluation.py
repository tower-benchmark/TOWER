"""Standalone authorization and immutable evidence retain the frozen checks."""

import json
from pathlib import Path

import numpy as np
import pytest
from tower_sim.bench.evaluation import EvaluationContext, immutable
from tower_sim.bench.ik_identity import shared_spec_identity
from tower_sim.bench.resources import Resources, sha
from tower_sim.bench.tasks import BenchTask

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def authorization(tmp_path):
    task = BenchTask.from_resources("task1-2", Resources.from_roots(checkout=ROOT, assets=ROOT))

    def put(name, value):
        p = tmp_path / name
        p.write_text(json.dumps(value))
        return str(p)

    review = put(
        "review.json",
        dict(
            passed=True,
            native_inference_qualified=True,
            package_digest="authorized",
            scene_reference_review_passed=True,
            shared_ik_parity_passed=True,
            shared_spec_sha256=shared_spec_identity(),
        ),
    )
    package = put("package.json", dict(package_digest="authorized", checkpoint_sha256="checkpoint"))
    manifest = put("manifest.json", {})
    protocol = put(
        "protocol.json",
        dict(
            seeds=[1],
            package_digest="authorized",
            scene_version="tower_v4",
            duration_s=task.deadline_s,
            success_rule=task.success_rule,
            authorization=dict(scene_selection="Current TOWER v4 scene"),
            pilot_review=dict(path=review, sha256=sha(review)),
            ik=dict(shared_spec_sha256=shared_spec_identity()),
        ),
    )
    runtime = put(
        "runtime.json", dict(files={p: sha(p) for p in (manifest, package, review, protocol)})
    )
    request = dict(
        seeds=[1],
        duration=task.deadline_s,
        success_rule=task.success_rule,
        package=package,
        protocol=protocol,
        protocol_sha256=sha(protocol),
        runtime_manifest=runtime,
        runtime_manifest_sha256=sha(runtime),
        manifest=manifest,
    )
    health = dict(identity=dict(package_digest="authorized", extra={"seed": 42}))
    rows = [dict(candidate=dict(seed=1))]
    return request, health, rows, task


def test_authorization_rechecks_sources_and_full_identity(authorization, tmp_path):
    request, health, rows, task = authorization
    context = EvaluationContext(request, health, rows, task)
    context.verify_prediction(health["identity"])
    with pytest.raises(ValueError, match="identity"):
        context.verify_prediction(dict(health["identity"], drift=True))
    context.closure({request["package"]: sha(request["package"])})
    unrelated = tmp_path / "unbound.py"
    unrelated.write_text("pass")
    with pytest.raises(ValueError, match="Unbound"):
        context.closure({str(unrelated): sha(unrelated)})
    Path(request["package"]).write_text("{}")
    with pytest.raises(ValueError, match="Unbound"):
        context.closure({request["package"]: sha(request["package"])})


@pytest.mark.parametrize("field", ["package", "protocol", "manifest", "runtime_manifest"])
def test_changed_authorization_fails(authorization, field):
    request, health, rows, task = authorization
    path = Path(request[field])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        EvaluationContext(request, health, rows, task)


def test_evidence_is_exclusive_and_supports_numpy(tmp_path):
    path = tmp_path / "commands.json"
    immutable(path, [dict(time=0.0, encoder=np.zeros(16))])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        immutable(path, {})
    assert path.read_bytes() == before


@pytest.fixture
def diagnostic_recording(tmp_path):
    import h5py

    from scripts.maintainer.golden_support import golden_archive, unpack

    output = tmp_path / "output"
    run = output / "simulation/seed-1"
    unpack(golden_archive("task6-2-openwam.tar.xz"), run)
    status_path = run / "status.json"
    status = json.loads(status_path.read_text())
    status.update(
        diagnostic_only=True,
        valid_trial=False,
        evidence_complete=True,
        status="diagnostic_prefix_complete",
    )
    status_path.write_text(json.dumps(status))
    with h5py.File(run / "episode/states.hdf5", "r") as states:
        times = states["time"][:]
    # The immutable golden contains physical records, not camera bytes. Supply
    # only the camera completion/timestamp metadata checked by both scorers.
    with h5py.File(run / "camera-frames.hdf5", "w") as camera:
        camera.create_dataset("time", data=times)
        camera.attrs.update(complete=True, streams_consistent=True)
    for path in (output / "worker-complete.json", output / "simulation/batch-policy-results.json"):
        path.write_text(json.dumps(dict(completed=True, phase="teardown")))
    task = BenchTask.from_resources("task6-2", Resources.from_roots(checkout=ROOT, assets=ROOT))
    return output, run, task


@pytest.mark.golden
@pytest.mark.archive
def test_diagnostic_score_matches_frozen_without_qualifying_or_mutating(diagnostic_recording):
    from tower_sim.bench.policy_entry import collect_diagnostic

    from scripts.maintainer.golden_support import json_value
    from tests.characterization.archived_diagnostic import score_physical_diagnostic
    from tests.characterization.golden_compare import assert_equivalent

    output, run, task = diagnostic_recording
    before = (run / "status.json").read_bytes()
    expected = score_physical_diagnostic(run, "task6-2")
    result = collect_diagnostic(output, "task6-2", 1, 0, bench_task=task)
    assert result["evidence_complete"]
    assert result["score"]["prefix_only"] is False
    assert_equivalent(
        json_value({k: v for k, v in result["score"].items() if k != "prefix_only"}),
        json_value(expected),
    )
    assert (run / "status.json").read_bytes() == before
    assert "valid" not in result["score"]
    assert result["score"]["physical_recording_valid"]
    assert not result["score"]["formal_eligible"] and not result["score"]["pilot_approved"]


@pytest.mark.golden
@pytest.mark.parametrize("invalid", ["incomplete", "timestamp"])
def test_diagnostic_camera_evidence_failure_is_incomplete(diagnostic_recording, invalid):
    import h5py
    from tower_sim.bench.policy_entry import collect_diagnostic

    output, run, task = diagnostic_recording
    with h5py.File(run / "camera-frames.hdf5", "r+") as camera:
        if invalid == "incomplete":
            camera.attrs["complete"] = False
        else:
            camera["time"][-1] += 0.1
    result = collect_diagnostic(output, "task6-2", 1, 0, bench_task=task)
    assert not result["evidence_complete"]
    assert "scoring_error" in result and "score" not in result
    assert not (run / "diagnostic-score.json").exists()


def test_review_and_protocol_cannot_agree_on_a_noncanonical_ik(authorization, monkeypatch):
    from tower_sim.bench import evaluation

    request, health, rows, task = authorization
    # The bound inputs agree; only a change to the canonical identity invalidates them.
    monkeypatch.setattr(evaluation, "shared_spec_identity", lambda: "different canonical identity")
    with pytest.raises(ValueError, match="canonical shared IK identity"):
        EvaluationContext(request, health, rows, task)


@pytest.mark.parametrize(
    "schema", ["tower-public-benchmark-protocol-unreviewed-v1", None, "unknown"]
)
def test_evaluation_schema_gate(authorization, schema):
    request, health, rows, task = authorization
    path = Path(request["protocol"])
    protocol = read_json_for_test = json.loads(path.read_text())
    protocol.pop("pilot_review")
    protocol["schema"] = schema
    path.write_text(json.dumps(read_json_for_test))
    request["protocol_sha256"] = sha(path)
    runtime = Path(request["runtime_manifest"])
    files = json.loads(runtime.read_text())
    files["files"][str(path)] = sha(path)
    runtime.write_text(json.dumps(files))
    request["runtime_manifest_sha256"] = sha(runtime)
    if schema == "tower-public-benchmark-protocol-unreviewed-v1":
        EvaluationContext(request, health, rows, task)
        protocol["ik"]["shared_spec_sha256"] = "wrong"
        path.write_text(json.dumps(protocol))
        request["protocol_sha256"] = sha(path)
        with pytest.raises(ValueError, match="IK"):
            EvaluationContext(request, health, rows, task)
    else:
        with pytest.raises((KeyError, ValueError)):
            EvaluationContext(request, health, rows, task)


@pytest.mark.archive
def test_golden_posture_evidence(diagnostic_recording):
    from tower_eval.posture_flags import classify

    _, run, _ = diagnostic_recording
    report = json.loads(
        (
            ROOT / "tests/golden/benchmark-2026.11/task6-2-openwam.benchmark-score-cli.json"
        ).read_text()
    )
    result = classify(run / "episode/states.hdf5", report)
    # The published physical golden omits TCP arrays. Never invent a pass.
    assert result["status"] == "not_evaluated"
    assert result["reason"] == "Missing TCP evidence: eef, eef_world"


@pytest.mark.archive
@pytest.mark.parametrize(
    "task", [f"task{n}-{layers}" for n in (1, 2, 4, 5, 6, 7) for layers in (2, 3)]
)
def test_expert_recorded_posture(task):
    import os

    import h5py
    from tower_eval.posture_flags import classify
    from tower_eval.scoring.api import score_recording
    from tower_eval.scoring.recording import from_campaign_rows

    from scripts.maintainer.archive_root import ArchiveUnavailable
    from scripts.maintainer.archive_scene import v1_scene

    configured = os.environ.get("TOWER_P706_RECORDED_ROOT")
    if not configured:
        raise ArchiveUnavailable(
            "SKIP: supply --archive-root and TOWER_P706_RECORDED_ROOT for recorded TCP evidence"
        )
    run = next((Path(configured) / task / "simulation").glob("seed-*"))
    seed = int(run.name.removeprefix("seed-"))
    spec = v1_scene(ROOT, task, seed)["candidate"]["episode_spec"]
    with h5py.File(run / "episode/states.hdf5") as stream:
        arrays = {key: np.asarray(stream[key]) for key in stream}
    for name, key, field in (
        ("finger-link-poses.npz", "finger_poses", "poses_m"),
        ("finger-feedback.npz", "finger_forces", "forces_n"),
    ):
        with np.load(run / name, allow_pickle=False) as stream:
            arrays[key] = stream[field]
    rows = [{key: value[i] for key, value in arrays.items()} for i in range(len(arrays["time"]))]
    geometry = json.loads((run / "finger_geometry.json").read_text())
    report = score_recording(
        from_campaign_rows(spec, rows, geometry), task, ruleset="bench-2026.11"
    )
    result = classify(run / "episode/states.hdf5", report)
    assert result["operations"], result
    assert all(row["status"] == "ordinary" for row in result["operations"]), result


@pytest.mark.parametrize(
    "field",
    [
        None,
        "ready",
        "checkpoint_sha256",
        "horizon",
        "execute_steps",
        "control_hz",
        "denoise_steps",
        "seed",
    ],
)
def test_unreviewed_server_contract(authorization, monkeypatch, field):
    from types import SimpleNamespace as NS

    from tower_sim.bench import policy_entry
    from tower_sim.bench.verification import ReleaseVerification

    request, health, rows, task = authorization
    path = Path(request["protocol"])
    protocol = json.loads(path.read_text())
    protocol["schema"] = "tower-public-benchmark-protocol-unreviewed-v1"
    path.write_text(json.dumps(protocol))
    request["protocol_sha256"] = sha(path)
    runtime = Path(request["runtime_manifest"])
    manifest = json.loads(runtime.read_text())
    manifest["files"][str(path)] = sha(path)
    runtime.write_text(json.dumps(manifest))
    request["runtime_manifest_sha256"] = sha(runtime)
    request.update(policy_url="ws://localhost:19000", prompt="test", task="task1-2")
    health.update(ready=True)
    health["identity"].update(
        checkpoint_sha256="checkpoint",
        horizon=32,
        execute_steps=16,
        control_hz=10,
        denoise_steps=10,
        seed=42,
    )
    if field == "ready":
        health["ready"] = False
    elif field:
        health["identity"].pop(field)
    closed = []
    worker = NS(authorize=lambda: health, close=lambda: closed.append(True))
    monkeypatch.setattr(policy_entry, "validate_entry", lambda *a: None)
    monkeypatch.setattr(policy_entry.policy, "from_package", lambda **kw: worker)
    monkeypatch.setattr(ReleaseVerification, "verify", lambda *a: None)
    verification = object.__new__(ReleaseVerification)

    def prepare():
        return policy_entry.prepare(
            rows,
            task,
            request,
            metrics=NS(call=lambda name, fn: fn()),
            solver=None,
            verification=verification,
        )

    if field is None:
        assert prepare()[0] is worker
    else:
        with pytest.raises(ValueError):
            prepare()
        assert closed == [True]


def test_optional_annotation_after_durable_commit(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    from tower_sim.bench import annotations

    context = object.__new__(EvaluationContext)
    context.rows = [dict(candidate=dict(seed=0, episode_spec={}), output=tmp_path)]
    context.completed = {}
    context.closed_archives = set()
    context.duration_ticks = 20
    context.task = "task6-2"
    context.verification = None
    context.package = dict(checkpoint_sha256="checkpoint", package_digest="package")
    context.request = dict(protocol_sha256="protocol")
    context.success_rule = "success"
    context.scorer = NS(
        recording=lambda *a: NS(status={}),
        score=lambda r: dict(valid=True, scores=dict(g=0, p=0, s=0, D=0), protocol="p"),
    )

    def evidence_write(*args):
        from tower_sim.bench.run_summary import REQUIRED

        for name in REQUIRED - {"score.json"}:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        return tmp_path

    context.write_evidence = evidence_write

    def fail(*args):
        assert context.completed[0]["sha256"] == sha(tmp_path / "status.json")
        raise PermissionError("Optional annotation unavailable")

    monkeypatch.setattr(annotations, "publish", fail)
    rows = [dict(time=0), dict(time=0.05)]
    context.finalize_trial(
        0,
        NS(records=[rows], env=NS(finger_geometry={})),
        rows,
        NS(times=[0, 0.05], close=lambda **k: None),
        {},
        {},
        tick=20,
    )
    assert context.completed[0]["sha256"] == sha(tmp_path / "status.json")
