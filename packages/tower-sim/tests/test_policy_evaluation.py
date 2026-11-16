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
