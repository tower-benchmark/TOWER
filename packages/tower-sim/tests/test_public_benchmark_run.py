"""Fault probes for immutable, relocatable and explicitly scoped public results."""

import argparse
import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace as NS

import h5py
import pytest
from tower_sim.bench import annotations, run_summary
from tower_sim.bench.resources import sha
from tower_sim.bench.run_bundle import retain

ROOT = Path(__file__).resolve().parents[3]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools/simulator"))
    return importlib.import_module("benchmark_run")


def test_scene_selection(entry):
    assert entry.parse_scenes("3,7,10-12") == [3, 7, 10, 11, 12]
    assert entry.parse_scenes("0-49") == list(range(50))
    for bad in ("", "50", "-1", "4-2", "3,3", "1-3,2", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            entry.parse_scenes(bad)


def test_out_of_bounds_scene_range_is_rejected_before_expansion(entry, monkeypatch):
    def unexpected_range(*args):
        pytest.fail("Scene range bounds must be checked before expansion")

    monkeypatch.setattr(entry, "range", unexpected_range, raising=False)
    with pytest.raises(argparse.ArgumentTypeError, match="Expected unique scenes"):
        entry.parse_scenes("0-1000000000")


def fixture(root, seed=0, *, extra=(), complete=True, checkpoint="c" * 64, commit="a" * 40):
    root.mkdir(parents=True)
    work = root / "requests/batch-000"
    work.mkdir(parents=True)
    attempt = root / "attempts/batch-000"
    attempt.mkdir(parents=True)
    batch = root / "runs/batch-000"
    episode = batch / f"simulation/seed-{seed}"
    from tower_sim.bench.verification import envelope

    public = envelope()
    public["commit"] = commit
    release = work / "release.json"
    write(release, dict(schema="tower-release-v2", commit=public["commit"]))
    public["release_manifest_sha256"] = sha(release)
    protocol = dict(
        schema=run_summary.UNREVIEWED,
        qualification="unreviewed",
        task="task6-2",
        seeds=[seed],
        package_digest="package",
        public=public,
    )
    from tower_sim.bench.ik_identity import shared_spec_identity

    protocol.update(
        scene_version="tower_v4",
        ik=dict(shared_spec_sha256=shared_spec_identity()),
        duration_s=40,
        success_rule="success",
        policy={},
    )
    package = work / "package.json"
    write(package, dict(package_digest="package", checkpoint_sha256=checkpoint))
    write(work / "protocol.json", protocol)
    request = dict(
        task="task6-2", seeds=[seed], public=public, package=str(package), output=str(batch)
    )
    request.update(duration=40, success_rule="success")
    for key, filename in (
        ("protocol", "protocol.json"),
        ("release_manifest", "release.json"),
        ("dependency_manifest", "deps.json"),
        ("ik_dependency_manifest", "ikdeps.json"),
        ("ik_bounds", "ik.json"),
    ):
        path = work / filename
        if not path.exists():
            write(path, {})
        request[key] = str(path)
        request[key + "_sha256"] = sha(path)
    runtime = work / "inputs.json"
    write(
        runtime,
        dict(
            schema="tower-evaluation-inputs-v2",
            files={
                str(package): sha(package),
                str(work / "protocol.json"): sha(work / "protocol.json"),
            },
        ),
    )
    request.update(runtime_manifest=str(runtime), runtime_manifest_sha256=sha(runtime))
    write(work / "request.json", request)
    write(attempt / "started.json", dict(seeds=[seed]))
    retain(work / "request.json", attempt)
    batches = [dict(id="batch-000", seeds=[seed])]
    batches += [dict(id=f"batch-{i + 1:03d}", seeds=[s]) for i, s in enumerate(extra)]
    write(
        root / "run.json",
        dict(
            schema="tower-benchmark-run-v1",
            run_id=root.name,
            task="task6-2",
            requested=[seed, *extra],
            batches=batches,
        ),
    )
    for name in run_summary.REQUIRED:
        write(episode / name, {})
    with h5py.File(episode / "episode/states.hdf5", "w"):
        pass
    write(
        episode / "score.json",
        dict(task="task6-2", valid=True, scores=dict(g=1, p=0, s=1, D=2), planned_g=1, planned_p=1),
    )
    status = dict(
        schema="tower-v4-standalone-policy-trial-v1",
        seed=seed,
        valid_trial=True,
        evidence_complete=True,
        status="duration_reached",
        package_digest="package",
        checkpoint_sha256=checkpoint,
        protocol_sha256=request["protocol_sha256"],
        public=public,
        **{k: public[k] for k in ("commit", "release_manifest_sha256", "data_manifest_sha256")},
        source_hashes={name: sha(episode / name) for name in run_summary.REQUIRED},
    )
    write(episode / "status.json", status)
    write(batch / "worker-complete.json", dict(completed=True, phase="teardown"))
    write(batch / "result.json", dict(evidence_complete=complete, worker_returncode=-11))
    write(
        batch / "simulation/batch-policy-results.json",
        dict(
            completed=complete,
            diagnostic_only=False,
            checkpoint_package_digest="package",
            public=public,
            committed_formal_trials={
                str(seed): dict(
                    path=str(episode / "status.json"), sha256=sha(episode / "status.json")
                )
            },
        ),
    )
    return root, episode, batch


def commit_status(episode, batch):
    p = batch / "simulation/batch-policy-results.json"
    body = json.loads(p.read_text())
    seed = episode.name.split("-")[-1]
    body["committed_formal_trials"][seed]["sha256"] = sha(episode / "status.json")
    write(p, body)


def test_subset_partial_exclusions_and_relocation(tmp_path):
    root, episode, _ = fixture(tmp_path / "run", extra=(1, 2))
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    value = run_summary.aggregate([root], tmp_path / "out")
    assert value["metrics"] == dict(GSR=1, PSR=0, LHPR=0.5, TSR=1, ODR=2, C=1, E=1, trials=1)
    assert value["coverage"]["not_run"]["count"] == 2
    assert (
        not value["coverage"]["requested_complete"] and not value["coverage"]["full_task_coverage"]
    )
    assert not (episode / "posture-flags.json").exists()
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    moved = tmp_path / "moved"
    shutil.move(root, moved)
    for p in moved.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    assert run_summary.aggregate([moved], tmp_path / "out2") == value


def test_complete_subset_and_nonzero_teardown(tmp_path):
    root, _, _ = fixture(tmp_path / "run")
    value = run_summary.aggregate([root], tmp_path / "out")
    assert value["coverage"]["requested_complete"] is True
    assert value["coverage"]["full_task_coverage"] is False
    assert value["qualification"] == "unreviewed"


@pytest.mark.parametrize("marker", ["batch", "parent", "worker", "status"])
def test_failed_batch_not_salvaged(tmp_path, marker):
    root, episode, batch = fixture(tmp_path / "run")
    path, field = {
        "batch": (batch / "simulation/batch-policy-results.json", "completed"),
        "parent": (batch / "result.json", "evidence_complete"),
        "worker": (batch / "worker-complete.json", "completed"),
        "status": (episode / "status.json", "valid_trial"),
    }[marker]
    body = json.loads(path.read_text())
    body[field] = False
    write(path, body)
    if marker == "status":
        commit_status(episode, batch)
    value = run_summary.aggregate([root], tmp_path / "out")
    assert value["metrics"]["trials"] == 0 and value["metrics"]["TSR"] is None
    assert value["coverage"]["excluded"]["count"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_sha256", "bad"),
        ("package_digest", "bad"),
        ("seed", 3),
        ("seed", True),
        ("checkpoint_sha256", "bad"),
        ("schema", "bad"),
        ("public", {}),
    ],
)
def test_status_bindings(tmp_path, field, value):
    root, episode, batch = fixture(tmp_path / "run")
    path = episode / "status.json"
    body = json.loads(path.read_text())
    body[field] = value
    write(path, body)
    commit_status(episode, batch)
    with pytest.raises(ValueError):
        run_summary.aggregate([root], tmp_path / "out")


@pytest.mark.parametrize("name", sorted(run_summary.REQUIRED) + ["status.json"])
def test_all_retained_hashes(tmp_path, name):
    root, episode, _ = fixture(tmp_path / "run")
    path = episode / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        run_summary.aggregate([root], tmp_path / "out")


@pytest.mark.parametrize(
    "field,value",
    [("valid", False), ("valid", "true"), ("task", "task1-2"), ("ruleset", "paper-v1")],
)
def test_report_validation(tmp_path, field, value):
    root, episode, batch = fixture(tmp_path / "run")
    path = episode / "score.json"
    body = json.loads(path.read_text())
    body[field] = value
    write(path, body)
    status = episode / "status.json"
    body = json.loads(status.read_text())
    body["source_hashes"]["score.json"] = sha(path)
    write(status, body)
    commit_status(episode, batch)
    with pytest.raises(ValueError):
        run_summary.aggregate([root], tmp_path / "out")


def test_strict_duplicates_and_identity(tmp_path):
    one, _, _ = fixture(tmp_path / "one", complete=False)
    two, _, _ = fixture(tmp_path / "two")
    with pytest.raises(ValueError, match="Duplicate"):
        run_summary.aggregate([one, two], tmp_path / "duplicate")
    three, _, _ = fixture(tmp_path / "three", seed=1, checkpoint="e" * 64)
    with pytest.raises(ValueError, match="Mixed"):
        run_summary.aggregate([one, three], tmp_path / "mixed")


@pytest.mark.parametrize("schema", [None, "unknown", run_summary.REVIEWED])
def test_review_fail_closed(schema):
    with pytest.raises((ValueError, KeyError, AttributeError)):
        run_summary.qualification(dict(schema=schema), None)


@pytest.mark.parametrize("failure", ["import", "classify", "serialize", "write"])
def test_annotation_failure_nonfatal(tmp_path, monkeypatch, failure):
    import builtins

    from tower_eval import posture_flags

    root, episode, _ = fixture(tmp_path / "run")
    write(episode / "posture-flags.json", {"malformed_or_different": True})

    def fail(*args, **kwargs):
        raise PermissionError("optional annotation")

    if failure == "import":
        original = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "tower_eval.posture_flags":
                raise ImportError("blocked")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
    elif failure == "classify":
        monkeypatch.setattr(posture_flags, "flags", fail)
    elif failure == "serialize":
        monkeypatch.setattr(posture_flags, "flags", lambda *a: {"bad": object()})
    else:
        monkeypatch.setattr(annotations, "save_exclusive", fail)
    value = run_summary.aggregate([root], tmp_path / "out")
    assert value["metrics"]["trials"] == 1
    assert value["scenes"][0]["posture"]["status"] == "not_evaluated"


@pytest.mark.parametrize("keep", [False, True])
def test_prepare_failure_recovery(entry, monkeypatch, tmp_path, keep, capsys):
    public = importlib.import_module("benchmark_candidate")
    calls = []

    def prepare(args, **kwargs):
        assert (args.output.parents[1] / "run.json").is_file()
        calls.append(args.seeds)
        raise RuntimeError("preparation unavailable")

    monkeypatch.setattr(public, "prepare", prepare)
    monkeypatch.setattr(entry.benchmark, "require_eula", lambda *a: None)
    args = NS(
        task="task6-2",
        scenes=[0, 1],
        batch_size=1,
        output=tmp_path / "out",
        keep_going=keep,
        accept_eula=True,
    )
    assert entry.run(args) == 1
    value = json.loads((args.output / "score.json").read_text())
    assert value["coverage"]["attempted"]["count"] == (2 if keep else 1)
    assert value["coverage"]["not_run"]["count"] == (0 if keep else 1)
    assert len(value["sources"]) == 2


def test_interrupted_prepared_attempt(tmp_path):
    root, _, batch = fixture(tmp_path / "run", extra=(1,))
    shutil.rmtree(batch)
    result = run_summary.aggregate([root], tmp_path / "out")
    assert result["coverage"]["attempted"]["count"] == 1
    assert result["coverage"]["excluded"]["count"] == 1
    assert result["coverage"]["not_run"]["count"] == 1
    assert result["sources"][0]["request_sha256"]


@pytest.mark.parametrize(
    "field",
    [
        "release_manifest",
        "runtime_manifest",
        "dependency_manifest",
        "ik_dependency_manifest",
        "ik_bounds",
        "protocol",
    ],
)
def test_retained_input_hashes(tmp_path, field):
    root, _, _ = fixture(tmp_path / "run")
    attempt = root / "attempts/batch-000"
    request = json.loads((attempt / "request.json").read_text())
    path = attempt / "inputs" / request[field + "_sha256"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Changed"):
        run_summary.aggregate([root], tmp_path / "out")


@pytest.mark.parametrize("field,value", [("valid_trial", "true"), ("evidence_complete", 1)])
def test_trial_strict_booleans(tmp_path, field, value):
    root, episode, batch = fixture(tmp_path / "run")
    path = episode / "status.json"
    body = json.loads(path.read_text())
    body[field] = value
    write(path, body)
    commit_status(episode, batch)
    result = run_summary.aggregate([root], tmp_path / "out")
    assert result["metrics"]["trials"] == 0


def test_forged_qualification_ignored(tmp_path):
    root, _, _ = fixture(tmp_path / "run")
    plan = root / "run.json"
    body = json.loads(plan.read_text())
    body["qualification"] = "posture-reviewed"
    write(plan, body)
    assert run_summary.aggregate([root], tmp_path / "out")["qualification"] == "unreviewed"


def test_missing_required_hash(tmp_path):
    root, episode, batch = fixture(tmp_path / "run")
    path = episode / "status.json"
    body = json.loads(path.read_text())
    del body["source_hashes"]["camera-frames.hdf5"]
    write(path, body)
    commit_status(episode, batch)
    with pytest.raises(ValueError, match="required"):
        run_summary.aggregate([root], tmp_path / "out")


def rebind(root, change):
    work = root / "requests/batch-000"
    protocol = json.loads((work / "protocol.json").read_text())
    change(protocol)
    write(work / "protocol.json", protocol)
    request = json.loads((work / "request.json").read_text())
    request["protocol_sha256"] = sha(work / "protocol.json")
    runtime = json.loads((work / "inputs.json").read_text())
    runtime["files"][str(work / "protocol.json")] = request["protocol_sha256"]
    write(work / "inputs.json", runtime)
    request["runtime_manifest_sha256"] = sha(work / "inputs.json")
    write(work / "request.json", request)
    attempt = root / "attempts/batch-000"
    shutil.rmtree(attempt / "inputs")
    (attempt / "prepared.json").unlink()
    retain(work / "request.json", attempt)
    return attempt


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(ruleset="other"),
        lambda p: p.update(qualification="posture-reviewed"),
        lambda p: p.update(package_digest="other"),
        lambda p: p.update(seeds=[1]),
        lambda p: p.update(task="task1-2"),
        lambda p: p.update(public=dict(p["public"], commit="b" * 40)),
    ],
)
def test_protocol_identity_gates(tmp_path, change):
    from tower_sim.bench.run_bundle import Bundle

    root, _, _ = fixture(tmp_path / "run")
    attempt = rebind(root, change)
    with pytest.raises(ValueError):
        run_summary.identity(Bundle(attempt))


def test_reviewed_qualification_and_strict_flags(tmp_path):
    from tower_sim.bench.ik_identity import shared_spec_identity
    from tower_sim.bench.run_bundle import Bundle

    root, _, _ = fixture(tmp_path / "run")
    work = root / "requests/batch-000"
    proof = work / "proof.json"
    write(proof, {})
    posture = dict(
        schema="tower-no-reversed-grasp-review-v1",
        task="task6-2",
        package_digest="package",
        planned_approach_carry_passed=True,
        recorded_approach_carry_passed=True,
        reverse_180_grasp_absent=True,
        task_contact_geometry_preserved=True,
        planned_trajectory=dict(path=str(proof), sha256=sha(proof)),
        recorded_approach_carry=dict(path=str(proof), sha256=sha(proof)),
    )
    review = work / "review.json"
    body = dict(
        passed=True,
        native_inference_qualified=True,
        scene_reference_review_passed=True,
        shared_ik_parity_passed=True,
        package_digest="package",
        shared_spec_sha256=shared_spec_identity(),
        posture_review=posture,
    )
    for flag in (True, "true"):
        body["passed"] = flag
        write(review, body)
        runtime = json.loads((work / "inputs.json").read_text())
        runtime["files"].update({str(review): sha(review), str(proof): sha(proof)})
        write(work / "inputs.json", runtime)
        attempt = rebind(
            root,
            lambda p: p.update(
                schema=run_summary.REVIEWED,
                qualification="posture-reviewed",
                ik=dict(shared_spec_sha256=shared_spec_identity()),
                pilot_review=dict(path=str(review), sha256=sha(review)),
            ),
        )
        if flag is True:
            assert run_summary.identity(Bundle(attempt))[1]["qualification"] == "posture-reviewed"
        else:
            with pytest.raises(ValueError, match="review"):
                run_summary.identity(Bundle(attempt))


def test_retained_nonjson_evidence_hash(tmp_path):
    root, _, _ = fixture(tmp_path / "run")
    work = root / "requests/batch-000"
    proof = work / "geometry.bin"
    proof.write_bytes(b"physical geometry")
    runtime = json.loads((work / "inputs.json").read_text())
    runtime["files"][str(proof)] = sha(proof)
    write(work / "inputs.json", runtime)
    attempt = rebind(root, lambda p: None)
    (attempt / "inputs" / sha(proof)).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed"):
        run_summary.aggregate([root], tmp_path / "out")


def test_all_fifty_coverage(tmp_path):
    roots = [fixture(tmp_path / f"run-{seed}", seed=seed)[0] for seed in range(50)]
    result = run_summary.aggregate(roots, tmp_path / "out")
    assert result["coverage"]["requested_complete"] is True
    assert result["coverage"]["full_task_coverage"] is True
    assert result["coverage"]["scored"]["count"] == 50


@pytest.mark.parametrize("keep", [False, True])
def test_launch_failure_retains_prelaunch_request(entry, monkeypatch, tmp_path, keep):
    public = importlib.import_module("benchmark_candidate")
    fixtures = [fixture(tmp_path / f"input-{seed}", seed=seed)[0] for seed in (0, 1)]
    output = tmp_path / "run"
    calls = []
    monkeypatch.setattr(entry.benchmark, "require_eula", lambda *a: None)
    monkeypatch.setattr(
        public,
        "prepare",
        lambda args, **kw: fixtures[args.seeds[0]] / "requests/batch-000/request.json",
    )

    def launch(path, *a, **kw):
        index = len(calls)
        assert (output / f"attempts/batch-{index:03d}/prepared.json").is_file()
        calls.append(path)
        raise RuntimeError("execution interrupted")

    monkeypatch.setattr(public, "launch", launch)
    assert (
        entry.run(
            NS(
                task="task6-2",
                scenes=[0, 1],
                batch_size=1,
                output=output,
                keep_going=keep,
                accept_eula=True,
            )
        )
        == 1
    )
    assert len(calls) == (2 if keep else 1)


@pytest.mark.parametrize("checkpoint", [None, "", "malformed"])
def test_missing_checkpoint_identity(tmp_path, checkpoint):
    root, _, _ = fixture(tmp_path / "run", checkpoint=checkpoint)
    with pytest.raises(ValueError, match="checkpoint"):
        run_summary.aggregate([root], tmp_path / "out")


def test_batch_package_identity(tmp_path):
    root, _, batch = fixture(tmp_path / "run")
    path = batch / "simulation/batch-policy-results.json"
    body = json.loads(path.read_text())
    body["checkpoint_package_digest"] = "other"
    write(path, body)
    with pytest.raises(ValueError, match="identity"):
        run_summary.aggregate([root], tmp_path / "out")


def test_malformed_release_commit(tmp_path):
    root, _, _ = fixture(tmp_path / "run", commit="malformed")
    with pytest.raises(ValueError, match="commit"):
        run_summary.aggregate([root], tmp_path / "out")


def test_annotation_does_not_swallow_interrupts(tmp_path, monkeypatch):
    from tower_eval import posture_flags

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(posture_flags, "flags", interrupt)
    with pytest.raises(KeyboardInterrupt):
        annotations.publish(tmp_path / "states", tmp_path / "score", tmp_path / "flags")


@pytest.mark.parametrize(
    "relative",
    [
        "run.json",
        "attempts/batch-000",
        "attempts/batch-000/started.json",
        "attempts/batch-000/prepared.json",
        "attempts/batch-000/request.json",
        "runs/batch-000",
        "runs/batch-000/result.json",
        "runs/batch-000/worker-complete.json",
        "runs/batch-000/simulation/batch-policy-results.json",
        "runs/batch-000/simulation/seed-0",
        "runs/batch-000/simulation/seed-0/status.json",
        "runs/batch-000/simulation/seed-0/score.json",
        "runs/batch-000/simulation/seed-0/camera-frames.hdf5",
    ],
)
def test_external_evidence_links_rejected(tmp_path, relative):
    root, _, _ = fixture(tmp_path / "run")
    path = root / relative
    external = tmp_path / "external"
    path.rename(external)
    path.symlink_to(external, target_is_directory=external.is_dir())
    with pytest.raises(ValueError, match="escapes bundle"):
        run_summary.aggregate([root], tmp_path / "out")
    assert not (tmp_path / "out/score.json").exists()


@pytest.mark.parametrize("relative", ["annotations", "annotations/scene-000.json", "score.json"])
@pytest.mark.parametrize("initial", [False, True])
def test_output_links_rejected_without_source_writes(tmp_path, relative, initial):
    root, _, _ = fixture(tmp_path / "run")
    output = root if initial else tmp_path / "out"
    link = output / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    destination = root / "injected.json" if relative.endswith(".json") else root / "attempts"
    link.symlink_to(destination, target_is_directory=not relative.endswith(".json"))
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="output contains a symlink"):
        run_summary.aggregate([root], output, initial=initial)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert not (root / "injected.json").exists()


def test_internal_evidence_link_survives_relocation(tmp_path):
    root, episode, _ = fixture(tmp_path / "run")
    score = episode / "score.json"
    score.rename(episode / "retained-score.json")
    score.symlink_to("retained-score.json")
    value = run_summary.aggregate([root], tmp_path / "out")
    moved = tmp_path / "moved"
    root.rename(moved)
    assert run_summary.aggregate([moved], tmp_path / "out2") == value


def test_cli_output_redirect_rejected(tmp_path):
    import os
    import subprocess
    import sys

    root, _, _ = fixture(tmp_path / "run")
    output = tmp_path / "out"
    output.mkdir()
    (output / "annotations").symlink_to(root, target_is_directory=True)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/simulator/benchmark_run.py"),
            "aggregate",
            "--runs",
            str(root),
            "--output",
            str(output),
        ],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Aggregate output contains a symlink" in result.stderr
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
