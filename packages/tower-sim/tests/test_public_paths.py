"""Exercise the default preparation/run/diagnostic paths and posture authorization."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_policy_evaluation import diagnostic_recording as diagnostic_recording
from test_public_default import benchmark, public
from tower_sim.bench import policy_entry
from tower_sim.bench import verification as v
from tower_sim.bench.resources import sha


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    def put(name, body):
        p = tmp_path / name
        p.write_text(json.dumps(body))
        return p

    package = put("package.json", dict(package_digest="policy"))
    planned, recorded = put("planned.json", {}), put("recorded.json", {})
    from nero_kinematics.profiles import load_profile

    review = put(
        "review.json",
        dict(
            passed=True,
            native_inference_qualified=True,
            scene_reference_review_passed=True,
            shared_ik_parity_passed=True,
            package_digest="policy",
            shared_spec_sha256=load_profile("command")["source_sha256"],
            posture_review=dict(
                schema="tower-no-reversed-grasp-review-v1",
                task="task1-2",
                package_digest="policy",
                planned_approach_carry_passed=True,
                recorded_approach_carry_passed=True,
                reverse_180_grasp_absent=True,
                task_contact_geometry_preserved=True,
                planned_trajectory=dict(path=str(planned), sha256=sha(planned)),
                recorded_approach_carry=dict(path=str(recorded), sha256=sha(recorded)),
            ),
        ),
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    isaac, ik = tmp_path / "isaac", tmp_path / "ik"
    isaac.touch()
    ik.touch()
    release = put("release.json", {"build": True})
    release_identity = dict(commit="a" * 40, manifest_sha256=sha(release))
    monkeypatch.setattr(public, "verify", lambda **kw: {})
    monkeypatch.setattr(public, "verify_startup", lambda *a: None)
    monkeypatch.setattr(public, "verify_release", lambda *args, **kwargs: release_identity)

    def scenes(
        task,
        seeds,
        assets,
        output,
        *,
        reference=True,
        benchmark_version=None,
        data_root=None,
        v1_root=None,
    ):
        assert reference is False
        output.mkdir()
        path = output / "manifest.json"
        path.write_text(
            json.dumps(
                dict(
                    assets=str(assets),
                    benchmark=benchmark_version,
                    data_root=data_root,
                    v1_root=v1_root,
                )
            )
        )
        return path

    monkeypatch.setattr(public.benchmark_assets, "prepare_scenes", scenes)
    monkeypatch.setattr(
        public.benchmark_assets,
        "load_rows",
        lambda path: [dict(manifest_entry=dict(task="task1-2"), candidate=dict(seed=1))],
    )
    monkeypatch.setattr(public.subprocess, "run", lambda *a, **kw: NS(stdout="{}"))
    # Only the release location is redirected; data and IK bounds still use actual ROOT.
    original_read_bytes = Path.read_bytes
    original_sha = public.sha
    expected_release = public.ROOT / "dist/release-manifest.json"
    monkeypatch.setattr(
        public, "sha", lambda p: sha(release) if Path(p) == expected_release else original_sha(p)
    )
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: original_read_bytes(release if p == expected_release else p)
    )
    args = NS(
        output=tmp_path / "inputs",
        run_output=tmp_path / "run",
        task="task1-2",
        seeds=[1],
        assets=assets,
        package=package,
        pilot_review=review,
        isaac_python=isaac,
        ik_python=ik,
        gpu=0,
        policy_url="ws://localhost:19000",
    )
    path = public.prepare(args)
    return args, path, planned, recorded, release_identity


def save_request(path, request):
    path.write_text(json.dumps(request))


def update_protocol(request, update, *, bind_review=False):
    path = Path(request["protocol"])
    settings = json.loads(path.read_text())
    update(settings)
    path.write_text(json.dumps(settings))
    request["protocol_sha256"] = sha(path)
    runtime = Path(request["runtime_manifest"])
    files = json.loads(runtime.read_text())
    files["files"][str(path)] = sha(path)
    if bind_review:
        review = settings["pilot_review"]
        files["files"][review["path"]] = sha(review["path"])
    runtime.write_text(json.dumps(files))
    request["runtime_manifest_sha256"] = sha(runtime)


def test_default_preparation_binds_envelope_and_all_posture_files(inputs):
    args, path, planned, recorded, identity = inputs
    request = public.preflight(path)
    assert request["public"] == v.envelope(identity)
    files = json.loads(Path(request["runtime_manifest"]).read_text())["files"]
    for p in (planned, recorded, args.pilot_review, args.package):
        assert files[str(p)] == sha(p)


def test_prepare_request_only_spawns_cpu_dependency_probes(inputs, monkeypatch):
    args, _, _, _, _ = inputs
    args.output = args.output.with_name("second-inputs")
    calls = []

    def probe(argv, **kwargs):
        calls.append(argv)
        return NS(stdout="{}")

    monkeypatch.setattr(public.subprocess, "run", probe)
    monkeypatch.setattr(public, "create_runner", lambda **kw: pytest.fail("Preparation starts Kit"))
    assert public.prepare(args).is_file()
    assert len(calls) == 2
    assert all(argv[1:3] == ["-m", "tower_sim.bench.verification"] for argv in calls)


@pytest.mark.parametrize(
    "change",
    ["envelope", "review-hash", "planned-binding", "recorded-binding", "bounds", "bounds-binding"],
)
def test_default_preflight_rejects_unbound_authorization(inputs, change, tmp_path):
    _, path, planned, recorded, _ = inputs
    request = json.loads(path.read_text())
    if change == "envelope":
        request["public"] = {}
    elif change == "review-hash":
        update_protocol(request, lambda settings: settings["pilot_review"].update(sha256="0" * 64))
    elif change.endswith("binding") and change != "bounds-binding":
        runtime = Path(request["runtime_manifest"])
        body = json.loads(runtime.read_text())
        body["files"].pop(str(planned if change == "planned-binding" else recorded))
        runtime.write_text(json.dumps(body))
        request["runtime_manifest_sha256"] = sha(runtime)
    elif change == "bounds-binding":
        request["ik_bounds_sha256"] = "0" * 64
    else:
        bounds = tmp_path / "other-bounds.json"
        bounds.write_text("{}")
        request.update(ik_bounds=str(bounds), ik_bounds_sha256=sha(bounds))
    save_request(path, request)
    with pytest.raises(ValueError):
        public.preflight(path)


@pytest.mark.parametrize(
    "field",
    [
        "planned_approach_carry_passed",
        "recorded_approach_carry_passed",
        "reverse_180_grasp_absent",
        "task_contact_geometry_preserved",
        "task",
        "package_digest",
        "schema",
    ],
)
def test_default_path_requires_current_task_specific_recorded_posture(inputs, field):
    args, path, _, _, _ = inputs
    request = json.loads(path.read_text())
    review = json.loads(args.pilot_review.read_text())
    review["posture_review"][field] = (
        False if field.endswith(("passed", "absent", "preserved")) else "wrong"
    )
    args.pilot_review.write_text(json.dumps(review))
    update_protocol(
        request,
        lambda settings: settings["pilot_review"].update(sha256=sha(args.pilot_review)),
        bind_review=True,
    )
    save_request(path, request)
    with pytest.raises(ValueError, match="posture review|Posture review"):
        public.preflight(path)


def test_default_preflight_calls_release_verifier(inputs, monkeypatch):
    path = inputs[1]

    def rejected(*args, **kwargs):
        raise ValueError("release rejected before execution")

    monkeypatch.setattr(public, "verify_release", rejected)
    with pytest.raises(ValueError, match="release rejected"):
        public.preflight(path)


def test_formal_authorization_rejects_verify_duck_type():
    with pytest.raises(ValueError, match="ReleaseVerification instance"):
        policy_entry.prepare(
            [], None, {}, metrics=None, solver=None, verification=NS(verify=lambda: None)
        )


@pytest.mark.parametrize("reference", [False, True])
def test_score_bad_output_has_parser_exit_two(tmp_path, reference):
    output = tmp_path / "score.json"
    output.touch()
    with pytest.raises(SystemExit) as result:
        benchmark.main(
            [
                "score",
                "--run",
                str(tmp_path / "run"),
                "--task",
                "task1-2",
                "--output",
                str(output),
                *(["--reference"] if reference else []),
            ]
        )
    assert result.value.code == 2


def test_describe_does_not_require_simulator_metadata(monkeypatch, capsys):
    monkeypatch.setattr(
        public,
        "package_identity",
        lambda: pytest.fail("CPU describe requested installed simulator metadata"),
    )
    assert benchmark.main(["describe"]) == 0
    envelope = json.loads(capsys.readouterr().out)["public"]
    assert "revision" not in envelope
    assert envelope["benchmark"] == "2026.11.1"


def test_replay_reference_is_not_abbreviated():
    with pytest.raises(SystemExit):
        benchmark.main(["replay", "--reference", "/anything"])


def test_public_parsers_reject_host_admission_and_abbreviations(tmp_path):
    for flag in ("--allow-desktop-gpu", "--isaac-p"):
        with pytest.raises(SystemExit) as result:
            benchmark.main(["run", "--request", "request.json", "--isaac-python", "python", flag])
        assert result.value.code == 2
    assert not hasattr(public, "admission")


@pytest.mark.parametrize("code", [0, 1])
@pytest.mark.parametrize("portable", [False, True])
def test_public_and_portable_launch_keep_seed_links_on_failure(
    tmp_path, monkeypatch, code, portable
):
    from scripts.maintainer import candidate_policy

    request = dict(output=str(tmp_path / "output"), task="task1-2", seeds=[11, 12], gpu=0)
    path = tmp_path / "request.json"
    save_request(path, request)
    monkeypatch.setattr(public, "preflight", lambda *a, **kw: request)
    monkeypatch.setattr(benchmark, "check_environment", lambda *a: None)

    def child(command, **kw):
        out = Path(request["output"])
        out.mkdir()
        for seed in request["seeds"]:
            (out / "simulation" / f"seed-{seed}").mkdir(parents=True)
        return NS(returncode=code)

    monkeypatch.setattr(public.subprocess, "run", child)
    monkeypatch.setattr(candidate_policy, "finish", lambda *a, **kw: code)
    monkeypatch.setattr(benchmark, "require_eula", lambda *a: None)
    result = (
        importlib.import_module("benchmark_portable").main(["run", "--request", str(path)])
        if portable
        else public.launch(path, "python")
    )
    assert result == code
    for seed in request["seeds"]:
        assert (Path(request["output"]) / f"seed-{seed}").resolve() == Path(
            request["output"]
        ) / "simulation" / f"seed-{seed}"


def test_public_smoke_child_environment_is_clean(tmp_path, monkeypatch, capsys):
    from scripts.maintainer import public_smoke

    args = NS(
        _worker=False,
        output=tmp_path / "output",
        assets=tmp_path,
        task="task1-2",
        seeds=[1],
        gpu=0,
        accept_eula=True,
    )
    for key in public.METRICS_ENV:
        monkeypatch.setenv(key, "stale")
    calls = []

    def child(command, **kw):
        assert not set(public.METRICS_ENV) & set(kw["env"])
        calls.append(command)
        args.output.mkdir()
        (args.output / "smoke-complete.json").write_text('{"passed":true,"phase":"teardown"}')
        return NS(returncode=0)

    monkeypatch.setattr(public_smoke.subprocess, "run", child)
    assert public_smoke.run(args) == 0
    assert "--_worker" in calls[0] and "--reference" not in calls[0]
    capsys.readouterr()


@pytest.mark.golden
@pytest.mark.parametrize("benchmark_version", ["2026.11.1"])
@pytest.mark.archive
def test_diagnostic_default_score_matches_reference(diagnostic_recording, benchmark_version):
    from scripts.maintainer.golden_support import json_value
    from tests.characterization.golden_compare import assert_equivalent
    from tools.simulator.benchmark_diagnostic_score import score_physical_diagnostic

    _, run, task = diagnostic_recording
    from tests.characterization.archived_diagnostic import (
        score_physical_diagnostic as archived_score,
    )

    expected = archived_score(run, task.identity.legacy_id)
    actual = score_physical_diagnostic(
        run, task.identity.legacy_id, benchmark_version=benchmark_version
    )
    assert actual.pop("public") == v.envelope(benchmark=benchmark_version)
    assert_equivalent(json_value(expected), json_value(actual))


def test_reference_route_refuses_even_with_noncheckout_eval(monkeypatch, tmp_path):
    import tower_eval

    monkeypatch.setattr(tower_eval, "__file__", str(tmp_path / "tower_eval/__init__.py"))
    with pytest.raises(RuntimeError, match="maintainers reproduce it from the private archive"):
        benchmark.main(["describe", "--reference"])


def test_worker_does_not_import_characterization_helpers():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.maintainer.candidate_policy; import sys; assert 'scripts.maintainer.golden_support' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("gate", ["verify_data", "verify_release"])
def test_public_replay_identity_runs_verification_before_runtime(gate, monkeypatch, tmp_path):
    from scripts.maintainer import candidate_replay as replay

    monkeypatch.setattr(
        replay, "reference_contract", lambda: {"contract": {"hardware_runtime_profile": {}}}
    )
    monkeypatch.setattr(replay, "data_identity", lambda *a, **kw: {})
    monkeypatch.setattr(replay, "runtime_profile", lambda gpu: {})
    monkeypatch.setattr(
        replay, "execution_contract", lambda *a, **kw: {"hardware_runtime_profile": {}}
    )
    monkeypatch.setattr(v, "verify_data", lambda *a, **kw: {})

    def reject(*a, **kw):
        raise ValueError("rejected " + gate)

    monkeypatch.setattr(
        v, "verify_release", lambda *a, **kw: pytest.fail("missed earlier data gate")
    )
    monkeypatch.setattr(v, gate, reject)
    with pytest.raises(ValueError, match="rejected " + gate):
        replay.CandidateRunner(root=tmp_path, public=True).identity(tmp_path, 0, (), None)


def test_public_setup_uses_nonreference_scene_path(monkeypatch, tmp_path):
    from scripts.maintainer import candidate_replay as replay

    monkeypatch.setattr(v, "verify_data", lambda *a, **kw: {})
    monkeypatch.setattr(v, "verify_release", lambda *a, **kw: {})
    monkeypatch.setattr(replay, "RuntimeProvenance", lambda *a, **kw: None)
    assets = importlib.import_module("benchmark_assets")

    def prepare(*args, **kwargs):
        assert kwargs.get("reference") is False
        raise ValueError("nonreference scenes selected")

    monkeypatch.setattr(assets, "prepare_scenes", prepare)
    with pytest.raises(ValueError, match="nonreference scenes selected"):
        replay.CandidateRunner(root=tmp_path, public=True).setup(
            NS(assets=tmp_path, output=tmp_path, task="task1-2", seed=1)
        )


def test_public_setup_verifies_release_before_construction(monkeypatch, tmp_path):
    from scripts.maintainer import candidate_replay as replay

    monkeypatch.setattr(v, "verify_data", lambda *a, **kw: {})

    def reject(*a, **kw):
        raise ValueError("unverified setup release")

    monkeypatch.setattr(v, "verify_release", reject)
    with pytest.raises(ValueError, match="unverified setup release"):
        replay.CandidateRunner(root=tmp_path, public=True).setup(NS(assets=tmp_path))


def test_public_diagnostic_retains_inputs_under_output(tmp_path, monkeypatch, capsys):
    output = tmp_path / "diagnostic"
    args = NS(output=output, seed=1, task="task1-2", accept_eula=True)
    monkeypatch.setattr(benchmark, "require_eula", lambda *a: None)
    identity = v.envelope(dict(commit="a" * 40, manifest_sha256="b" * 64))

    def prepare(inputs, *, diagnostic):
        assert diagnostic and inputs.output == output / "inputs"
        inputs.output.mkdir(parents=True)
        return inputs.output / "request.json"

    def launch(path, python, *, diagnostic, cache_options):
        assert cache_options is args
        assert path == output / "inputs/request.json" and diagnostic
        (output / "diagnostic-result.json").write_text(
            json.dumps(
                dict(
                    evidence_complete=True,
                    diagnostic_only=True,
                    prefix_only=False,
                    benchmark_deadline_s=66,
                    score={"prefix_only": False, "scores": {"s": 0}},
                )
            )
        )
        (output / "implementation-identity.json").write_text(json.dumps(dict(public=identity)))
        return 0

    monkeypatch.setattr(public, "prepare", prepare)
    monkeypatch.setattr(public, "launch", launch)
    assert public.diagnostic(args) == 0
    result = json.loads((output / "result.json").read_text())
    assert result == dict(
        evidence_complete=True,
        diagnostic_only=True,
        score={"scores": {"s": 0}},
        canonical_task_id=benchmark.resolve(args.task).id,
        public=identity,
    )
    assert not output.with_name("diagnostic-inputs").exists()
    assert json.loads(capsys.readouterr().out) == result


def test_replay_parser_disables_abbreviated_options(monkeypatch):
    from scripts.maintainer import gpu_baseline

    monkeypatch.setattr(
        gpu_baseline, "run", lambda args: pytest.fail("abbreviated option reached GPU harness")
    )
    with pytest.raises(SystemExit) as result:
        benchmark.main(
            ["replay", "--assets", "a", "--expert-root", "b", "--output", "c", "--ref", "unbound"]
        )
    assert result.value.code == 2


def test_preflight_rejects_noncanonical_protocol_identity(inputs):
    _, path, _, _, _ = inputs
    request = json.loads(path.read_text())
    update_protocol(request, lambda settings: settings["ik"].update(shared_spec_sha256="0" * 64))
    save_request(path, request)
    with pytest.raises(ValueError, match="Protocol changes the canonical shared IK identity"):
        public.preflight(path)


def test_preflight_rejects_scene_manifest_without_benchmark(inputs):
    request_path = inputs[1]
    request = json.loads(request_path.read_text())
    path = Path(request["manifest"])
    body = json.loads(path.read_text())
    body.pop("benchmark")
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="must declare its benchmark revision"):
        public.preflight(request_path)


def test_diagnostic_default_enters_public_data_preflight(monkeypatch, tmp_path):
    from tools.simulator.benchmark_diagnostic_score import score_physical_diagnostic

    def preflight(*args, **kwargs):
        raise ValueError("public data preflight reached")

    monkeypatch.setattr(v, "verify_data", preflight)
    with pytest.raises(ValueError, match="public data preflight reached"):
        score_physical_diagnostic(tmp_path, "task1-2")
