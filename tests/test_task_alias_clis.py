# SPDX-License-Identifier: Apache-2.0
"""Keep aliases at unfrozen input boundaries and legacy strings at runtime boundaries."""

from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

import pytest
from tower_eval.cli import main as tower

from scripts.maintainer import archived_benchmark, gpu_baseline
from scripts.maintainer.golden_support import benchmark

portable = importlib.import_module("benchmark_portable")
diagnostic = importlib.import_module("benchmark_diagnostic")
assets = importlib.import_module("benchmark_assets")


@pytest.mark.archive
@pytest.mark.parametrize("alias", ["B2", "two-block-transfer-2", "TASK_000012", "Task 1-2"])
def test_score_alias_keeps_report_bytes_and_deadline(alias, monkeypatch, tmp_path):
    recording = SimpleNamespace(times=[0.0, 125.0])
    expected = {"task": "task1-2", "valid": True, "scores": {"s": 0}}

    def score(row, task, *, success_rule):
        assert row is recording
        assert task == "task1-2"
        assert success_rule == "tower-task12-four-place-hold-v1"
        return expected

    monkeypatch.setattr(
        archived_benchmark,
        "load_scorer",
        lambda: SimpleNamespace(__package__="_p5", score_recording=score),
    )
    monkeypatch.setitem(
        sys.modules,
        "_p5.offline.trajectory",
        SimpleNamespace(DEFAULT_RULES=None, load_recording=lambda *args: recording),
    )
    from tower_eval.scoring import api
    from tower_eval.tasks import benchmark_task

    monkeypatch.setattr(api, "from_files", lambda *args: recording)
    monkeypatch.setattr(
        api,
        "score_recording",
        lambda row, task, **kwargs: score(
            row, benchmark_task(task).legacy_id, success_rule=benchmark_task(task).success_rule
        ),
    )
    output = tmp_path / "score.json"
    assert (
        benchmark.main(
            [
                "score",
                "--run",
                str(tmp_path / "recording"),
                "--task",
                alias,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    from tower_sim.bench.verification import envelope

    exported = {**expected, **api.provenance(alias, None, "files"), "public": envelope()}
    assert output.read_bytes() == (json.dumps(exported, indent=2) + "\n").encode()
    assert "canonical_task_id" not in benchmark.score_run(tmp_path, alias)
    recording.times[-1] = 125.01
    with pytest.raises(ValueError, match="deadline"):
        benchmark.score_run(tmp_path, alias)
    with pytest.raises(ValueError, match="deadline"):
        api.score_run(tmp_path, alias)


@pytest.fixture
def valid_request(tmp_path, monkeypatch):
    def write(name, body):
        path = tmp_path / name
        path.write_text(json.dumps(body, indent=2) + "\n")
        return path

    task = "task1-2"
    seed = next(
        row["seed"]
        for row in benchmark.read(archived_benchmark.BUNDLE / "scene-index.json")["scenes"]
        if row["task"] == task
    )
    spec = archived_benchmark.RUNTIME / "shared/shared-ik-spec.json"
    evidence = [write(name, {}) for name in ("planned.json", "recorded.json")]
    posture = dict(
        schema="tower-no-reversed-grasp-review-v1",
        task=task,
        package_digest="policy-a",
        planned_approach_carry_passed=True,
        recorded_approach_carry_passed=True,
        reverse_180_grasp_absent=True,
        task_contact_geometry_preserved=True,
        **{
            key: {"path": str(path), "sha256": benchmark.sha256(path)}
            for key, path in zip(
                ("planned_trajectory", "recorded_approach_carry"), evidence, strict=True
            )
        },
    )
    review = write("review.json", {"posture_review": posture})
    contract = benchmark.read(archived_benchmark.BUNDLE / "protocol.json")
    policy = dict(contract["policy"])
    policy["hz"] = policy.pop("control_hz")
    protocol = write(
        "protocol.json",
        {
            "policy": policy,
            "pilot_review": {"path": str(review), "sha256": benchmark.sha256(review)},
        },
    )
    files = {
        str((archived_benchmark.BUNDLE / relative).resolve()): binding["sha256"]
        for relative, binding in benchmark.read(archived_benchmark.BUNDLE / "SOURCE_MANIFEST.json")[
            "files"
        ].items()
        if relative.startswith("runtime/")
    }
    files.update({str(path): benchmark.sha256(path) for path in evidence})
    runtime = write("runtime.json", {"files": files})
    manifest = write("scenes.json", {"assets": str(tmp_path), "scenes": [{}]})
    package = write("package.json", {"package_digest": "policy-a"})
    monkeypatch.setattr(
        assets,
        "load_rows",
        lambda path: [{"manifest_entry": {"task": task}, "candidate": {"seed": seed}}],
    )
    monkeypatch.setattr(assets, "manifest", lambda path: {"files": []})
    request = dict(
        task="B2",
        duration=125,
        success_rule=contract["tasks"][task]["success_rule"],
        output=str(tmp_path / "new-run"),
        seeds=[seed],
        portable_assets=str(tmp_path),
        manifest=str(manifest),
        ik_python=sys.executable,
        package=str(package),
        **{
            key: str(tmp_path)
            for key in ("capsule", "live_tools", "startup_environment", "template_runtime")
        },
    )
    for key, path in (("protocol", protocol), ("runtime_manifest", runtime), ("ik_bounds", spec)):
        request[key], request[key + "_sha256"] = str(path), benchmark.sha256(path)
    path = write("request.json", request)
    return path, review


@pytest.mark.archive
def test_preflight_normalizes_only_request_identity(valid_request):
    path, review = valid_request
    original = path.read_bytes()
    request = benchmark.preflight_request(path)
    assert request["task"] == "task1-2"
    with benchmark.runtime_request(path, request) as normalized:
        assert normalized != path
        assert benchmark.read(normalized) == request
        assert benchmark.preflight_request(normalized) == request
    assert not normalized.exists()
    assert path.read_bytes() == original
    # Alias support cannot bypass the task/package-bound posture checks.
    body = benchmark.read(review)
    body["posture_review"]["recorded_approach_carry_passed"] = False
    review.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="Changed pilot review"):
        benchmark.preflight_request(path)


def test_runtime_request_preserves_existing_legacy_file(tmp_path):
    path = tmp_path / "request.json"
    path.write_text('{"task":"task1-2"}')
    with benchmark.runtime_request(path, benchmark.read(path)) as normalized:
        assert normalized == path
    assert path.read_bytes() == b'{"task":"task1-2"}'


@pytest.mark.archive
@pytest.mark.parametrize("entry", ["benchmark", "portable"])
def test_reference_run_refuses_before_dispatch_preserving_request(
    valid_request, monkeypatch, entry
):
    path, _ = valid_request
    original = path.read_bytes()
    monkeypatch.setattr(
        benchmark.subprocess, "run", lambda *a, **kw: pytest.fail("retired reference executed")
    )
    args = ["run", "--reference", "--request", str(path)]
    if entry == "benchmark":
        args.extend(["--isaac-python", sys.executable])
    with pytest.raises(RuntimeError, match="maintainers reproduce it from the private archive"):
        (benchmark if entry == "benchmark" else portable).main(args)
    assert path.read_bytes() == original


@pytest.mark.parametrize("command", ["smoke", "prepare-request"])
def test_portable_cli_resolves_alias_before_dispatch(command, monkeypatch, tmp_path):
    def dispatch(args):
        assert args.task == "task6-3"
        return tmp_path / "request.json"

    monkeypatch.setattr(benchmark, "require_eula", lambda *args: None)
    import benchmark_candidate

    monkeypatch.setattr(benchmark_candidate, "portable", dispatch)
    argv = [
        command,
        "--task",
        "two_block_extraction_3",
        "--assets",
        str(tmp_path),
        "--output",
        str(tmp_path),
        "--seeds",
        "1",
    ]
    if command == "smoke":
        argv.append("--_worker")
    else:
        for flag in (
            "--run-output",
            "--package",
            "--pilot-review",
            "--policy-url",
            "--ik-python",
            "--isaac-python",
        ):
            argv.extend([flag, str(tmp_path)])
    portable.main(argv)


def test_diagnostic_cli_resolves_alias_before_dispatch(monkeypatch, tmp_path):
    def dispatch(args):
        assert args.task == "task4-3"

    monkeypatch.setattr(benchmark, "require_eula", lambda *args: None)
    import benchmark_candidate

    monkeypatch.setattr(benchmark_candidate, "diagnostic", lambda args: dispatch(args) or 0)
    argv = ["--task", "E3", "--seed", "1"]
    for flag in ("--assets", "--output", "--package", "--ik-python", "--policy-url"):
        argv.extend([flag, str(tmp_path)])
    assert diagnostic.main(argv) == 0


@pytest.mark.parametrize("command", ["run", "_worker"])
def test_gpu_baseline_cli_resolves_alias_before_dispatch(command, monkeypatch, tmp_path):
    def dispatch(args):
        assert args.task == "task7-2"

    monkeypatch.setattr(gpu_baseline, "run", dispatch)
    monkeypatch.setattr(gpu_baseline, "worker", dispatch)
    argv = ["gpu_baseline", command, "--task", "F-2"]
    for flag in ("--assets", "--output", "--expert-root"):
        argv.extend([flag, str(tmp_path)])
    monkeypatch.setattr(sys, "argv", argv)
    gpu_baseline.main()


def test_cpu_cli_emits_canonical_metadata_and_legacy_config(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text(
        'episodes = 1\nduration_seconds = 0.1\noutput = "run"\n[environment]\ntask = "B2"\n'
    )
    assert tower(["config-check", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["environment"]["task"] == "tower_transfer_2"
    assert tower(["evaluate", "--config", str(config)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["canonical_task_id"] == "two-block-transfer-2"
    assert summary["config"]["environment"]["task"] == "tower_transfer_2"


def test_cli_unknown_alias_explains_close_matches(capsys):
    with pytest.raises(SystemExit) as exc:
        benchmark.main(
            ["score", "--run", "run", "--task", "two-block-tranfer-3", "--output", "out"]
        )
    assert exc.value.code == 2
    assert "Close matches:" in capsys.readouterr().err


def test_asset_preparation_cli_forwards_legacy_id(monkeypatch, tmp_path):
    def prepare(task, seeds, root, output, *, reference, benchmark_version, data_root, v1_root):
        assert task == "task2-3"
        assert (reference, benchmark_version, data_root, v1_root) == (
            False,
            "2026.11.1",
            None,
            None,
        )
        return output / "scenes.json"

    monkeypatch.setattr(assets, "prepare_scenes", prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assets",
            "prepare-scenes",
            "--task",
            "A3",
            "--assets",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--seeds",
            "1",
        ],
    )
    assets.main()


def test_offline_cli_normalizes_alias_before_scheduling(monkeypatch, tmp_path):
    offline = importlib.import_module("tower_eval.scoring.offline.__main__")
    submitted = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, job):
            submitted.append(job)
            assert job[2] == "task6-2"
            return SimpleNamespace(result=lambda: (job[0], {"task": job[2], "valid": True}))

    monkeypatch.setattr(offline, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(offline, "as_completed", lambda futures: futures)
    assert (
        offline.main(
            ["--run", str(tmp_path / "run"), "--task", "C2", "--output", str(tmp_path / "out")]
        )
        == 0
    )
    assert len(submitted) == 1
    scores = json.loads((tmp_path / "out/scores.json").read_text())
    assert list(scores.values()) == [{"task": "task6-2", "valid": True}]


def test_multi_scene_task_alias(monkeypatch):
    entry = importlib.import_module("benchmark_run")
    captured = []
    monkeypatch.setattr(entry, "run", lambda args: captured.append(args) or 0)
    assert (
        entry.main(
            [
                "run",
                "--task",
                "task6-2",
                "--assets",
                "assets",
                "--package",
                "package.json",
                "--policy-url",
                "ws://localhost:19000",
                "--ik-python",
                "python",
                "--output",
                "output",
            ]
        )
        == 0
    )
    assert benchmark.resolve(captured[0].task).id == "extract-and-stack-2"
    assert captured[0].scenes == list(range(50))
    assert captured[0].pilot_review is None
