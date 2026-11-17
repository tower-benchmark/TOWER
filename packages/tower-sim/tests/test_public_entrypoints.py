"""Public Kit workers must enter through committed, startup-bound scripts."""

import importlib
import json
import runpy
from contextlib import nullcontext
from pathlib import Path

import pytest
from tower_sim.bench import verification
from tower_sim.bench.import_guard import matched_archive_stems
from tower_sim.bench.resources import sha
from tower_sim.bench.source_receipt import validate_imports

from scripts.maintainer.candidate_replay import expected_entrypoints

ROOT = Path(__file__).resolve().parents[3]
PORTABLE = "tools/simulator/benchmark_portable.py"


@pytest.mark.parametrize(
    "change",
    [None, "receipt-bytes", "live-bytes", "path", "missing-marker", "wrong-marker", "archive"],
)
def test_public_smoke_main_receipt_is_exactly_bound(tmp_path, change):
    """Use the smoke receipt's __main__ row and the real hashed-stem policy."""
    checkout = tmp_path / "checkout"
    files = {}
    for relative in verification.ENTRYPOINTS:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
        files[relative] = sha(path)
    assert PORTABLE in files
    allowed = expected_entrypoints(checkout, {"checkout_files": files})
    original = checkout / PORTABLE
    row = dict(path=str(original), sha256=sha(original), startup_sha256=sha(original))
    archive = tmp_path / "archive"
    archive.mkdir()
    if change == "receipt-bytes":
        row["sha256"] = "0" * 64
    elif change == "live-bytes":
        original.write_text("# changed after source collection\n")
    elif change in {"path", "archive"}:
        copied = (archive if change == "archive" else tmp_path) / original.name
        copied.write_bytes(original.read_bytes())
        row["path"] = str(copied)
    elif change == "missing-marker":
        row.pop("startup_sha256")
    elif change == "wrong-marker":
        row["startup_sha256"] = "0" * 64
    receipt = {"modules": {"__main__": row}}
    stems = matched_archive_stems(receipt)
    assert "__main__" in stems
    # Public workers have no archive root: even an archived alias must be
    # rejected by the pinned stems plus the exact entry binding alone.
    options = dict(forbidden_roots=[], forbidden_stems=stems, allowed_modules=allowed)
    if change is None:
        validate_imports(receipt, **options)
    else:
        with pytest.raises(ValueError, match="Forbidden candidate import: __main__"):
            validate_imports(receipt, **options)


@pytest.mark.parametrize("entry", verification.ENTRYPOINTS)
def test_public_entries_record_startup_bytes(entry, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools/simulator"))
    namespace = runpy.run_path(str(ROOT / entry), run_name="entry_marker_probe")
    assert namespace["__tower_entry_sha256__"] == sha(ROOT / entry)


@pytest.mark.parametrize(
    "route,entry",
    [
        ("benchmark-run", "tools/simulator/benchmark_candidate.py"),
        ("benchmark-check-request", None),
        ("portable-run", "tools/simulator/benchmark_candidate.py"),
        ("portable-smoke", PORTABLE),
        ("portable-prepare-request", None),
        ("diagnostic", "tools/simulator/benchmark_candidate.py"),
        ("multi-scene", "tools/simulator/benchmark_candidate.py"),
        ("candidate-policy-worker", "scripts/maintainer/candidate_policy.py"),
    ],
)
def test_documented_commands_use_registered_kit_workers(route, entry, tmp_path, monkeypatch):
    """Capture the actual child argv before Kit, rather than assuming a route."""
    monkeypatch.syspath_prepend(str(ROOT / "tools/simulator"))
    benchmark = importlib.import_module("benchmark")
    public = importlib.import_module("benchmark_candidate")
    portable = importlib.import_module("benchmark_portable")
    diagnostic = importlib.import_module("benchmark_diagnostic")
    from scripts.maintainer import candidate_policy

    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    request = dict(
        output=str(tmp_path / "output"),
        task="task4-2",
        seeds=[0],
        gpu=0,
        public={},
        protocol=str(protocol),
        protocol_sha256=sha(protocol),
    )
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(public, "preflight", lambda *a, **kw: request)
    monkeypatch.setattr(benchmark, "check_environment", lambda *a: None)
    monkeypatch.setattr(benchmark, "require_eula", lambda *a: None)
    monkeypatch.setattr(benchmark, "runtime_request", lambda *a: nullcontext(path))
    prepared = []
    monkeypatch.setattr(public, "prepare", lambda *a, **kw: prepared.append(kw) or path)
    from tower_sim.bench import run_bundle

    monkeypatch.setattr(run_bundle, "retain", lambda *a: None)
    calls = []

    class WorkerLaunch(BaseException):
        pass

    def child(argv, **kw):
        calls.append(argv)
        raise WorkerLaunch

    monkeypatch.setattr(public.subprocess, "run", child)

    def invoke():
        if route.startswith("benchmark-"):
            argv = [route.removeprefix("benchmark-"), "--request", str(path)]
            if route == "benchmark-run":
                argv += ["--isaac-python", "python"]
            return benchmark.main(argv)
        if route == "portable-run":
            return portable.main(["run", "--request", str(path)])
        if route == "candidate-policy-worker":
            # The documented _worker enters here directly; no wrapper may
            # replace it with an unregistered Kit entry.
            monkeypatch.setattr(
                candidate_policy,
                "worker",
                lambda *a, **kw: child(["python", candidate_policy.__file__, "_worker"]),
            )
            return candidate_policy.main(
                ["_worker", "--request", str(path), "--ik-python", "python", "--diagnostic"]
            )
        argv = [
            "--assets",
            str(tmp_path),
            "--task",
            "task4-2",
            "--output",
            str(tmp_path / "output"),
        ]
        if route == "multi-scene":
            return importlib.import_module("benchmark_run").main(
                [
                    "run",
                    *argv,
                    "--scenes",
                    "0,1",
                    "--package",
                    "package",
                    "--ik-python",
                    "python",
                    "--policy-url",
                    "ws://localhost",
                ]
            )
        if route == "diagnostic":
            return diagnostic.main(
                argv
                + [
                    "--seed",
                    "0",
                    "--package",
                    "package",
                    "--ik-python",
                    "python",
                    "--policy-url",
                    "ws://localhost",
                ]
            )
        argv += ["--seeds", "0"]
        if route == "portable-prepare-request":
            argv += [
                "--run-output",
                "run",
                "--package",
                "package",
                "--pilot-review",
                "review",
                "--policy-url",
                "ws://localhost",
                "--ik-python",
                "python",
                "--isaac-python",
                "python",
            ]
        return portable.main([route.removeprefix("portable-"), *argv])

    if entry is None:
        assert invoke() == 0
        assert calls == []
    else:
        with pytest.raises(WorkerLaunch):
            invoke()
        assert len(calls) == 1
        script = next(Path(arg).resolve() for arg in calls[0] if str(arg).endswith(".py"))
        assert script == ROOT / entry
        assert entry in verification.ENTRYPOINTS
    assert bool(prepared) == (route in {"portable-prepare-request", "diagnostic", "multi-scene"})
