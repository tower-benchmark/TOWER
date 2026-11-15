"""Golden public CLI and D5 release/data/authorization boundaries."""

import importlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_policy_evaluation import authorization as authorization
from tower_sim.bench import policy_entry, verification
from tower_sim.bench.resources import Resources, sha

from scripts.maintainer import archived_benchmark
from scripts.maintainer.golden_support import benchmark

ROOT = Path(__file__).resolve().parents[3]
public = importlib.import_module("benchmark_candidate")
benchmark_diagnostic = importlib.import_module("benchmark_diagnostic")
benchmark_portable = importlib.import_module("benchmark_portable")


@pytest.mark.archive
def test_describe_cli_golden_except_version_envelope(capsys):
    original = benchmark.read(archived_benchmark.BUNDLE / "protocol.json")
    assert benchmark.main(["describe"]) == 0
    candidate = json.loads(capsys.readouterr().out)
    assert candidate.pop("public") == verification.envelope()
    original["guards"].pop("new_pilot_acceptance_requires")
    original["guards"].pop("reverse_grasp_fallback")

    # Public protocol retains evaluation rules; archived generation guidance is omitted.
    assert candidate["guards"]["command_bounds"] == original["guards"]["command_bounds"]
    assert (
        candidate["guards"]["observed_joint_tolerance_deg"]
        == original["guards"]["observed_joint_tolerance_deg"]
    )
    assert candidate == json.loads((ROOT / "benchmarks/2026.11.1/protocol.json").read_bytes())


def test_verify_cli_routes(capsys, monkeypatch):
    with pytest.raises(RuntimeError, match="maintainers reproduce it from the private archive"):
        benchmark.main(["verify", "--reference"])
    monkeypatch.setattr(
        benchmark, "verify_bundle", lambda: pytest.fail("candidate verified frozen code")
    )
    assert benchmark.main(["verify"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tasks"] == 12 and result["scenes"] == 600
    assert set(result["packages"]) == set(verification.PACKAGES)
    assert result["request_schema"] == verification.CONTRACT
    assert "verified_files" not in result


@pytest.mark.parametrize("entry", ["run", "portable", "diagnostic"])
def test_public_wrappers_dispatch_candidate(entry, tmp_path, monkeypatch):
    calls = []
    if entry == "run":
        monkeypatch.setattr(public, "command", lambda args: calls.append(args.command) or 7)
        result = benchmark.main(["run", "--request", "missing", "--isaac-python", "unused"])
    elif entry == "portable":
        monkeypatch.setattr(public, "portable", lambda args: calls.append(args.command) or 7)
        result = benchmark_portable.main(["run", "--request", "missing"])
    else:
        monkeypatch.setattr(public, "diagnostic", lambda args: calls.append("run") or 7)
        result = benchmark_diagnostic.main(
            [
                "--assets",
                "missing",
                "--output",
                str(tmp_path),
                "--package",
                "missing",
                "--ik-python",
                "unused",
                "--task",
                "task1-2",
                "--seed",
                "1",
                "--policy-url",
                "unused",
            ]
        )
    assert result == 7 and calls == ["run"]


def test_replay_uses_public_factory_and_keeps_policy_replay_rejection(tmp_path, monkeypatch):

    with pytest.raises(SystemExit) as error:
        benchmark.main(["replay", "--assets", "a", "--expert-root", "e", "--output", "o"])
    assert error.value.code == 2
    assert public.create_runner().public is True
    path = tmp_path / "request.json"
    path.write_text(json.dumps(dict(command_replay=True)))
    monkeypatch.setattr(public, "verify", lambda: pytest.fail("replay reached verification"))
    with pytest.raises(ValueError, match="learned-policy"):
        public.preflight(path)


def test_public_smoke_rejects_changed_release_before_kit(tmp_path, monkeypatch):
    from scripts.maintainer import kit_safety, public_smoke

    def changed_release(path):
        raise ValueError("Changed smoke release")

    monkeypatch.setattr(public, "verify", lambda **kw: {})
    monkeypatch.setattr(public, "verify_release", changed_release)
    monkeypatch.setattr(verification, "dependency_identity", lambda: {})
    monkeypatch.setattr(benchmark, "check_environment", lambda python: {})
    monkeypatch.setattr(
        kit_safety, "configure", lambda path: pytest.fail("unverified smoke reached Kit setup")
    )
    output = tmp_path / "smoke"
    with pytest.raises(ValueError, match="Changed smoke release"):
        public_smoke.worker(NS(output=output))
    assert not output.exists()


@pytest.fixture
def data_tree(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    protocol = {"tasks": {str(i): {"scene_count": 50} for i in range(12)}}
    index = {
        "scenes": [
            dict(task=str(i), scene_id=f"scene-{j:03d}") for i in range(12) for j in range(50)
        ]
    }
    for name, body in (
        ("protocol.json", protocol),
        ("scene-index.json", index),
        ("scenes.json", {"physical": True}),
    ):
        (root / name).write_text(json.dumps(body))
    (root / "old-worker.py").write_text("old = True\n")
    package = tmp_path / "package"
    package.mkdir()
    (package / "public-data.json").write_text(
        json.dumps(
            dict(
                data_version=verification.DATA_VERSION,
                protocol="protocol.json",
                scene_index="scene-index.json",
                files={p.name: sha(p) for p in root.glob("*.json")},
            )
        )
    )
    monkeypatch.setattr(verification, "__file__", str(package / "verification.py"))
    return root


def test_data_verification_does_not_freeze_python(data_tree):
    before = verification.verify_data(data_tree)
    (data_tree / "old-worker.py").write_text("source_refactored = True\n")
    assert verification.verify_data(data_tree) == before


def test_scene_and_runtime_bindings_cover_consumed_checkout_resources(tmp_path):
    from tower_sim.backend.resources import DATA
    from tower_sim.bench.verification import verify_package_data

    bindings = verification.verify_data(ROOT)["files"] | verify_package_data()
    resources = Resources.from_roots(checkout=ROOT, assets=tmp_path / "assets")
    checked = []
    for path in resources.paths.values():
        paths = path.rglob("*") if path.is_dir() else (path,)
        for member in paths:
            if (member.is_relative_to(ROOT) or member.is_relative_to(DATA)) and member.suffix in (
                ".json",
                ".yaml",
                ".yml",
                ".urdf",
            ):
                assert bindings[str(member.resolve())] == sha(member)
                checked.append(member.resolve())
    assert (DATA / "calibration/robot_dynamics.json").resolve() in checked


@pytest.mark.parametrize("name", ["protocol.json", "scene-index.json", "scenes.json"])
def test_data_drift_rejected(data_tree, name):
    path = data_tree / name
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Changed benchmark data"):
        verification.verify_data(data_tree)


@pytest.fixture
def release(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".gitignore").write_text("dist/\n__pycache__/\n*.pth\n")
    tracked = {}
    for tree in verification.CHECKOUT_TREES:
        source = root / tree / "bound.py"
        source.parent.mkdir(parents=True)
        source.write_text("original = True\n")
        tracked[source.relative_to(root).as_posix()] = sha(source)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dist = root / "dist"
    dist.mkdir()
    packages = {}
    for name, version in verification.PACKAGES.items():
        path = dist / (name + ".whl")
        package = name.replace("-", "_")
        with zipfile.ZipFile(path, "w") as wheel:
            wheel.writestr(package + "/__init__.py", name)
        installed = tmp_path / "installed" / package
        installed.mkdir(parents=True)
        (installed / "__init__.py").write_text(name)
        packages[name] = dict(wheel=path.name, sha256=sha(path), version=version)
    path = dist / "release.json"
    path.write_text(
        json.dumps(
            dict(
                schema="tower-release-v2",
                public=verification.envelope(),
                commit=commit,
                checkout_trees=list(verification.CHECKOUT_TREES),
                checkout_files=tracked,
                packages=packages,
            )
        )
    )
    monkeypatch.setattr(
        verification,
        "package_identity",
        lambda: {
            name: dict(
                version=version,
                path=str(tmp_path / "installed" / name.replace("-", "_") / "__init__.py"),
            )
            for name, version in verification.PACKAGES.items()
        },
    )
    monkeypatch.setattr(
        verification,
        "imported_sources",
        lambda: {str(root / name): value for name, value in tracked.items()},
    )
    return path


@pytest.mark.parametrize(
    "change", [None, "wheel", "installed", "version", "missing-package", "commit"]
)
def test_release_identity_binds_all_packages(release, change):
    body = json.loads(release.read_text())
    if change == "wheel":
        with zipfile.ZipFile(release.parent / body["packages"]["tower-sim"]["wheel"], "a") as wheel:
            wheel.comment = b"changed metadata, unchanged installed payload"
    elif change == "installed":
        (release.parent.parent.parent / "installed/tower_sim/__init__.py").write_text("changed")
    elif change == "version":
        body["packages"]["tower-sim"]["version"] = "99.0"
    elif change == "missing-package":
        body["packages"].pop("nero-kinematics")
    elif change == "commit":
        body["commit"] = "unknown"
    release.write_text(json.dumps(body))
    if change:
        with pytest.raises(ValueError):
            verification.verify_release(release)
    else:
        identity = verification.verify_release(release)
        assert set(identity["installed_packages"]) == set(verification.PACKAGES)
        assert identity["manifest_sha256"] == sha(release)


def test_installed_package_version_is_required(monkeypatch):
    monkeypatch.setattr(verification.metadata, "version", lambda name: "99")
    with pytest.raises(ValueError, match="Unsupported package version"):
        verification.package_identity()


def test_dependency_drift_rejected(release, monkeypatch, tmp_path):
    dependency = tmp_path / "dependencies.json"
    dependency.write_text('{"packages": {"numpy": "original"}}')
    request = dict(
        release_manifest=str(release),
        release_manifest_sha256=sha(release),
        dependency_manifest=str(dependency),
        dependency_manifest_sha256=sha(dependency),
        ik_dependency_manifest=str(dependency),
        ik_dependency_manifest_sha256=sha(dependency),
        public=verification.envelope(verification.verify_release(release)),
        assets="unused",
        ik_python="unused",
    )
    monkeypatch.setattr(
        verification, "dependency_identity", lambda: {"packages": {"numpy": "changed"}}
    )
    monkeypatch.setattr(verification, "verify_data", lambda root: {})
    original_run = verification.subprocess.run
    monkeypatch.setattr(
        verification.subprocess,
        "run",
        lambda command, **kw: (
            NS(stdout=dependency.read_text())
            if command[0] == "unused"
            else original_run(command, **kw)
        ),
    )
    with pytest.raises(ValueError, match="Installed dependencies differ"):
        verification.ReleaseVerification(
            checkout=release.parent.parent, request=request, verify_assets=lambda p: {}
        )


def test_public_formal_prepare_injects_release_verification(authorization, monkeypatch):
    request, health, rows, task = authorization
    request.update(policy_url="unused", prompt="prompt", task="task1-2")
    calls = []
    verifier = object.__new__(verification.ReleaseVerification)
    verifier.request = request
    monkeypatch.setattr(
        verification.ReleaseVerification, "verify", lambda self: calls.append("verify")
    )
    monkeypatch.setattr(
        verification.ReleaseVerification,
        "verify_sources",
        lambda self, sources: calls.append("closure") or sources,
    )
    monkeypatch.setattr(policy_entry, "validate_entry", lambda *a: None)
    worker = NS(authorize=lambda: health, close=lambda: None)
    monkeypatch.setattr(policy_entry.policy, "from_package", lambda **kw: worker)
    session, context = policy_entry.prepare(
        rows, task, request, metrics=NS(call=lambda _, fn: fn()), solver=None, verification=verifier
    )
    assert session is worker and context.verification is verifier
    assert calls == ["verify"]
    context.closure({"new-package-source.py": "not-a-frozen-authorization"})
    assert calls == ["verify", "verify", "closure"]
    Path(request["package"]).write_text("{}")
    with pytest.raises(ValueError, match="Runtime input changed"):
        context.closure({"new-package-source.py": "receipt"})


def test_public_launch_removes_metrics_env_and_requires_parent_completion(tmp_path, monkeypatch):
    request = dict(output=str(tmp_path / "out"), seeds=[1], gpu=0, task="task1-2")
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    calls = []
    for name in public.METRICS_ENV:
        monkeypatch.setenv(name, "legacy-value")
    monkeypatch.setattr(public, "preflight", lambda *a, **kw: request)
    monkeypatch.setattr(benchmark, "check_environment", lambda *a: None)
    monkeypatch.setattr(
        public.subprocess, "run", lambda *a, **kw: calls.append(kw) or NS(returncode=0)
    )
    assert public.launch(path, "python") == 1
    assert not set(public.METRICS_ENV) & calls[0]["env"].keys()


def test_public_launch_normalizes_alias_without_rewriting_input(tmp_path, monkeypatch):
    request = dict(output=str(tmp_path / "out"), seeds=[1], gpu=0, task="B2")
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    original = path.read_bytes()
    monkeypatch.setattr(public, "preflight", lambda *a, **kw: dict(request, task="task1-2"))
    monkeypatch.setattr(benchmark, "check_environment", lambda *a: None)
    seen = []

    def launch(command, **kwargs):
        child = Path(command[command.index("--request") + 1])
        assert child != path and json.loads(child.read_text())["task"] == "task1-2"
        seen.append(child)
        return NS(returncode=1)

    monkeypatch.setattr(public.subprocess, "run", launch)
    assert public.launch(path, "python") == 1
    assert path.read_bytes() == original and seen and not seen[0].exists()


def test_import_has_no_metrics_io(tmp_path):
    destination = tmp_path / "metrics.jsonl"
    subprocess.run(
        [sys.executable, "-c", "import tower_sim.bench.verification, tower_sim.bench.policy_entry"],
        cwd=tmp_path,
        env=dict(os.environ, TOWER_BASELINE_METRICS=str(destination)),
        check=True,
    )
    assert not destination.exists()
