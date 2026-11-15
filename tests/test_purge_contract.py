"""Public distribution boundaries and preserved evaluation data readers."""

import io
import json
import subprocess
import zipfile
from pathlib import Path

import pytest
from tower_sim.backend import resources
from tower_sim.bench import public_surface, verification
from tower_sim.bench.resources import Resources

from scripts.maintainer.archive_root import ArchiveUnavailable, archive_bundle
from scripts.maintainer.archive_scene import dropped_key_reads
from scripts.maintainer.check_public_tree import check_tree

ROOT = Path(__file__).resolve().parents[1]


def test_public_tree():
    check_tree(ROOT)


@pytest.mark.parametrize(
    "payload",
    [
        b"class GenerationConfig: pass",
        b"def plan_free_space(): pass",
        b"from curobo import Planner",
        b"class TaskLayoutConfig: pass",
        b"from tower_sim.generation import config",
        b"import jenga_generator.motion",
        b"import importlib\nimportlib.import_module('jenga_generator.motion')",
    ],
)
def test_generation_gate_rejects_code(payload):
    with pytest.raises(ValueError, match="Generation"):
        public_surface.check_generation("injected.py", payload)


@pytest.mark.parametrize(
    "name",
    [
        "tower_sim/generation/__init__.py",
        "src/tower-datagenerator/probe.py",
        "jenga_generator/__init__.py",
        "replay_runner_" + "isaac_sim/__init__.py",
    ],
)
def test_generation_gate_rejects_paths(name):
    with pytest.raises(ValueError, match="Generation"):
        public_surface.check_generation(name, b"")


@pytest.mark.parametrize(
    "name,payload",
    [
        ("tower_sim/bench/archive_compat/__init__.py", b""),
        ("simulator/benchmark-202611/receipt.json", b"{}"),
        ("example.py", b"from tower_sim.bench.archive_compat import manifest"),
        ("example.py", b"from . import archive_compat"),
        ("example.py", b"importlib.import_module('tower_sim.bench.archive_compat')"),
    ],
)
def test_archive_gate_rejects_module_and_import(name, payload):
    with pytest.raises(ValueError, match="Archive"):
        public_surface.check_archive(name, payload)


@pytest.mark.parametrize(
    "name,payload",
    [
        ("tower_sim/generation/__init__.py", b""),
        ("tower_sim/bench/archive_compat/__init__.py", b""),
        ("tower_sim/private.txt", ("person@" + "gmail.com").encode()),
    ],
)
def test_wheel_release_gate(name, payload):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as wheel:
        wheel.writestr(name, payload)
    with pytest.raises(ValueError):
        verification.wheel_payload(stream.getvalue(), "tower-sim", "0.1.0")


def test_empty_allowlist_and_personal_email():
    from tests.test_no_internal_leaks import FROZEN_ALLOWLIST

    assert FROZEN_ALLOWLIST == {}
    assert public_surface.violations("person@" + "gmail.com")


def test_generic_dropped_key_alias_is_caught(tmp_path):
    path = tmp_path / "packages/tower-sim/src/tower_sim/probe.py"
    path.parent.mkdir(parents=True)
    path.write_text('def read(prov):\n    return prov.get("physical_fingerprint")\n')
    assert dropped_key_reads(tmp_path)


def test_package_data_pins_and_resource_readers(monkeypatch, tmp_path):
    pins = json.loads((resources.DATA / "pins.json").read_bytes())
    files = verification.verify_package_data()
    assert len(files) == len(pins) == 25
    monkeypatch.setenv("TOWER_SOURCE_ROOT", str(tmp_path))
    assert resources.default_path("motion_profile") == resources.DATA / "motion_profile_v1.json"
    assert (
        resources.resource_path(tmp_path, "calibration_report")
        == resources.DATA / "calibration/robot_dynamics.json"
    )
    resolved = Resources.from_roots(checkout=ROOT, assets=tmp_path)
    for name in (
        "motion_profile",
        "calibration_report",
        "runtime_profiles",
        "shared_ik_spec",
        "camera_profile",
        "scene_cache",
        "asset_lock",
    ):
        assert resolved.path(name).is_relative_to(resources.DATA)
    from tower_sim.backend.profiles import load_profile

    assert load_profile(resolved.path("motion_profile"))["profile_sha256"]


def test_package_data_tamper_refused(monkeypatch, tmp_path):
    original = verification.sha
    monkeypatch.setattr(
        verification,
        "sha",
        lambda p: "0" * 64 if str(p).endswith("motion_profile_v1.json") else original(p),
    )
    with pytest.raises(ValueError, match="Changed evaluation package data"):
        verification.verify_package_data()


def test_archive_absence_skips_and_invalid_root_errors(monkeypatch, tmp_path):
    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    with pytest.raises(ArchiveUnavailable, match="SKIP"):
        archive_bundle()
    with pytest.raises(ValueError, match="Invalid explicit"):
        archive_bundle(tmp_path)


def test_archive_root_is_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    (tmp_path / "protocol.json").write_text("{}")
    assert archive_bundle(tmp_path) == tmp_path.resolve()
    monkeypatch.setenv("TOWER_ARCHIVE_ROOT", str(tmp_path))
    assert archive_bundle() == tmp_path.resolve()


def test_replay_default_cannot_write_to_package():
    from tower_sim.replay.replay_actions import default_output_dir

    with pytest.raises(ValueError, match="explicit output"):
        default_output_dir()
    with pytest.raises(ValueError, match="explicit output"):
        default_output_dir("episode.hdf5")


def test_public_cli_has_no_replay():
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/simulator/benchmark.py"), "replay"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_release_checkout_binding():
    assert verification.CHECKOUT_TREES == ("tools/simulator", "scripts")


@pytest.mark.parametrize(
    "name,payload",
    [
        ("tower_sim/generation/probe.py", b""),
        ("tower_sim/bench/archive_compat/probe.py", b""),
        ("notes.txt", ("/m" + "nt/n" + "as/private").encode()),
        ("notes.txt", ("worker.ecs." + "internal").encode()),
        ("notes.txt", ".".join(("10", "31", "2", "7")).encode()),
        ("notes.txt", ("person@" + "gmail.com").encode()),
    ],
)
def test_integrated_public_surface(name, payload):
    with pytest.raises(ValueError):
        public_surface.check_public(name, payload)


def test_tree_scan_rejects_tracked_payload(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "notes.txt"
    path.write_text("person@" + "gmail.com")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    with pytest.raises(ValueError, match="Private"):
        check_tree(tmp_path)


def test_release_tree_scan_includes_non_runtime_files(tmp_path, monkeypatch):
    path = tmp_path / "notes.txt"
    path.write_text("person@" + "gmail.com")
    monkeypatch.setattr(verification, "verify_checkout_layout", lambda root: None)
    monkeypatch.setattr(verification, "checkout_files", lambda root: [])
    monkeypatch.setattr(
        verification,
        "git",
        lambda root, *args: b"head" if args == ("rev-parse", "HEAD") else b"notes.txt\0",
    )
    with pytest.raises(ValueError, match="Private"):
        verification.verify_checkout(
            tmp_path,
            dict(
                commit="head", checkout_files={}, checkout_trees=list(verification.CHECKOUT_TREES)
            ),
        )


def test_every_resource_reader_uses_installed_pin(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_SOURCE_ROOT", str(tmp_path))
    resolved = Resources.from_roots(checkout=tmp_path, assets=tmp_path)
    expected = {
        "tasks": "registry/tasks.json",
        "asset_lock": "benchmark/assets.lock.json",
        "scene_cache": "benchmark/runtime/ik-capsule/runtime/scene-cache.json",
        "runtime_profiles": "benchmark/runtime/ik-capsule/runtime-profiles",
        "shared_ik_spec": "benchmark/runtime/shared/shared-ik-spec.json",
        "camera_profile": "benchmark/runtime/shared/camera-profile-near10.json",
        "motion_profile": "motion_profile_v1.json",
        "calibration_report": "calibration/robot_dynamics.json",
        "tcp_cache": "calibration/tcp-v2.json",
        "tcp_fit_receipt": "calibration/fit-equivalence.json",
    }
    for name, relative in expected.items():
        assert resolved.path(name) == resources.DATA / relative
    for name, relative in {
        "motion_profile": "motion_profile_v1.json",
        "runtime_profiles": "runtime_profiles_20260902",
        "calibration_report": "calibration/robot_dynamics.json",
    }.items():
        assert resources.default_path(name) == resources.DATA / relative
        assert resources.resource_path(tmp_path, name) == resources.DATA / relative


def test_package_pins_are_bound_to_release_verification(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    for name in ("verify_release", "verify_startup", "dependency_identity", "envelope", "read"):
        monkeypatch.setattr(verification, name, lambda *a, **kw: {})
    monkeypatch.setattr(verification, "verify_data", lambda *a, **kw: {"files": {}})
    monkeypatch.setattr(verification, "verify_package_data", lambda: {"package-pin": "digest"})
    monkeypatch.setattr(verification.subprocess, "run", lambda *a, **kw: NS(stdout="{}"))
    value = verification.ReleaseVerification.__new__(verification.ReleaseVerification)
    value.checkout = tmp_path
    value.request = {
        key: "fixture"
        for key in (
            "release_manifest",
            "release_manifest_sha256",
            "dependency_manifest",
            "dependency_manifest_sha256",
            "ik_dependency_manifest",
            "ik_dependency_manifest_sha256",
            "ik_python",
            "assets",
        )
    }
    value.request["public"] = {}
    value.verify_assets = lambda *a: {}
    assert value.verify()["data"]["files"] == {"package-pin": "digest"}


def test_golden_lookup_uses_explicit_archive(monkeypatch, tmp_path):
    from scripts.maintainer.golden_support import reference_bundle

    (tmp_path / "protocol.json").write_text("{}")
    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    assert reference_bundle(tmp_path) == tmp_path.resolve()
    with pytest.raises(ArchiveUnavailable):
        reference_bundle()


def test_archived_pin_is_checked_before_use(tmp_path):
    import hashlib

    from scripts.maintainer.archive_scene import V1_PREFIX, V1Source

    (tmp_path / "protocol.json").write_text("{}")
    source = V1Source(ROOT, tmp_path)
    pin = hashlib.sha256(b"{}").hexdigest()
    assert source.read(V1_PREFIX + "protocol.json", pin) == b"{}"
    (tmp_path / "protocol.json").write_text("changed")
    with pytest.raises(ValueError, match="Changed immutable"):
        source.read(V1_PREFIX + "protocol.json", pin)


@pytest.mark.parametrize("allow_skip", [False, True])
@pytest.mark.parametrize("module", ["check_golden_exact", "bench_scoring_replay"])
def test_exact_commands_skip_without_archive(module, allow_skip, monkeypatch, tmp_path):
    import os
    import sys

    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    receipt = tmp_path / (module + ".json")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.maintainer." + module,
            "--receipt",
            str(receipt),
            *(["--allow-skip"] if allow_skip else []),
        ],
        cwd=ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == (0 if allow_skip else 77), result.stderr
    assert json.loads(receipt.read_bytes())["skipped"] is True
    assert json.loads(receipt.read_bytes())["passed"] is False


@pytest.mark.archive
def test_maintainer_readers_use_explicit_archive(monkeypatch, tmp_path):
    from copy import deepcopy

    from scripts.maintainer import candidate_replay

    bundle = archive_bundle()
    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    assert candidate_replay.frozen_module_stems(tmp_path, v1_root=bundle)
    protocol = json.loads((bundle / "protocol.json").read_bytes())
    contract = candidate_replay.execution_contract(tmp_path, {}, {}, v1_root=bundle)
    assert contract["task_order"] == list(protocol["tasks"])
    (tmp_path / "manifest.json").write_text('{"files": []}')
    expected = deepcopy(candidate_replay.reference_contract())
    expected["contract"]["immutable_data"]["assets_manifest_sha256"] = verification.sha(
        tmp_path / "manifest.json"
    )
    monkeypatch.setattr(candidate_replay, "reference_contract", lambda: expected)
    assert (
        candidate_replay.data_identity(tmp_path, tmp_path, v1_root=bundle)
        == expected["contract"]["immutable_data"]
    )


@pytest.mark.archive
def test_renamed_module_in_archive_still_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    from scripts.maintainer import candidate_replay

    bundle = archive_bundle()
    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    monkeypatch.setattr(candidate_replay, "expected_entrypoints", lambda *a: {})
    monkeypatch.setattr(candidate_replay, "expected_import_roots", lambda *a: {})
    receipt = NS(
        collect=lambda: {
            "files": {},
            "modules": {
                "unknown_namespace": {
                    "path": str(bundle / "runtime/undeclared_probe.py"),
                    "sha256": "a" * 64,
                },
            },
        }
    )
    prepared = candidate_replay.PreparedReplay(
        NS(output=tmp_path, frozen_components=()),
        None,
        [],
        None,
        receipt,
        provenance=NS(root=tmp_path, v1_root=bundle, verify=lambda: {}),
    )
    with pytest.raises(ValueError, match="Forbidden candidate import"):
        prepared.collect_sources()


def test_portable_has_no_frozen_execution_path(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(ROOT / "tools/simulator"))
    portable = importlib.import_module("benchmark_portable")
    for name in ("install_imports", "configure", "prepare_request", "smoke"):
        assert not hasattr(portable, name)
    with pytest.raises(RuntimeError, match="private archive"):
        portable.main(
            [
                "smoke",
                "--assets",
                ".",
                "--task",
                "task1-2",
                "--seeds",
                "0",
                "--output",
                "unused",
                "--reference",
            ]
        )


def test_tcp_archive_selection_precedes_native_import(monkeypatch):
    from types import SimpleNamespace as NS

    from scripts.maintainer.check_tcp_fit import run

    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    with pytest.raises(ArchiveUnavailable):
        run(NS(archive_root=None))


@pytest.mark.parametrize("name", ["check_golden_exact", "bench_scoring_replay"])
def test_exact_readers_select_archive_corpus(name, tmp_path, monkeypatch):
    import importlib
    import sys

    from scripts.maintainer import archive_root

    module = importlib.import_module("scripts.maintainer." + name)
    (tmp_path / "protocol.json").write_text("{}")
    monkeypatch.setenv("TOWER_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [name, "--receipt", str(tmp_path / "receipt.json")])

    class SelectedArchive(Exception):
        pass

    def selected(root):
        raise SelectedArchive

    monkeypatch.setattr(archive_root, "golden_root", selected)
    with pytest.raises(SelectedArchive):
        module.main()


def test_original_recordings_remain_outside_the_public_tree():
    with pytest.raises(ValueError, match="Archived original recording"):
        public_surface.check_archive("tests/golden/trial.tar.xz", b"")


def test_maintainer_parser_rejects_abbreviations(monkeypatch):
    from scripts.maintainer import gpu_baseline

    monkeypatch.setattr(gpu_baseline, "run", lambda args: None)
    with pytest.raises(SystemExit) as result:
        gpu_baseline.main(["run", "--assets", "a", "--expert-root", "e", "--out", "o"])
    assert result.value.code == 2


@pytest.mark.parametrize(
    "value",
    [
        "paper-" + "isaac51",
        "refs/" + "tags/example",
        "tag " + "benchmark-2026.11.0",
        "benchmark-2026.11.0" + ":path",
        "jenga" + "-desktop",
        "datagen-archive-" + "v1",
        "simulator-assets-" + "v1",
        "simulator-assets-" + "v2",
        "simulator-" + "v2.0.1",
        "git checkout " + "benchmark-2026.11.0",
    ],
)
def test_archive_history_leaks(value):
    assert public_surface.violations(value)
    with pytest.raises(ValueError, match="Private"):
        public_surface.check_public("notes.txt", value.encode())


def test_archive_identity_exception_is_exact():
    name, (value,) = next(iter(public_surface.ARCHIVE_IDENTITY_ALLOWLIST.items()))
    assert not public_surface.violations(value, name=name)
    assert public_surface.violations(value, name="notes.txt")
    assert public_surface.violations(value + " refs/" + "tags/other", name=name)


def test_packaged_registry_and_calibration_bytes():
    source = ROOT / "benchmarks/2026.11"
    assert (source / "tasks.json").read_bytes() == (
        resources.DATA / "registry/tasks.json"
    ).read_bytes()
    for path in (source / "calibration").rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (resources.DATA / path.relative_to(source)).read_bytes()


def test_archive_checkout_refuses_bundle_only_root(tmp_path):
    from scripts.maintainer.archive_root import archive_checkout

    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="full export"):
        archive_checkout(tmp_path)


def test_remove_declared_fields_is_exact(monkeypatch):
    from copy import deepcopy

    from scripts.maintainer import archive_scene

    original = {
        "keep": 3,
        "rows": [{"drop": "metadata", "keep": [1, 2]}, {"keep": 7}],
        "nested": {"drop": 8, "keep": False},
    }
    before = deepcopy(original)
    monkeypatch.setattr(
        archive_scene,
        "field_contract",
        lambda: {"dropped": ["/rows/*/drop", "/nested/drop", "/absent/drop"]},
    )
    assert archive_scene.remove_declared_fields(original) == {
        "keep": 3,
        "rows": [{"keep": [1, 2]}, {"keep": 7}],
        "nested": {"keep": False},
    }
    assert original == before


def test_gpu_runner_rejects_ignored_revision(monkeypatch):
    from types import SimpleNamespace as NS

    from scripts.maintainer import gpu_baseline

    monkeypatch.setattr(
        gpu_baseline, "resolve_candidate", lambda: pytest.fail("Ignored benchmark revision")
    )
    with pytest.raises(ValueError, match="requires --benchmark 2026.11.1"):
        gpu_baseline.candidate_preflight(NS(implementation="tower-sim", benchmark="2026.11.0"))


@pytest.mark.parametrize("allow_skip", [False, True])
@pytest.mark.parametrize("module", ["check_tcp_fit", "scene_v2_release"])
def test_other_archive_commands_signal_skip(module, allow_skip, monkeypatch, tmp_path):
    import os
    import sys

    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    output = tmp_path / "receipt.json"
    arguments = (
        ["--episode", "unused", "--kinematics", "unused", "--stage", "unused"]
        if module == "check_tcp_fit"
        else ["--assets", "unused"]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.maintainer." + module,
            *arguments,
            "--output",
            str(output),
            *(["--allow-skip"] if allow_skip else []),
        ],
        cwd=ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == (0 if allow_skip else 77), result.stderr
    assert json.loads(result.stdout)["skipped"] is True


def test_release_only_binds_existing_source_trees():
    assert verification.BOUND_TREES == ("tools/simulator", "scripts", "packages", "src")


def test_candidate_summary_records_selected_benchmark(monkeypatch):
    from scripts.maintainer import gpu_host_profile

    monkeypatch.setattr(gpu_host_profile, "selected", lambda: None)
    monkeypatch.setattr(gpu_host_profile.original, "candidate_summary", lambda *a: {})
    result = gpu_host_profile.candidate_summary({"benchmark_version": "2026.11.1"}, {}, [])
    assert result["benchmark_version"] == "2026.11.1"


@pytest.mark.parametrize("profile", [None, "desktop"])
def test_summary_requires_measured_benchmark_version(monkeypatch, profile):
    from scripts.maintainer import gpu_host_profile, gpu_identity

    monkeypatch.setattr(gpu_host_profile, "selected", lambda: profile)
    monkeypatch.setattr(gpu_host_profile.original, "candidate_summary", lambda *a: {})
    monkeypatch.setattr(gpu_host_profile, "reference_contract", lambda: {"reference_identity": {}})
    monkeypatch.setattr(gpu_host_profile, "read_summary", lambda document: document)
    identity = {"implementation": "tower-sim"}
    with pytest.raises(ValueError, match="benchmark_version is required"):
        gpu_host_profile.candidate_summary(identity, {}, [])
    with pytest.raises(ValueError, match="benchmark_version is required"):
        gpu_identity.validate_candidate(identity)


@pytest.mark.parametrize(
    "source",
    [
        "value = benchmark.BUNDLE / 'assets.lock.json'",
        "value = RUNTIME",
        "from scripts.maintainer.archive_root import ArchivedPath",
    ],
)
def test_public_route_archive_path_gate(source):
    with pytest.raises(ValueError, match="Archive path in public simulator route"):
        public_surface.check_public("tools/simulator/example.py", source.encode())


def test_public_worker_archive_calls_are_guarded():
    import ast

    forbidden = {
        "archive_bundle",
        "archive_data_identity",
        "data_identity",
        "frozen_module_stems",
        "execution_contract",
        "ArchivedPath",
    }
    guards = {
        ast.dump(ast.parse(expression, mode="eval").body)
        for expression in (
            'self.v1_root is not None or self.benchmark_version == "2026.11.0"',
            "self.provenance.v1_root is not None or "
            'getattr(self.provenance, "benchmark_version", "2026.11.1") == "2026.11.0"',
        )
    }
    tree = ast.parse((ROOT / "scripts/maintainer/candidate_replay.py").read_text())
    aliases = {
        alias.asname: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname
    }

    def check(node, guarded=False):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            assert guarded or aliases.get(name, name) not in forbidden, (
                f"Unguarded archive call at line {node.lineno}: {name}"
            )
        if isinstance(node, ast.If):
            check(node.test, guarded)
            for child in node.body:
                check(child, guarded or ast.dump(node.test) in guards)
            for child in node.orelse:
                check(child, guarded)
        else:
            for child in ast.iter_child_nodes(node):
                check(child, guarded)

    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    for name in ("RuntimeProvenance", "PreparedReplay"):
        check(classes[name])
    setup = next(
        node
        for node in classes["CandidateRunner"].body
        if isinstance(node, ast.FunctionDef) and node.name == "setup"
    )
    check(setup)
    guard = ast.parse((ROOT / "packages/tower-sim/src/tower_sim/bench/import_guard.py").read_text())
    for node in ast.walk(guard):
        if isinstance(node, ast.ImportFrom):
            assert "archive" not in (node.module or "").lower()
            assert not any("archive" in alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.Import):
            assert not any("archive" in alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            assert name not in forbidden
