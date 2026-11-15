"""Public interpreter layout and fail-fast source/authorization boundaries."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_public_default import benchmark, public
from test_public_default import release as release
from test_public_paths import inputs as inputs
from test_public_verification import verified_request as verified_request
from tower_sim.bench import policy_entry
from tower_sim.bench import verification as v
from tower_sim.bench.resources import sha


@pytest.mark.parametrize(
    "directory", [".venv-isaac61", ".venv-benchmark-ik", "benchmark-assets", "benchmark-smoke"]
)
def test_guide_layout_allows_untracked_files_outside_bound_trees(release, directory):
    root = release.parent.parent
    path = root / directory / "content.py"
    path.parent.mkdir()
    path.write_text("private content")
    assert v.verify_release(release)["checkout"] == str(root)


@pytest.mark.parametrize("tree", v.BOUND_TREES)
@pytest.mark.parametrize("filename", ["untracked.py", "ignored.pth", "__pycache__/ignored.pyc"])
def test_all_bound_trees_reject_untracked_and_ignored_files(release, tree, filename):
    path = release.parent.parent / tree / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unbound")
    with pytest.raises(ValueError, match="Untracked runtime file"):
        v.verify_release(release)


def test_tracked_changes_outside_bound_trees_are_rejected(release):
    (release.parent.parent / ".gitignore").write_text("changed tracked file")
    with pytest.raises(ValueError, match="clean checkout"):
        v.verify_release(release)


def test_guide_venv_import_allowed_but_checkout_evil_rejected(release):
    root = release.parent.parent
    sdk = root / ".venv-isaac61/lib/python3.12/site-packages"
    native = sdk / "numpy/__init__.py"
    native.parent.mkdir(parents=True)
    native.write_text("# interpreter package")
    identity = v.verify_release(release)
    source = {str(native): sha(native)}
    assert v.verify_sources(source, identity, allowed_roots=[sdk]) == source
    evil = root / "evil.py"
    evil.write_text("# unbound checkout import")
    with pytest.raises(ValueError, match="Unbound public import"):
        v.verify_sources({str(evil): sha(evil)}, identity, allowed_roots=[sdk])


@pytest.mark.parametrize("location", ["ancestor", "same", "child", "symlink"])
def test_sdk_roots_cannot_overlap_bound_tree(release, tmp_path, location):
    identity = v.verify_release(release)
    tree = release.parent.parent / "packages"
    tree.mkdir()
    roots = {"ancestor": tree.parent, "same": tree, "child": tree / "sdk"}
    link = tmp_path / "sdk-link"
    link.symlink_to(tree, target_is_directory=True)
    roots["symlink"] = link
    bound = next(iter(identity["wheel_files"]))
    with pytest.raises(ValueError, match="SDK root overlaps a bound tree"):
        v.verify_sources({bound: sha(bound)}, identity, allowed_roots=[roots[location]])


def test_installed_package_cannot_inherit_purelib_allowance(release):
    identity = v.verify_release(release)
    installed = Path(identity["installed_packages"]["tower-sim"]["path"]).parent
    extra = installed / "unbound.py"
    extra.write_text("# not a wheel member")
    with pytest.raises(ValueError, match="Unbound public import in bound tree"):
        v.verify_sources({str(extra): sha(extra)}, identity, allowed_roots=[installed.parent])


def test_startup_rejects_already_imported_checkout_code(release, monkeypatch):
    identity = v.verify_release(release)
    evil = release.parent.parent / "evil.py"
    evil.write_text("pass")
    monkeypatch.setattr(v, "imported_sources", lambda: {str(evil): sha(evil)})
    with pytest.raises(ValueError, match="Unbound public import"):
        v.verify_startup(identity)


def test_imported_snapshot_contains_already_loaded_file(tmp_path, monkeypatch):
    path = tmp_path / "existing.py"
    path.write_text("# existing")
    monkeypatch.setattr(v, "sys", NS(modules={"p711_already_imported": NS(__file__=str(path))}))
    assert v.imported_sources()[str(path.resolve())] == sha(path)


def reject_startup(*args, **kwargs):
    raise ValueError("startup closure rejected before GPU work")


def test_public_preflight_checks_startup_closure(inputs, monkeypatch):
    monkeypatch.setattr(public, "verify_startup", reject_startup)
    with pytest.raises(ValueError, match="startup closure rejected"):
        public.preflight(inputs[1])


def test_formal_verification_checks_startup_closure(verified_request, monkeypatch):
    monkeypatch.setattr(v, "verify_startup", reject_startup)
    with pytest.raises(ValueError, match="startup closure rejected"):
        verified_request[0].verify()


def test_replay_identity_checks_startup_closure(release, monkeypatch):
    from scripts.maintainer import candidate_replay as candidate

    contract = candidate.reference_contract()["contract"]
    monkeypatch.setattr(candidate, "data_identity", lambda *a, **kw: contract["immutable_data"])
    monkeypatch.setattr(
        candidate, "runtime_profile", lambda *a, **kw: contract["hardware_runtime_profile"]
    )
    monkeypatch.setattr(candidate, "execution_contract", lambda *a, **kw: contract)
    monkeypatch.setattr(v, "verify_data", lambda *a, **kw: {})
    monkeypatch.setattr(v, "verify_release", lambda *a, **kw: {"marker": True})
    monkeypatch.setattr(v, "verify_startup", reject_startup)
    with pytest.raises(ValueError, match="startup closure rejected"):
        candidate.CandidateRunner(root=release.parent.parent, public=True).identity(
            release.parent.parent, 0, (), None
        )


def test_replay_setup_rechecks_startup_before_bootstrap(release, monkeypatch, tmp_path):
    from tower_sim.bench import bootstrap, tasks

    from scripts.maintainer import candidate_replay as candidate

    identity = v.verify_release(release)
    monkeypatch.setattr(v, "verify_data", lambda *a, **kw: {})
    monkeypatch.setattr(v, "verify_release", lambda *a, **kw: identity)
    monkeypatch.setattr(v, "verify_startup", reject_startup)
    monkeypatch.setattr(candidate, "RuntimeProvenance", lambda *a, **kw: None)
    monkeypatch.setattr(tasks.BenchTask, "from_resources", lambda *a, **kw: None)
    monkeypatch.setattr(bootstrap, "bootstrap", lambda **kw: pytest.fail("GPU startup reached"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("TOWER_SOURCE_ROOT", "before-test")
    with pytest.raises(ValueError, match="startup closure rejected"):
        candidate.CandidateRunner(root=release.parent.parent, public=True).setup(
            NS(assets=tmp_path, task="task1-2"), rows=[]
        )


def test_smoke_checks_startup_before_output_or_kit(tmp_path, monkeypatch):
    from scripts.maintainer import public_smoke

    monkeypatch.setattr(public, "verify", lambda **kw: {})
    monkeypatch.setattr(public, "verify_release", lambda *a, **kw: {})
    monkeypatch.setattr(v, "dependency_identity", lambda **kw: {})
    monkeypatch.setattr(public, "verify_startup", reject_startup)
    output = tmp_path / "smoke"
    with pytest.raises(ValueError, match="startup closure rejected"):
        public_smoke.worker(NS(output=output))
    assert not output.exists()


@pytest.mark.parametrize("flag", ["--allow-desktop-gpu", "--allow-desktop-gpu=true"])
def test_public_replay_rejects_maintainer_admission_flag(flag, monkeypatch):
    from scripts.maintainer import gpu_baseline

    monkeypatch.setattr(
        gpu_baseline, "main", lambda *a, **kw: pytest.fail("maintainer admission reached")
    )
    with pytest.raises(SystemExit):
        benchmark.main(["replay", "--assets", "a", "--expert-root", "e", "--output", "o", flag])


def test_formal_policy_message_is_public():
    with pytest.raises(
        ValueError, match="Formal runs require a ReleaseVerification; use diagnostic=True otherwise"
    ):
        policy_entry.prepare([], None, {}, metrics=None, solver=None)


@pytest.mark.parametrize("kind", ["duck", "subclass"])
def test_policy_worker_rejects_forged_verifier_before_any_side_effect(tmp_path, monkeypatch, kind):
    from scripts.maintainer import candidate_policy, kit_safety

    class Forged(v.ReleaseVerification):
        pass

    fake = (
        NS(verify=lambda: None, identity={"forged": True})
        if kind == "duck"
        else object.__new__(Forged)
    )
    monkeypatch.setattr(
        kit_safety, "configure", lambda *a, **kw: pytest.fail("wrote Kit configuration")
    )
    with pytest.raises(ValueError, match="ReleaseVerification instance"):
        candidate_policy.worker(NS(private_kit_root=tmp_path / "kit"), verification=fake)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "arguments", [["--portable-root"], ["--portable-root="], ["--portable-root", "--other"]]
)
def test_portable_root_missing_value_is_clear_error(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["worker", *arguments])
    with pytest.raises(ValueError, match="--portable-root requires a path"):
        v.sdk_roots()


@pytest.mark.parametrize("equals", [False, True])
def test_portable_root_both_forms_and_explicit_override(tmp_path, monkeypatch, equals):
    args = ["--portable-root=" + str(tmp_path)] if equals else ["--portable-root", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", ["worker", *args])
    assert str(tmp_path / "cache/ogn_generated") in v.sdk_roots()
    monkeypatch.setattr(sys, "argv", ["worker", "--portable-root"])
    assert str(tmp_path / "cache/ogn_generated") in v.sdk_roots(generated=tmp_path)


def test_startup_rechecks_layout_after_release_verification(release):
    identity = v.verify_release(release)
    extra = release.parent.parent / "scripts/late.pth"
    extra.write_text("late unbound file")
    with pytest.raises(ValueError, match="Untracked runtime file"):
        v.verify_startup(identity)


def test_release_builder_checks_tracked_changes_before_reading_wheels(release):
    from scripts.maintainer.candidate_release import create

    root = release.parent.parent
    (root / ".gitignore").write_text("changed tracked file")
    with pytest.raises(ValueError, match="clean checkout"):
        create(root, release.parent / "new-release.json")
