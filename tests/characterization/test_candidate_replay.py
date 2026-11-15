"""Maintainer candidate identity must bind real bytes and collect before close."""

from copy import deepcopy
from types import SimpleNamespace as NS
from typing import Any

import pytest

from scripts.maintainer import candidate_replay as candidate
from scripts.maintainer.gpu_runner import resolve_candidate


def test_runner_is_available_without_isaac_startup():
    runner = resolve_candidate()
    assert isinstance(runner, candidate.CandidateRunner)


@pytest.mark.public_assets
def test_gpu_preflight_accepts_only_pinned_public_manifest_identity(tmp_path, monkeypatch):
    from tower_sim.bench.asset_identity import CANONICAL_ASSET_MANIFEST_SHA256

    from scripts.maintainer import gpu_baseline
    from scripts.maintainer.fetch_public_test_data import public_asset_root

    binding = dict(
        candidate_identity=dict(frozen_components=[]),
        contract=dict(immutable_data=dict(assets_manifest_sha256=CANONICAL_ASSET_MANIFEST_SHA256)),
    )
    runner = NS(identity=lambda *args: binding)
    monkeypatch.setattr(gpu_baseline, "resolve_candidate", lambda: runner)
    monkeypatch.setattr(gpu_baseline, "candidate_summary", lambda *args: None)
    args = NS(
        implementation="tower-sim",
        assets=tmp_path,
        gpu=0,
        frozen_component=(),
        reference_root=None,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes((public_asset_root() / "manifest.json").read_bytes())
    assert gpu_baseline.candidate_preflight(args) == (runner, binding)
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Candidate assets differ"):
        gpu_baseline.candidate_preflight(args)


def test_policy_parent_rejects_kit_exit_zero_without_evidence(tmp_path, monkeypatch):
    import json

    from scripts.maintainer import candidate_policy, gpu_baseline

    request = dict(
        task="task1-2",
        seeds=[1],
        output=str(tmp_path / "output"),
        assets=str(candidate.ROOT),
        gpu=0,
    )
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    admitted = []
    monkeypatch.setattr(
        gpu_baseline, "gpu_idle", lambda gpu, desktop: admitted.append((gpu, desktop))
    )
    monkeypatch.setattr(candidate_policy, "subprocess", NS(run=lambda *a, **kw: NS(returncode=0)))
    monkeypatch.setenv("TOWER_GPU_LOCK", str(tmp_path / "gpu.lock"))
    assert (
        candidate_policy.main(
            ["run", "--request", str(path), "--ik-python", "unused", "--diagnostic"]
        )
        == 1
    )
    receipt = json.loads((tmp_path / "output/diagnostic-result.json").read_text())
    assert admitted == [(0, False)]
    assert not receipt["evidence_complete"] and receipt["diagnostic_only"]
    assert not receipt["formal_eligible"] and not receipt["pilot_approved"]


def test_policy_parent_requires_formal_completion(tmp_path):
    import json

    from scripts.maintainer.candidate_policy import finish

    request = dict(output=str(tmp_path), seeds=[1])
    assert finish(request, 0, diagnostic=False) == 1
    run = tmp_path / "simulation/seed-1"
    run.mkdir(parents=True)
    for path in (
        tmp_path / "worker-complete.json",
        tmp_path / "simulation/batch-policy-results.json",
    ):
        path.write_text(json.dumps(dict(completed=True)))
    (run / "status.json").write_text(json.dumps(dict(valid_trial=True, evidence_complete=True)))
    assert finish(request, 0, diagnostic=False) == 0
    assert finish(request, 7, diagnostic=False) == 7


def test_candidate_rejects_wrong_data_before_runtime(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(candidate, "runtime_profile", lambda *a: called.append("runtime"))
    with pytest.raises(FileNotFoundError):
        candidate.CandidateRunner(root=tmp_path).identity(tmp_path, 2, (), None)
    assert called == []


def test_wrong_runtime_profile_rejects_before_build(tmp_path, monkeypatch):
    contract = deepcopy(candidate.reference_contract()["contract"])
    profile = dict(contract["hardware_runtime_profile"], driver="wrong")
    monkeypatch.setattr(candidate, "data_identity", lambda *a, **kw: contract["immutable_data"])
    monkeypatch.setattr(candidate, "runtime_profile", lambda *a: profile)
    monkeypatch.setattr(
        candidate, "build_identity", lambda *a: pytest.fail("must reject before build")
    )
    with pytest.raises(ValueError, match="driver"):
        candidate.CandidateRunner(root=tmp_path).identity(tmp_path, 2, (), None)


def test_incomplete_artifacts_reject_before_collection(tmp_path):
    import json

    trial = tmp_path / "simulation/seed-1"
    trial.mkdir(parents=True)
    (trial / "status.json").write_text(json.dumps({"evidence_complete": False}))
    (tmp_path / "simulation/batch-policy-results.json").write_text(json.dumps({"completed": True}))
    with pytest.raises(ValueError, match="incomplete"):
        candidate.collect(tmp_path, {"output": trial})


def test_prepared_close_shuts_sink_before_app():
    events = []
    prepared: Any = candidate.PreparedReplay(
        None,
        NS(close=lambda: events.append("app.close")),
        [],
        None,
        NS(close=lambda: events.append("receipt.close")),
    )
    prepared.sink = NS(close=lambda: events.append("sink.close"))
    prepared.close()
    assert events == ["receipt.close", "sink.close", "app.close"]


@pytest.mark.archive
def test_all_explicit_substitution_names_validate_against_reference():
    from scripts.maintainer import candidate_substitutions as substitutions
    from scripts.maintainer.archive_root import archive_bundle
    from scripts.maintainer.gpu_identity import COMPONENTS

    bundle = archive_bundle()
    assert substitutions.validate(bundle.parents[1], COMPONENTS) == bundle
    with pytest.raises(ValueError, match="known components"):
        substitutions.validate(candidate.ROOT, ("implicit_fallback",))


def test_native_substitution_rejects_missing_reference_without_process_changes(monkeypatch):
    import os
    import sys

    from scripts.maintainer import candidate_substitutions as substitutions

    before_env, before_path = dict(os.environ), list(sys.path)
    with pytest.raises(ValueError, match="reference checkout"):
        substitutions.native_context(NS(reference_root=None, frozen_components=("cameras",)))
    assert dict(os.environ) == before_env and sys.path == before_path


def test_frozen_function_binding_uses_original_source(tmp_path):
    from scripts.maintainer.candidate_substitutions import function

    source = tmp_path / "original.py"
    source.write_text(
        'raise RuntimeError("unrelated module side effect")\ndef selected(x):\n    return dependency(x) + 1\n'
    )
    fn = function(source, "selected", {"dependency": lambda x: 2 * x})
    assert fn(3) == 7
    assert fn.__code__.co_filename == str(source)
    with pytest.raises(ValueError, match="unique"):
        function(source, "missing", {})


@pytest.mark.archive
def test_execution_contract_does_not_copy_changed_expected_physics():
    pinned = candidate.reference_contract()["contract"]
    actual = candidate.execution_contract(
        candidate.ROOT, pinned["immutable_data"], pinned["hardware_runtime_profile"]
    )
    assert actual == pinned
    changed = deepcopy(pinned)
    changed["physics"]["bootstrap_hz"] = 400
    with pytest.raises(ValueError, match="bootstrap_hz"):
        candidate.require_equal(changed, actual, "contract")


@pytest.fixture
def built_checkout(tmp_path, monkeypatch):
    import subprocess
    import zipfile
    from pathlib import Path

    root, installed = tmp_path / "checkout", tmp_path / "installed"
    root.mkdir()
    (root / ".gitignore").write_text("dist/\n")
    (root / "dist").mkdir()
    names = ("tower_sim", "tower_eval", "nero_kinematics")
    payloads = {}
    for name in names:
        payload = {f"{name}/__init__.py": b"# package\n", f"{name}/module.py": b"# module\n"}
        if name == "tower_eval":
            payload.update(dict.fromkeys(candidate.EVAL_RESOURCES, b"resource"))
        for member, content in payload.items():
            path = root / candidate.wheel_source_path(member)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path = installed / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        payloads[name] = payload

    def write_wheel(name):
        with zipfile.ZipFile(root / "dist" / f"{name}-test.whl", "w") as archive:
            for member, content in payloads[name].items():
                archive.writestr(member, content)

    for name in names:
        write_wheel(name)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=TOWER maintainers",
            "-c",
            "user.email=maintainers@tower-benchmark.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        check=True,
    )
    original = candidate.importlib.import_module
    monkeypatch.setattr(
        candidate.importlib,
        "import_module",
        lambda name: (
            NS(__file__=str(installed / name / "__init__.py")) if name in names else original(name)
        ),
    )
    dependency = tmp_path / "dependency.py"
    dependency.write_text("dependency bytes")
    monkeypatch.setattr(
        candidate.importlib.metadata,
        "distribution",
        lambda name: NS(
            files=[Path("dependency.py")], locate_file=lambda f: dependency, version="test"
        ),
    )
    return NS(root=root, installed=installed, payloads=payloads, write_wheel=write_wheel)


def test_build_identity_records_loaded_paths_and_three_force_includes(built_checkout):
    fixture = built_checkout
    commit, wheels, dependencies, imports = candidate.build_identity(fixture.root)
    assert len(commit) == 40 and len(wheels) == 3 and len(dependencies) == 6
    assert imports == {
        name: str(fixture.installed / name / "__init__.py") for name in fixture.payloads
    }
    assert len(candidate.EVAL_RESOURCES) == 3


def test_build_identity_rejects_untracked_file(built_checkout):
    (built_checkout.root / "untracked.py").write_text("# could run instead\n")
    with pytest.raises(ValueError, match="clean committed checkout"):
        candidate.build_identity(built_checkout.root)


@pytest.mark.parametrize(
    "member",
    [
        "tower_sim/module.py",
        "tower_eval/module.py",
        "nero_kinematics/module.py",
        *candidate.EVAL_RESOURCES,
    ],
)
def test_wheel_and_loaded_copy_cannot_agree_against_commit(built_checkout, member):
    fixture = built_checkout
    name = member.split("/", 1)[0]
    fixture.payloads[name][member] = b"different bytes in both wheel and installed copy"
    (fixture.installed / member).write_bytes(fixture.payloads[name][member])
    fixture.write_wheel(name)
    with pytest.raises(ValueError, match="Wheel differs from committed source"):
        candidate.build_identity(fixture.root)


@pytest.fixture
def prepared_resources(tmp_path, monkeypatch):
    import json

    from tower_sim.bench.resources import Resources, sha

    assets = tmp_path / "assets"
    assets.mkdir()
    member = assets / "asset.json"
    member.write_text("{}")
    manifest = assets / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "asset.json", "size": 2, "sha256": sha(member)}]})
    )
    contract = deepcopy(candidate.reference_contract())
    contract["contract"]["immutable_data"]["assets_manifest_sha256"] = sha(manifest)
    monkeypatch.setattr(candidate, "reference_contract", lambda: contract)
    template = tmp_path / "template"
    template.mkdir()
    runtime = tmp_path / "kinematics"
    runtime.mkdir()
    for name in ("nero.urdf", "nero.yaml", "model.json"):
        (template / name).write_text("fixture resource")
        (runtime / name).write_bytes((template / name).read_bytes())
    resource = Resources({"kinematics": template})
    provenance = candidate.RuntimeProvenance(
        candidate.ROOT, assets, resource, benchmark_version="2026.11.0"
    )
    provenance.bind_runtime(runtime)
    prepared = candidate.PreparedReplay(
        NS(output=tmp_path, frozen_components=()),
        None,
        [],
        None,
        NS(collect=lambda: {"modules": {}, "files": {}}),
        provenance=provenance,
    )
    return NS(prepared=prepared, template=template, runtime=runtime, asset=member)


@pytest.mark.archive
def test_explicit_archive_binds_public_revision_provenance(prepared_resources, monkeypatch):
    import json

    from scripts.maintainer.archive_root import archive_bundle

    fixture = prepared_resources
    old = fixture.prepared.provenance
    bundle = archive_bundle()
    monkeypatch.setattr(candidate.RuntimeProvenance, "scene_files", lambda self: {})
    provenance = candidate.RuntimeProvenance(
        old.root,
        old.assets,
        old.resources,
        benchmark_version="2026.11.1",
        v1_root=bundle,
    )
    assert str(bundle / "protocol.json") in provenance.files
    assert (
        str(bundle.parents[1] / "tests/golden/benchmark-2026.11/MANIFEST.json") in provenance.files
    )
    provenance.bind_runtime(fixture.runtime)
    fixture.prepared.provenance = provenance
    fixture.prepared.collect_sources()
    receipt = json.loads((fixture.prepared.request.output / "imported-sources.json").read_text())
    policy = receipt["validation_policy"]
    assert policy["forbidden_roots"] == [str(bundle / "runtime")]
    assert len(policy["forbidden_stems"]) == 89


@pytest.mark.archive
@pytest.mark.parametrize("location", ["template", "runtime", "asset"])
def test_final_sources_recheck_resources(prepared_resources, location):
    fixture = prepared_resources
    sources = fixture.prepared.collect_sources()
    assert sources[str(fixture.runtime / "nero.urdf")]
    path = fixture.asset if location == "asset" else getattr(fixture, location) / "nero.urdf"
    path.write_text("changed after runtime read")
    with pytest.raises(ValueError, match="runtime_resources|Changed asset"):
        fixture.prepared.collect_sources()


@pytest.mark.archive
@pytest.mark.parametrize("module", ["batch_reference_worker", "runtime_worker", "hidden_alias"])
def test_final_sources_reject_external_frozen_import(
    prepared_resources, module, tmp_path, monkeypatch
):
    fixture = prepared_resources
    path = tmp_path / "other-root/runtime_worker.py"
    import importlib.util
    import sys

    from tower_sim.bench.source_receipt import ImportReceipt

    path.parent.mkdir()
    path.write_text("# frozen worker copied outside the checkout\n")
    spec = importlib.util.spec_from_file_location(module, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module, loaded)
    spec.loader.exec_module(loaded)
    fixture.prepared.receipt.collect = lambda: {
        "files": {},
        "modules": {module: ImportReceipt.module(module)},
    }
    with pytest.raises(ValueError, match="Forbidden candidate import"):
        fixture.prepared.collect_sources()
    fixture.prepared.request.frozen_components = ("scoring",)
    fixture.prepared.collect_sources()  # Explicit diagnostic substitution retains its exemption.


@pytest.mark.parametrize("field,value", [("seed", 2), ("checkpoint_package_digest", "wrong")])
def test_collect_rejects_manifest_binding_mismatch(tmp_path, field, value):
    import json

    trial = tmp_path / "simulation/seed-1"
    trial.mkdir(parents=True)
    status = {"evidence_complete": True, "seed": 1, "checkpoint_package_digest": "a" * 64}
    status[field] = value
    (trial / "status.json").write_text(json.dumps(status))
    (tmp_path / "simulation/batch-policy-results.json").write_text(json.dumps({"completed": True}))
    with pytest.raises(ValueError, match="differently bound"):
        candidate.collect(tmp_path, {"output": trial}, NS(seed=1, expert_sha256="a" * 64))


def test_installed_runner_has_clear_checkout_error(monkeypatch):
    import builtins

    from tower_sim.bench.gpu_baseline import create_runner

    original = builtins.__import__

    def missing(name, *a, **kw):
        if name == "scripts.maintainer.candidate_replay":
            raise ModuleNotFoundError("No scripts", name="scripts")
        return original(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match="maintainer-only; run from a TOWER checkout"):
        create_runner()


@pytest.mark.archive
@pytest.mark.parametrize("stem", ["gpu_baseline", "candidate_policy"])
@pytest.mark.parametrize("qualified", [False, True])
def test_committed_harness_is_allowed_without_allowing_frozen_copies(stem, qualified, tmp_path):
    from tower_sim.bench.resources import sha
    from tower_sim.bench.source_receipt import validate_imports

    bindings = candidate.expected_entrypoints(candidate.ROOT)
    name = f"scripts.maintainer.{stem}" if qualified else "__main__"
    original = candidate.ROOT / f"scripts/maintainer/{stem}.py"
    row = dict(path=str(original), sha256=sha(original), startup_sha256=sha(original))
    kwargs = dict(
        forbidden_stems=candidate.frozen_module_stems(candidate.ROOT) | {stem},
        allowed_modules=bindings,
    )
    validate_imports({"modules": {name: row}}, **kwargs)
    copied = tmp_path / f"{stem}.py"
    copied.write_bytes(original.read_bytes())
    for module, bad in (
        (name, dict(row, path=str(copied))),
        (name, dict(row, sha256="changed")),
        ("hidden_alias", row),
    ):
        with pytest.raises(ValueError, match="Forbidden candidate import"):
            validate_imports({"modules": {module: bad}}, **kwargs)


def test_entrypoint_allowance_rechecks_live_bytes_and_never_overrides_forbidden_root(tmp_path):
    from tower_sim.bench.resources import sha
    from tower_sim.bench.source_receipt import validate_imports

    path = tmp_path / "gpu_baseline.py"
    path.write_text("# authorized entry\n")
    row = dict(path=str(path), sha256=sha(path), startup_sha256=sha(path))
    kwargs = dict(forbidden_stems={path.stem}, allowed_modules={"__main__": [row]})
    with pytest.raises(ValueError, match="Forbidden candidate import"):
        validate_imports({"modules": {"__main__": row}}, forbidden_roots=(tmp_path,), **kwargs)
    path.write_text("# changed since collection\n")
    with pytest.raises(ValueError, match="Forbidden candidate import"):
        validate_imports({"modules": {"__main__": row}}, **kwargs)


@pytest.mark.parametrize("category", ["exts", "extscache", "extsDeprecated"])
def test_kit_extension_namespaces_are_sdk_bound(category, tmp_path, monkeypatch):
    import sys

    from tower_sim.bench.source_receipt import validate_imports

    sdk = tmp_path / "sdk"
    namespace = sdk / category / "camera-extension" / "omni"
    namespace.mkdir(parents=True)
    source = namespace / "capture.py"
    source.write_text("# SDK camera module\n")
    outside = tmp_path / "external/omni"
    outside.mkdir(parents=True)
    copied = outside / "capture.py"
    copied.write_bytes(source.read_bytes())
    linked = sdk / category / "linked-extension"
    linked.mkdir()
    (linked / "omni").symlink_to(outside, target_is_directory=True)
    monkeypatch.setitem(sys.modules, "isaacsim", NS(__file__=str(sdk / "__init__.py")))
    monkeypatch.setitem(sys.modules, "omni", NS(__path__=[]))
    roots = candidate.expected_import_roots(candidate.ROOT)
    assert roots["omni"] == [str(namespace)]
    kwargs = dict(forbidden_stems={"capture"}, allowed_packages=roots)
    validate_imports({"modules": {"omni.kit.capture": {"path": str(source)}}}, **kwargs)
    for name, path in (("omni.kit.capture", copied), ("hidden_alias", source)):
        with pytest.raises(ValueError, match="Forbidden candidate import"):
            validate_imports({"modules": {name: {"path": str(path)}}}, **kwargs)


def test_command_substitution_binds_function_ast(tmp_path):
    from scripts.maintainer.candidate_substitutions import require_same_function

    original, current = tmp_path / "old.py", tmp_path / "new.py"
    original.write_text("EXTRA = 1\ndef joint_commands(x): return x + 1\n")
    current.write_text("EXTRA = 2\ndef joint_commands(x): return x + 1\n")
    require_same_function(original, current, "joint_commands")
    current.write_text("def joint_commands(x): return x + 2\n")
    with pytest.raises(ValueError, match="command mapping differs"):
        require_same_function(original, current, "joint_commands")
