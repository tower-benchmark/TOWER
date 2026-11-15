"""Adversarial release, checkout and installed-byte closure regressions."""

import base64
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_public_default import data_tree as data_tree
from test_public_default import release as release
from tower_sim.bench import verification as v
from tower_sim.bench.resources import sha


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "change",
    ["head", "tracked", "untracked", "ignored", "missing-binding", "wrong-binding", "trees"],
)
def test_release_rejects_checkout_drift(release, change):
    body = v.read(release)
    root = release.parent.parent
    if change == "head":
        body["commit"] = "0" * 40
    elif change == "tracked":
        (root / "scripts/bound.py").write_text("changed = True")
    elif change in ("untracked", "ignored"):
        (
            root / "tools/simulator" / ("extra.py" if change == "untracked" else "extra.pth")
        ).write_text("import evil")
    elif change == "missing-binding":
        body["checkout_files"].pop("scripts/bound.py")
    elif change == "wrong-binding":
        body["checkout_files"]["scripts/bound.py"] = "0" * 64
    else:
        body["checkout_trees"] = []
    write(release, body)
    with pytest.raises(ValueError):
        v.verify_release(release)


@pytest.mark.parametrize(
    "member",
    [
        "evil.pth",
        "tower_sim-0.1.0.data/purelib/x.py",
        "unlisted/extra.py",
        "../tower_sim/escape.py",
        "tower_sim/evil.pth",
    ],
)
def test_wheel_rejects_extra_payload_even_with_updated_hash(release, member):
    body = v.read(release)
    entry = body["packages"]["tower-sim"]
    wheel = release.parent / entry["wheel"]
    with zipfile.ZipFile(wheel, "a") as stream:
        stream.writestr(member, "import evil")
    entry["sha256"] = sha(wheel)
    write(release, body)
    with pytest.raises(ValueError, match="Unexpected release wheel member"):
        v.verify_release(release)


def test_installed_extra_file_rejected(release):
    root = release.parent.parent.parent / "installed/tower_sim"
    (root / "unlisted").mkdir()
    (root / "unlisted/extra.py").write_text("unbound = True")
    with pytest.raises(ValueError, match="file set differs"):
        v.verify_release(release)


@pytest.mark.parametrize("escape", ["wheel", "installed"])
def test_release_path_containment(release, tmp_path, escape):
    body = v.read(release)
    if escape == "wheel":
        entry = body["packages"]["tower-sim"]
        wheel = release.parent / entry["wheel"]
        outside = tmp_path / "outside.whl"
        outside.write_bytes(wheel.read_bytes())
        entry["wheel"] = str(outside)
        write(release, body)
    else:
        installed = tmp_path / "installed/tower_sim/__init__.py"
        outside = tmp_path / "outside.py"
        outside.write_bytes(installed.read_bytes())
        installed.unlink()
        installed.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        v.verify_release(release)


def test_data_path_containment(data_tree, monkeypatch):
    manifest_path = Path(v.__file__).with_name("public-data.json")
    body = v.read(manifest_path)
    outside = data_tree.parent / "outside.json"
    outside.write_text("{}")
    body["files"]["../outside.json"] = sha(outside)
    write(manifest_path, body)
    read_bytes = Path.read_bytes
    reads = []

    def record_read(path):
        reads.append(path.resolve())
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)
    with pytest.raises(ValueError, match="escapes"):
        v.verify_data(data_tree)
    assert outside.resolve() not in reads, "containment must be checked before reading"


def test_data_hashes_and_parses_one_read(data_tree, monkeypatch):
    original = Path.read_bytes
    reads = {}

    def once(path):
        if path in (data_tree / "protocol.json", data_tree / "scene-index.json"):
            reads[path] = reads.get(path, 0) + 1
            assert reads[path] == 1, "verified data re-read after hashing"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", once)
    assert v.verify_data(data_tree)["scenes"] == 600
    assert len(reads) == 2


def test_release_wheel_hash_and_zip_share_buffer(release, monkeypatch):
    original = Path.read_bytes
    reads = {}

    def once(path):
        if path == release or path.suffix == ".whl":
            reads[path] = reads.get(path, 0) + 1
            assert reads[path] == 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", once)
    assert v.verify_release(release)["manifest_sha256"] == sha(release)
    assert len(reads) == 4


def test_public_closure_requires_bound_bytes_or_sdk_root(release, tmp_path):
    release = v.verify_release(release)
    source = Path(release["checkout"]) / "scripts/bound.py"
    wheel = next(iter(release["wheel_files"]))
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    native = sdk / "native.so"
    native.write_bytes(b"native sdk")
    good = {str(p): sha(p) for p in (source, Path(wheel), native)}
    assert v.verify_sources(good, release, allowed_roots=[sdk]) == good
    unknown = tmp_path / "unknown.py"
    unknown.write_text("unbound = True")
    with pytest.raises(ValueError, match="Unbound public import"):
        v.verify_sources({str(unknown): sha(unknown)}, release, allowed_roots=[sdk])
    source.write_text("edited after release")
    with pytest.raises(ValueError, match="differs from release"):
        v.verify_sources({str(source): sha(source)}, release, allowed_roots=[sdk])
    native.write_bytes(b"changed since receipt")
    with pytest.raises(ValueError, match="Changed imported file"):
        v.verify_sources(good, release, allowed_roots=[sdk])


def test_checkout_and_package_do_not_inherit_broad_sdk_allowance(release):
    release = v.verify_release(release)
    root = Path(release["checkout"])
    unbound = root / "elsewhere.py"
    unbound.write_text("pass")
    with pytest.raises(ValueError, match="SDK root overlaps a bound tree"):
        v.verify_sources({str(unbound): sha(unbound)}, release, allowed_roots=[root.parent])


def test_entrypoints_are_bound_to_release_not_live_git(release, monkeypatch):
    from scripts.maintainer.candidate_replay import expected_entrypoints

    identity = v.verify_release(release)
    identity["checkout_files"].update({name: "a" * 64 for name in v.ENTRYPOINTS})
    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **kw: pytest.fail("entrypoint rebound to live git")
    )
    entries = expected_entrypoints(Path(identity["checkout"]), identity)
    assert entries["benchmark_candidate"][0]["sha256"] == "a" * 64
    assert {r["sha256"] for r in entries["__main__"]} == {"a" * 64}
    assert len(entries["__main__"]) == 4


@pytest.fixture
def distribution(tmp_path, monkeypatch):
    source = tmp_path / "native.py"
    source.write_text("initial = True")
    record = tmp_path / "native-1.dist-info/RECORD"
    record.parent.mkdir()
    encoded = (
        base64.urlsafe_b64encode(hashlib.sha256(source.read_bytes()).digest()).rstrip(b"=").decode()
    )
    record.write_text(
        f"native.py,sha256={encoded},{source.stat().st_size}\nnative-1.dist-info/RECORD,,\n"
    )
    dist = NS(
        version="1",
        files=[Path("native.py"), Path("native-1.dist-info/RECORD")],
        locate_file=lambda file: tmp_path / str(file),
    )
    monkeypatch.setattr(v.metadata, "distribution", lambda name: dist)
    return source, record, dist


def test_dependency_record_is_checked_against_actual_files(distribution):
    source, _, _ = distribution
    before = v.dependency_identity(("example",))
    assert before["packages"]["example"]["files"]["native.py"] == sha(source)
    source.write_text("tampered while RECORD unchanged")
    with pytest.raises(ValueError, match="differs from RECORD"):
        v.dependency_identity(("example",))


def test_unhashed_python_dependency_is_still_bound(distribution):
    source, record, _ = distribution
    record.write_text("native.py,,\nnative-1.dist-info/RECORD,,\n")
    before = v.dependency_identity(("example",))
    source.write_text("changed unhashed source")
    assert v.dependency_identity(("example",)) != before


def test_generated_bytecode_does_not_replace_verified_dependency_source(distribution):
    source, record, _ = distribution
    cached = source.parent / "__pycache__" / ("native." + v.sys.implementation.cache_tag + ".pyc")
    cached.parent.mkdir()
    cached.write_bytes(b"interpreter cache")
    with record.open("a") as stream:
        stream.write(f"{cached.relative_to(source.parent)},sha256=oldcache,1\n")
    assert v.dependency_identity(("example",))["packages"]["example"]["files"]["native.py"] == sha(
        source
    )
    source.write_text("changed source")
    with pytest.raises(ValueError, match="differs from RECORD"):
        v.dependency_identity(("example",))


@pytest.mark.parametrize(
    "name,version",
    [("pin", "3.8.0"), ("casadi", "3.7.2"), ("pin-pink", "3.4.0"), ("numpy", "1.26.4")],
)
def test_native_ik_versions_are_pinned(distribution, name, version):
    dist = distribution[2]
    dist.version = version
    assert v.dependency_identity((name,))["packages"][name]["version"] == version
    dist.version = "99"
    with pytest.raises(ValueError, match="Unsupported IK dependency"):
        v.dependency_identity((name,))


def test_ik_interpreter_verifies_release_and_all_native_dependencies(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(v, "verify_release", lambda *args, **kw: calls.append((args, kw)))
    monkeypatch.setattr(v, "dependency_identity", lambda names: dict(names=names))
    monkeypatch.setattr(v, "ik_python_sources", lambda: {"python": "bound"})
    v.main(
        ["--ik", "--release", "release.json", "--release-sha256", "bound", "--checkout", "checkout"]
    )
    assert calls == [((Path("release.json"),), dict(checkout=Path("checkout"), expected="bound"))]
    result = json.loads(capsys.readouterr().out)
    assert result["names"] == ["pin", "casadi", "pin-pink", "numpy", "scipy"]
    assert result["python_sources"] == {"python": "bound"}


@pytest.fixture
def verified_request(release, tmp_path, monkeypatch):
    identity = v.verify_release(release)
    dep = tmp_path / "dep.json"
    ik = tmp_path / "ik.json"
    write(dep, {"ordinary": True})
    write(ik, {"native": True})
    request = dict(
        public=v.envelope(identity),
        release_manifest=str(release),
        release_manifest_sha256=sha(release),
        dependency_manifest=str(dep),
        dependency_manifest_sha256=sha(dep),
        ik_dependency_manifest=str(ik),
        ik_dependency_manifest_sha256=sha(ik),
        ik_python="native-python",
        assets="assets",
    )
    monkeypatch.setattr(v, "dependency_identity", lambda: {"ordinary": True})
    original_run = subprocess.run
    calls = []

    def run(command, **kwargs):
        if command[0] != "native-python":
            return original_run(command, **kwargs)
        calls.append(command)
        return NS(stdout=json.dumps({"native": True}))

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(v, "verify_data", lambda root, **kw: {"data": True, "files": {}})
    monkeypatch.setattr(v, "verify_package_data", lambda: {})
    assets = []
    verifier = v.ReleaseVerification(
        checkout=release.parent.parent,
        request=request,
        verify_assets=lambda path: assets.append(path) or {"assets": True},
    )
    return verifier, request, calls, assets


@pytest.mark.parametrize(
    "field", ["release_manifest", "dependency_manifest", "ik_dependency_manifest"]
)
def test_request_binding_rechecked(verified_request, field):
    verifier, request, _, _ = verified_request
    path = Path(request[field])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Changed request binding"):
        verifier.verify()


def test_release_verification_checks_ik_assets_and_envelope(verified_request):
    verifier, request, calls, assets = verified_request
    assert assets == ["assets"]
    assert calls[0][-6:] == [
        "--release",
        request["release_manifest"],
        "--release-sha256",
        request["release_manifest_sha256"],
        "--checkout",
        str(verifier.checkout),
    ]
    assert verifier.provenance["public"] == request["public"]
    assert set(verifier.provenance) == {
        "public",
        "release_manifest_sha256",
        "commit",
        "data_manifest_sha256",
    }
    path = Path(request["ik_dependency_manifest"])
    write(path, {"native": False})
    request["ik_dependency_manifest_sha256"] = sha(path)
    with pytest.raises(ValueError, match="IK dependencies differ"):
        verifier.verify()


def test_release_verification_rejects_forged_envelope(verified_request):
    verifier, request, _, _ = verified_request
    request["public"] = {}
    with pytest.raises(ValueError, match="envelope"):
        verifier.verify()


def test_completed_replay_applies_public_import_closure(release, monkeypatch, tmp_path):
    from scripts.maintainer import candidate_replay as replay

    identity = v.verify_release(release)
    root = Path(identity["checkout"])
    source = root / "scripts/bound.py"
    receipt = dict(
        modules={"bound": dict(path=str(source), sha256=sha(source))},
        files={str(source): sha(source)},
    )
    prepared = replay.PreparedReplay(
        NS(output=tmp_path, frozen_components=()),
        None,
        [],
        None,
        NS(collect=lambda: receipt),
        provenance=NS(root=root, v1_root=None, verify=lambda: {}),
        release=identity,
    )
    monkeypatch.setattr(replay, "expected_entrypoints", lambda *a: {})
    original = v.verify_sources
    calls = []

    def verify(sources, release, **kw):
        calls.append(sources)
        return original(sources, release, **kw)

    monkeypatch.setattr(v, "verify_sources", verify)
    # Its release path is fixed by the public build layout.
    (release.parent / "release-manifest.json").write_bytes(release.read_bytes())
    prepared.collect_sources()
    assert calls == [{str(source): sha(source)}]
    assert v.read(tmp_path / "imported-sources.json")["public_closure"]["passed"]
    unbound = tmp_path / "evil.py"
    unbound.write_text("pass")
    receipt["modules"]["evil"] = dict(path=str(unbound), sha256=sha(unbound))
    with pytest.raises(ValueError, match="Unbound public import"):
        prepared.collect_sources()


@pytest.mark.parametrize("tampered_compat", [False, True])
@pytest.mark.parametrize("gitlink", [False, True, "uninitialized"])
def test_release_builder_lists_committed_runtime_trees(release, tampered_compat, gitlink):
    from scripts.maintainer.candidate_release import create
    from scripts.maintainer.candidate_replay import wheel_source_path

    root = release.parent.parent
    body = v.read(release)
    for name, entry in body["packages"].items():
        package = name.replace("-", "_")
        member = package + "/__init__.py"
        source = root / wheel_source_path(member)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(name)
        old = release.parent / entry["wheel"]
        new = release.parent / (package + "-" + entry["version"] + "-py3-none-any.whl")
        old.rename(new)
        with zipfile.ZipFile(new, "a") as wheel:
            if name == "nero-kinematics":
                alias = root / wheel_source_path("nero_eef_ik/__init__.py")
                alias.parent.mkdir(parents=True)
                alias.write_text("compatibility")
                wheel.writestr(
                    "nero_eef_ik/__init__.py", "tampered" if tampered_compat else "compatibility"
                )
            wheel.writestr(
                f"{package}-{entry['version']}.dist-info/METADATA",
                f"Name: {name}\nVersion: {entry['version']}\n",
            )
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
            "package sources",
        ],
        check=True,
    )
    if gitlink:
        client = root / "third_party/XPolicyLab"
        source = client / "client_server/client.py"
        source.parent.mkdir(parents=True)
        source.write_text("synthetic = True\n")

        def git(directory, *args):
            return subprocess.check_output(["git", "-C", str(directory), *args])

        git(client, "init")
        for directory in (client, root):
            git(directory, "add", ".")
            git(
                directory,
                "-c",
                "user.name=Synthetic",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "-m",
                "synthetic gitlink",
            )
        if gitlink == "uninitialized":
            import shutil

            shutil.rmtree(client)
            client.mkdir()
            with pytest.raises(ValueError, match="Initialize the pinned XPolicyLab submodule"):
                create(root, release.parent / "new-release.json")
            return
    private = root / ".venv-isaac61/untracked.py"
    private.parent.mkdir()
    private.write_text("private interpreter file")
    if tampered_compat:
        with pytest.raises(ValueError, match="Wheel differs from committed source: nero_eef_ik"):
            create(root, release.parent / "new-release.json")
        return
    result = create(root, release.parent / "new-release.json")
    assert result["checkout_trees"] == list(v.CHECKOUT_TREES)
    assert result["checkout_files"] == {name: sha(root / name) for name in v.checkout_files(root)}
    assert set(result["checkout_files"]) == {f"{tree}/bound.py" for tree in v.CHECKOUT_TREES} | (
        {"third_party/XPolicyLab/client_server/client.py"} if gitlink else set()
    )


def test_untracked_file_outside_runtime_trees_is_not_executable_authorization(release):
    path = release.parent.parent / "untracked.py"
    path.write_text("pass")
    identity = v.verify_release(release)
    with pytest.raises(ValueError, match="Unbound public import"):
        v.verify_sources({str(path): sha(path)}, identity)


def test_native_python_sources_include_unrecorded_files(tmp_path, monkeypatch):
    for name in ("pinocchio", "casadi", "pink", "numpy", "scipy"):
        root = tmp_path / name
        root.mkdir()
        (root / "__init__.py").write_text(name)
        (root / "extra.py").write_text("not in RECORD")
    monkeypatch.setattr(
        v.importlib, "import_module", lambda name: NS(__file__=str(tmp_path / name / "__init__.py"))
    )
    files = v.ik_python_sources()
    assert set(files) == {"pinocchio", "casadi", "pink", "numpy", "scipy"}
    for name, paths in files.items():
        assert set(paths) == {"__init__.py", "extra.py"}
        assert paths["extra.py"] == sha(tmp_path / name / "extra.py")


def test_release_binds_declared_ik_compatibility_namespace(release):
    body = v.read(release)
    entry = body["packages"]["nero-kinematics"]
    wheel = release.parent / entry["wheel"]
    installed = release.parent.parent.parent / "installed/nero_eef_ik"
    installed.mkdir()
    (installed / "__init__.py").write_text("compatibility")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("nero_eef_ik/__init__.py", "compatibility")
    entry["sha256"] = sha(wheel)
    write(release, body)
    identity = v.verify_release(release)
    assert identity["wheel_files"][str(installed / "__init__.py")] == sha(installed / "__init__.py")
    (installed / "__init__.py").write_text("tampered")
    with pytest.raises(ValueError, match="differs from release wheel"):
        v.verify_release(release)


def test_formal_verifier_closes_imports_with_verified_resource_bindings(
    release, tmp_path, monkeypatch
):
    from tower_sim.bench.resources import Resources

    identity = v.verify_release(release)
    data_root = tmp_path / "data"
    data_root.mkdir()
    protocol = data_root / "protocol.json"
    protocol.write_text("{}")
    assets = tmp_path / "assets"
    kinematics = assets / "kinematics"
    kinematics.mkdir(parents=True)
    rows = []
    for name in ("nero.urdf", "nero.yaml", "model.json"):
        source = kinematics / name
        source.write_text("{}")
        rows.append(dict(path="kinematics/" + name, sha256=sha(source)))
    manifest = assets / "manifest.json"
    write(manifest, dict(files=rows))
    lock = release.parent.parent / "simulator/benchmark-202611/assets.lock.json"
    lock.parent.mkdir(parents=True)
    write(lock, dict(manifest_sha256=sha(manifest)))
    monkeypatch.setattr(
        Resources,
        "from_roots",
        lambda **kw: Resources(dict(protocol=protocol, asset_lock=lock, kinematics=kinematics)),
    )
    verifier = object.__new__(v.ReleaseVerification)
    verifier.checkout = release.parent.parent
    verifier.request = dict(assets=str(assets), output=str(tmp_path / "run"))
    verifier.identity = dict(release=identity, data=dict(files={str(lock): sha(lock)}))
    bound = next(iter(identity["wheel_files"]))
    assert verifier.verify_sources({bound: sha(bound)}) == {bound: sha(bound)}
    unbound = tmp_path / "unbound.py"
    unbound.write_text("pass")
    with pytest.raises(ValueError, match="Unbound public import"):
        verifier.verify_sources({str(unbound): sha(unbound)})


def test_stdlib_customization_link_allows_only_its_exact_target(release, tmp_path, monkeypatch):
    identity = v.verify_release(release)
    stdlib = tmp_path / "stdlib"
    stdlib.mkdir()
    configuration = tmp_path / "etc/python"
    configuration.mkdir(parents=True)
    customization = configuration / "sitecustomize.py"
    customization.write_text("# distribution configuration")
    (stdlib / "sitecustomize.py").symlink_to(customization)
    monkeypatch.setattr(v.sysconfig, "get_path", lambda name: str(stdlib))
    roots = v.sdk_roots()
    sources = {str(customization): sha(customization)}
    assert v.verify_sources(sources, identity, allowed_roots=roots) == sources
    neighbor = configuration / "unbound.py"
    neighbor.write_text("pass")
    with pytest.raises(ValueError, match="Unbound public import"):
        v.verify_sources({str(neighbor): sha(neighbor)}, identity, allowed_roots=roots)


@pytest.mark.parametrize(
    "change",
    [
        "bytes",
        "manifest-version",
        "module-version",
        "module-path",
        "module-none",
        "missing",
        "escape",
    ],
)
def test_archived_conda_pinocchio_has_verified_native_bytes(tmp_path, monkeypatch, change):
    root = tmp_path / "runtime"
    meta = root / "conda-meta"
    meta.mkdir(parents=True)
    source = root / "pinocchio/__init__.py"
    source.parent.mkdir()
    source.write_text("# native wrapper")
    native = root / "libpinocchio.so"
    native.write_bytes(b"native bytes")
    for name, paths in (
        ("pinocchio", []),
        ("pinocchio-python", [source]),
        ("libpinocchio", [native]),
    ):
        write(
            meta / (name + "-3.8.0.json"),
            dict(
                name=name,
                version="3.8.0",
                files=[str(p.relative_to(root)) for p in paths],
                paths_data=dict(
                    paths=[
                        dict(_path=str(p.relative_to(root)), sha256_in_prefix=sha(p)) for p in paths
                    ]
                ),
            ),
        )

    def no_record(name):
        raise v.metadata.PackageNotFoundError(name)

    module = NS(__version__="3.8.0", __file__=str(source))
    monkeypatch.setattr(v.metadata, "distribution", no_record)
    monkeypatch.setattr(v.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(v.sys, "base_prefix", str(root))
    identity = v.dependency_identity(("pin",))["packages"]["pin"]
    assert identity["files"] == {str(source): sha(source), str(native): sha(native)}
    manifest = meta / "pinocchio-python-3.8.0.json"
    if change == "bytes":
        native.write_bytes(b"changed native bytes")
    elif change == "manifest-version":
        body = v.read(manifest)
        body["version"] = "99"
        write(manifest, body)
    elif change == "module-version":
        module.__version__ = "99"
    elif change == "module-path":
        module.__file__ = str(tmp_path / "unbound.py")
    elif change == "module-none":
        module.__file__ = None
    elif change == "missing":
        manifest.unlink()
    else:
        outside = tmp_path / "outside.py"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
    with pytest.raises(ValueError):
        v.dependency_identity(("pin",))


@pytest.mark.parametrize("name", ["casadi", "numpy", "scipy"])
def test_conda_metadata_without_wheel_record_is_verified(tmp_path, monkeypatch, name):
    source = tmp_path / name / "__init__.py"
    source.parent.mkdir()
    source.write_text("# conda module")
    meta = tmp_path / "conda-meta"
    meta.mkdir()
    relative = str(source.relative_to(tmp_path))
    version = v.IK_PINS.get(name, "1.17.1")
    write(
        meta / (name + "-" + version + ".json"),
        dict(
            name=name,
            version=version,
            files=[relative],
            paths_data=dict(paths=[dict(_path=relative, sha256_in_prefix=sha(source))]),
        ),
    )
    monkeypatch.setattr(v.sys, "base_prefix", str(tmp_path))
    monkeypatch.setattr(v.metadata, "distribution", lambda name: NS(version=version, files=[]))
    monkeypatch.setattr(
        v.importlib, "import_module", lambda name: NS(__version__=version, __file__=str(source))
    )
    assert v.dependency_identity((name,))["packages"][name]["files"] == {str(source): sha(source)}
    source.write_text("changed conda source")
    with pytest.raises(ValueError, match="differs from Conda manifest"):
        v.dependency_identity((name,))


def test_conda_relocated_record_requires_verified_record_and_installed_hashes(
    distribution, monkeypatch
):
    source, record, dist = distribution
    dist.version = v.IK_PINS["numpy"]
    original = dict(sha256=sha(source), size=source.stat().st_size)
    source.write_text("relocated source")
    installed = source.read_bytes()
    record_sha = sha(record)

    def verified_conda(name):
        assert source.read_bytes() == installed
        return dict(files={str(source): sha(source), str(record): record_sha})

    monkeypatch.setattr(v, "conda_ik_identity", verified_conda)
    assert v.dependency_identity(("numpy",))["packages"]["numpy"]["files"]["native.py"] == sha(
        source
    )
    contents = record.read_text()
    encoded = base64.urlsafe_b64encode(bytes.fromhex(original["sha256"])).rstrip(b"=").decode()
    record.write_text(contents.replace(encoded, "unexplained"))
    with pytest.raises(ValueError, match="RECORD or installed bytes differ"):
        v.dependency_identity(("numpy",))


@pytest.mark.parametrize("change", [None, "bytes", "extra", "missing", "symlink"])
def test_shipped_nero_alias_is_bound_as_wheel_code(release, tmp_path, change):
    body = v.read(release)
    entry = body["packages"]["nero-kinematics"]
    wheel = release.parent / entry["wheel"]
    member = "nero_eef_ik/__init__.py"
    with zipfile.ZipFile(wheel, "a") as stream:
        stream.writestr(member, "legacy = True\n")
    entry["sha256"] = sha(wheel)
    write(release, body)
    alias = release.parent.parent.parent / "installed" / member
    alias.parent.mkdir()
    alias.write_text("legacy = True\n")
    if change == "bytes":
        alias.write_text("legacy = False\n")
    elif change == "extra":
        alias.with_name("extra.py").write_text("unbound = True\n")
    elif change == "missing":
        alias.unlink()
    elif change == "symlink":
        outside = tmp_path / "outside.py"
        outside.write_bytes(alias.read_bytes())
        alias.unlink()
        alias.symlink_to(outside)
    if change is not None:
        with pytest.raises(ValueError):
            v.verify_release(release)
    else:
        checked = v.verify_release(release)
        assert checked["wheel_files"][str(alias.resolve())] == sha(alias)
        from scripts.maintainer.candidate_replay import wheel_source_path

        assert wheel_source_path(member) == "packages/nero-kinematics/src/" + member


def test_release_binds_pinned_websocket_client(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()

    def git(directory, *args):
        return subprocess.check_output(["git", "-C", str(directory), *args])

    def commit(directory):
        git(directory, "add", ".")
        git(
            directory,
            "-c",
            "user.name=Synthetic",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "synthetic fixture",
        )

    git(root, "init")
    (root / "scripts").mkdir()
    (root / "scripts/bound.py").write_text("value = 1\n")
    client = root / "third_party/XPolicyLab"
    client.mkdir(parents=True)
    git(client, "init")
    (client / "client_server").mkdir()
    source = client / "client_server/client.py"
    source.write_text("value = 1\n")
    commit(client)
    commit(root)
    relative = "third_party/XPolicyLab/client_server/client.py"
    assert relative in v.checkout_files(root)
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="source must be clean"):
        v.checkout_files(root)
    commit(client)
    with pytest.raises(ValueError, match="pinned gitlink"):
        v.checkout_files(root)
