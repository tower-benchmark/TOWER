"""All-scene equality and adversarial scene/data revision boundaries."""

import copy
import importlib
import json
from pathlib import Path

import pytest
from tower_sim.bench import data_release, scene_format, verification
from tower_sim.bench.resources import sha

from scripts.maintainer import archive_scene as archive_compat
from scripts.maintainer import scene_v2_release
from scripts.maintainer.golden_support import benchmark as benchmark
from tests.test_no_internal_leaks import FROZEN_ALLOWLIST, scan_path, scan_scene_v2

ROOT = Path(__file__).resolve().parents[3]
assets_api = importlib.import_module("benchmark_assets")


@pytest.fixture(scope="module")
def archive_release_data(tmp_path_factory, v1_export):
    assets = tmp_path_factory.mktemp("scene-v2-assets")
    files = {}
    for path in (v1_export / "scenes").glob("*.json"):
        for row in json.loads(path.read_bytes())["scenes"]:
            for block in row["candidate"]["episode_spec"]["blocks"]:
                for key in ("asset_sha256", "texture_sha256"):
                    if block.get(key):
                        value = block[key]
                        files[value] = dict(path="objects/" + value, sha256=value)
    (assets / "manifest.json").write_text(
        json.dumps(dict(schema="tower-benchmark-assets-v1", files=list(files.values())))
    )
    data = assets / "benchmark-2026.11.1"
    result = scene_v2_release.stage(
        ROOT, assets, data, v1_root=v1_export, internal_output=assets / "internal"
    )
    assert result["scenes"] == 600
    return assets, data


@pytest.fixture(scope="module")
def release_data(tmp_path_factory):
    import shutil

    from scripts.maintainer.fetch_public_test_data import public_asset_root

    source = public_asset_root()
    assets = tmp_path_factory.mktemp("public-scene-assets")
    shutil.copyfile(source / "manifest.json", assets / "manifest.json")
    data = assets / "benchmark-2026.11.1"
    shutil.copytree(source / "benchmark-2026.11.1", data)
    return assets, data


@pytest.mark.archive
def test_all600_v1_v2_resolution_and_complete_compatibility(archive_release_data, v1_export):
    assets, data = archive_release_data
    old_golden = json.loads(
        (ROOT / "tests/golden/benchmark-2026.11/scene-resolve-sha256.json").read_bytes()
    )
    # Keep the immutable full-output golden, rather than dropping its provenance fields.
    from tests.characterization.scene_resolution import scene_resolution_table

    table = scene_resolution_table(v1_root=v1_export)
    assert table == old_golden
    total = 0
    asset_manifest = assets_api.manifest(assets)
    for path in (v1_export / "scenes").glob("*.json"):
        collection = data_release.read_payload("scenes/" + path.name, assets=assets, data_root=data)
        encoded = {(r["task"], r["scene_id"]): r for r in collection["scenes"]}
        for number, original in enumerate(json.loads(path.read_bytes())["scenes"]):
            task, seed = original["task"], original["seed"]
            body = encoded[(task, f"scene-{number:03d}")]
            # The actual readers, with each pinned collection read once per batch.
            old = archive_compat.relocate_v1(original, assets, asset_manifest)
            new = scene_format.resolve_physical(body, assets, asset_manifest)
            compat = archive_compat.compatibility_view(body, original, assets, asset_manifest)
            assert old == compat
            expected = archive_compat.expected_runtime(original, body, assets, asset_manifest)
            archive_compat.assert_runtime_labels(
                archive_compat.remove_declared_fields(old), expected
            )
            assert expected == new
            assert scene_format.encode_scene(scene_format.decode_scene(body)) == body
            assert all(
                "measured_asset_id" in block for block in scene_format.blocks(new["candidate"])
            )
            total += 1
            # Dispatch every scene, including the complete v1 compatibility view.
            assert assets_api.resolve_scene(task, number, assets, data_root=data) == new
            assert (
                assets_api.resolve_scene(
                    task, seed, assets, data_root=data, compatibility=True, v1_root=v1_export
                )
                == old
            )
    assert total == 600


@pytest.mark.archive
def test_hash_map_detects_tampering(archive_release_data, tmp_path, v1_export):
    assets, _ = archive_release_data
    payloads, metadata = scene_v2_release.generate(ROOT, assets, v1_root=v1_export)
    root = tmp_path / "checkout"
    (root / "benchmarks/2026.11.1").mkdir(parents=True)
    (root / "packages").symlink_to(ROOT / "packages", target_is_directory=True)
    for name, data in metadata.items():
        (root / "benchmarks/2026.11.1" / name).write_bytes(data)
    path = root / "benchmarks/2026.11.1/scene-equivalence.json"
    body = json.loads(path.read_bytes())
    body["scenes"][0]["physical_sha256"] = "0" * 64
    path.write_bytes(scene_v2_release.bytes_for(body))
    with pytest.raises(ValueError, match="mapping/metadata differs"):
        scene_v2_release.stage(root, assets, tmp_path / "output", v1_root=v1_export)
    assert payloads


@pytest.mark.archive
def test_redaction_and_task7_physical_binding(v1_export, archive_release_data):
    original = json.loads((v1_export / "scenes/task7-3.json").read_bytes())["scenes"][0]
    encoded = archive_compat.public_scene(original, "scene-000")
    assert "provenance" not in encoded["episode"]
    assert "source_bootstrap_sha256" not in encoded["initial_state"]
    assert "capsule_files" not in json.dumps(encoded)
    assert "original_capsule" not in json.dumps(encoded)
    scene_format.validate_public(encoded)
    line = archive_compat.lineage(original)
    assert set(line) == {"scene_id", "source_sha256", "content_sha256"}
    assert all(scene_format.HASH.fullmatch(h) for h in line["content_sha256"])
    changed = copy.deepcopy(encoded)
    changed["episode"]["task7_scene"]["source_center_bond"]["world_anchor"] = not changed[
        "episode"
    ]["task7_scene"]["source_center_bond"]["world_anchor"]
    assets, _ = archive_release_data
    with pytest.raises(ValueError, match="Full runtime input"):
        archive_compat.compatibility_view(changed, original, assets, assets_api.manifest(assets))


@pytest.mark.archive
@pytest.mark.parametrize("change", ["pose", "resource", "schema", "provenance"])
def test_v2_rejects_schema_or_binding_drift(change, v1_export):
    original = json.loads((v1_export / "scenes/task1-2.json").read_bytes())["scenes"][0]
    body = archive_compat.public_scene(original, "scene-000")
    if change == "pose":
        body["episode"]["episode_spec"]["blocks"][0]["pose"][0] += 1
    elif change == "resource":
        body["episode"]["episode_spec"]["blocks"][0]["asset_reference"] = "sha256:" + "0" * 64
    elif change == "schema":
        body["schema"] = "unknown"
    else:
        body["episode"]["provenance"] = dict(generation_run="/" + "mnt" + "/n" + "as/private")
    with pytest.raises(ValueError):
        archive_compat.compatibility_view(body, original, Path("unused"), {"files": []})


def test_envelope_revision_switch():
    assert verification.envelope()["benchmark"] == "2026.11.1"
    assert verification.envelope()["data_version"] == "benchmark-2026.11.1"
    assert verification.envelope()["schema"] == "tower-public-envelope-v1"
    assert verification.envelope()["request_schema"] == "tower-public-benchmark-v2"
    with pytest.raises(ValueError, match="Only the public"):
        verification.envelope(benchmark="2026.11.0")
    with pytest.raises(ValueError):
        verification.envelope(benchmark="future")


@pytest.mark.public_assets
def test_payload_manifest_binding(release_data, tmp_path):
    assets, data = release_data
    files = data_release.verify_payloads(assets=assets, data_root=data)
    assert len(files) == 12 and all(sha(p) == h for p, h in files.items())
    changed = tmp_path / "payload"
    (changed / "scenes").mkdir(parents=True)
    name = "scenes/task1-2.json"
    (changed / name).write_bytes((data / name).read_bytes().replace(b"task1-2", b"task1-3"))
    with pytest.raises(ValueError, match="Changed benchmark payload"):
        data_release.read_payload(name, assets=assets, data_root=changed)


@pytest.mark.public_assets
def test_new_release_has_no_leak_exemptions(release_data, tmp_path):
    _, data = release_data
    assert not any(name.startswith("benchmarks/2026.11.1") for name in FROZEN_ALLOWLIST)
    for path in data.rglob("*"):
        if path.is_file():
            assert not scan_path(data, path.relative_to(data).as_posix())
    planted = tmp_path / "benchmarks/2026.11.1/planted.json"
    planted.parent.mkdir(parents=True)
    planted.write_text(json.dumps({"path": "/" + "mnt" + "/n" + "as/private"}))
    assert scan_scene_v2(tmp_path)


@pytest.mark.public_assets
def test_data_manifest_binds_staged_release(release_data):
    assets, data = release_data
    receipt = verification.verify_data(ROOT, assets=assets, data_root=data)
    assert receipt["verified_payload_files"] == 12
    assert not (data / "lineage.json").exists()
    assert all("lineage" not in name for name in receipt["files"])


@pytest.mark.archive
def test_public_release_excludes_generation_details(archive_release_data, v1_export):
    assets, data = archive_release_data
    internal = assets / "internal"
    lineage = json.loads((internal / "lineage.json").read_bytes())
    assert len(lineage["scenes"]) == 600
    manifest = json.loads((internal / "MANIFEST.json").read_bytes())
    assert manifest["files"]["lineage.json"]["sha256"] == sha(internal / "lineage.json")
    assert not any("lineage" in str(p.relative_to(data)) for p in data.rglob("*"))
    for path in (data / "scenes").glob("*.json"):
        rows = json.loads(path.read_bytes())["scenes"]
        assert {r["scene_id"] for r in rows} == {f"scene-{n:03d}" for n in range(50)}
        for row in rows:
            scene_format.validate_public(row)
            assert "seed" not in row and "seed" not in row["episode"]
            assert "seed" not in row["episode"]["episode_spec"]
            assert "logical_id" not in row["episode"]["episode_spec"]
    original = json.loads((v1_export / "scenes/task1-2.json").read_bytes())["scenes"][0]
    with pytest.raises(ValueError, match="explicit archive root"):
        assets_api.resolve_scene(
            "task1-2", original["seed"], assets, benchmark_version="2026.11.1", data_root=data
        )
    neutral = assets_api.resolve_scene(
        "task1-2", 0, assets, benchmark_version="2026.11.1", data_root=data
    )
    archived = assets_api.resolve_scene(
        "task1-2",
        original["seed"],
        assets,
        benchmark_version="2026.11.1",
        data_root=data,
        v1_root=v1_export,
    )
    assert neutral["seed"] == 0 and archived["seed"] == original["seed"]
    assert archive_compat.relabel(neutral, "scene-000") == archive_compat.relabel(
        archived, "scene-000"
    )


@pytest.mark.archive
def test_v1_reader_preserves_every_original_field(tmp_path, v1_export):
    assets = tmp_path
    original = json.loads((v1_export / "scenes/task1-2.json").read_bytes())["scenes"][0]
    manifest = dict(
        files=[
            dict(path="objects/" + block[key], sha256=block[key])
            for block in original["candidate"]["episode_spec"]["blocks"]
            for key in ("asset_sha256", "texture_sha256")
            if block[key]
        ]
    )
    result = archive_compat.relocate_v1(original, assets, manifest)
    assert result["candidate"]["provenance"] == original["candidate"]["provenance"]
    assert result["initial_state"] == original["initial_state"]
    assert result["candidate_sha256"] == original["candidate_sha256"]
    assert (
        result["candidate"]["episode_spec"]["layout_metadata"]
        == original["candidate"]["episode_spec"]["layout_metadata"]
    )
    assert result["candidate"]["episode_spec"]["blocks"][0]["asset_reference"].startswith(
        str(assets)
    )


@pytest.mark.archive
def test_cross_revision_requires_complete_pinned_physical_equality(archive_release_data, v1_export):
    from scripts.maintainer import gpu_identity, scene_equivalence

    assets, data = archive_release_data
    proof = scene_equivalence.prove(ROOT, assets, data, v1_root=v1_export)
    old = gpu_identity.reference_contract()["contract"]
    new = dict(
        old,
        benchmark="benchmark-2026.11.1",
        data_version="benchmark-2026.11.1",
        scene_equivalence=proof,
    )
    assert scene_equivalence.normalize_contract(new) == old
    assert new["data_version"] == "benchmark-2026.11.1"  # receipts retain actual identity
    for field in proof:
        changed = copy.deepcopy(new)
        changed["scene_equivalence"][field] = "unproven"
        with pytest.raises(ValueError, match="Unproven"):
            scene_equivalence.normalize_contract(changed)
    changed = copy.deepcopy(new)
    changed["physics"]["simulation_hz"] += 1
    with pytest.raises(ValueError, match="simulation_hz"):
        gpu_identity.require_equal(old, scene_equivalence.normalize_contract(changed), "contract")


@pytest.mark.archive
def test_physical_proof_rejects_changed_task7(
    archive_release_data, tmp_path, monkeypatch, v1_export
):
    from scripts.maintainer import scene_equivalence

    assets, data = archive_release_data
    read = data_release.read_payload

    def changed(name, **kwargs):
        body = read(name, **kwargs)
        if name == "scenes/task7-3.json":
            body["scenes"][0]["episode"]["task7_scene"]["cycles"][0]["tool_index"] += 1
        return body

    monkeypatch.setattr(data_release, "read_payload", changed)
    with pytest.raises(ValueError, match="Full runtime input|equivalence mismatch"):
        scene_equivalence.prove(ROOT, assets, data, v1_root=v1_export)


@pytest.mark.archive
def test_public_describe_refuses_retired_revision_with_archive(v1_export):
    with pytest.raises(SystemExit) as error:
        benchmark.main(["describe", "--benchmark", "2026.11.0", "--v1-root", str(v1_export)])
    assert error.value.code == 2


def test_schema_resource_is_versioned():
    schema = json.loads(Path(scene_format.__file__).with_name("scene-v2.schema.json").read_bytes())
    assert schema["properties"]["schema"] == {"type": "string", "const": scene_format.SCHEMA}
    assert set(schema["required"]) == {"schema", "task", "scene_id", "episode", "initial_state"}
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json", "a\\escape.json"])
def test_resource_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        scene_format.safe_path(tmp_path, name)


def test_resource_symlink_cannot_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (root / "escape.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        scene_format.safe_path(root, "escape.json")


def test_smoke_binds_the_selected_revision(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts.maintainer import public_smoke

    public = importlib.import_module("benchmark_candidate")
    selected = []
    monkeypatch.setattr(public, "verify", lambda **kw: selected.append(kw) or {})

    def stop(*args):
        raise ValueError("stop before Kit")

    monkeypatch.setattr(public, "verify_release", stop)
    with pytest.raises(ValueError, match="stop before Kit"):
        public_smoke.worker(
            SimpleNamespace(output=tmp_path, assets=tmp_path, benchmark="2026.11.0", data_root=None)
        )
    assert selected == [
        dict(benchmark_version="2026.11.0", assets=tmp_path, data_root=None, v1_root=None)
    ]


def _metadata_only_checkout(root):
    import shutil

    shutil.copytree(ROOT / "benchmarks/2026.11.1", root / "benchmarks/2026.11.1")
    package = root / "packages/tower-sim/src/tower_sim/bench"
    package.mkdir(parents=True)
    for name in ("public-data.json", "scene-v2.schema.json"):
        shutil.copyfile(Path(data_release.__file__).with_name(name), package / name)
    assert not (root / "simulator").exists()
    return root


@pytest.mark.archive
def test_recheck_and_both_readers_survive_v1_tree_removal(
    archive_release_data, v1_export, tmp_path, monkeypatch
):
    from scripts.maintainer.scene_equivalence import prove

    assets, data = archive_release_data
    root = _metadata_only_checkout(tmp_path / "checkout")
    result = scene_v2_release.stage(root, assets, tmp_path / "rechecked", v1_root=v1_export)
    assert result["passed"] and result["scenes"] == 600
    assert prove(root, assets, data, v1_root=v1_export)["scenes"] == 600
    assert (
        verification.verify_data(root, assets=assets, data_root=data)["verified_payload_files"]
        == 12
    )
    from scripts.maintainer.archive_scene import verify_archive_data

    assert len(verify_archive_data(root, v1_root=v1_export)) == 14
    assert all(
        name.startswith("benchmarks/2026.11.1/") for name in data_release.manifest()["files"]
    )
    monkeypatch.setattr(assets_api.benchmark, "ROOT", root)
    original = json.loads((v1_export / "scenes/task7-3.json").read_bytes())["scenes"][0]
    old = assets_api.resolve_scene(
        "task7-3", original["seed"], assets, benchmark_version="2026.11.0", v1_root=v1_export
    )
    compat = assets_api.resolve_scene(
        "task7-3",
        original["seed"],
        assets,
        benchmark_version="2026.11.1",
        data_root=data,
        compatibility=True,
        v1_root=v1_export,
    )
    assert old == compat
    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="supply --archive-root"):
        assets_api.resolve_scene("task7-3", original["seed"], assets, benchmark_version="2026.11.0")


@pytest.mark.archive
def test_explicit_archive_recheck_without_checked_out_v1_files(
    archive_release_data, v1_export, tmp_path, monkeypatch
):
    from scripts.maintainer.archive_scene import MissingV1Data, V1Source
    from scripts.maintainer.scene_equivalence import prove

    assets, data = archive_release_data
    root = tmp_path / "metadata-checkout"
    root.mkdir()
    _metadata_only_checkout(root)
    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    with pytest.raises(MissingV1Data):
        V1Source(root)
    receipt = prove(root, assets, data, v1_root=v1_export)
    assert receipt["scenes"] == 600


def test_missing_historical_evidence_is_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    from scripts.maintainer.archive_scene import MissingV1Data, V1Source

    with pytest.raises(MissingV1Data, match="supply --archive-root"):
        V1Source(tmp_path)


def test_recheck_skips_only_missing_implicit_evidence(tmp_path, monkeypatch, capsys):
    import sys

    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    monkeypatch.setattr(scene_v2_release, "ROOT", tmp_path)
    argv = ["scene-v2-release", "--assets", str(tmp_path), "--output", str(tmp_path / "output")]
    monkeypatch.setattr(sys, "argv", argv)
    scene_v2_release.main()
    result = json.loads(capsys.readouterr().out)
    assert result["skipped"] and result["passed"] is False
    assert "supply --archive-root" in result["reason"]
    monkeypatch.setattr(sys, "argv", argv + ["--v1-root", str(tmp_path / "missing")])
    with pytest.raises(SystemExit) as error:
        scene_v2_release.main()
    assert error.value.code == 2
    assert "Invalid explicit archive root" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_fresh_public_checkout_refuses_retired_revision():
    with pytest.raises(SystemExit) as error:
        benchmark.main(["describe", "--benchmark", "2026.11.0"])
    assert error.value.code == 2


@pytest.mark.archive
def test_internal_lineage_cannot_enter_public_stage(archive_release_data, tmp_path, v1_export):
    assets, _ = archive_release_data
    with pytest.raises(ValueError, match="outside"):
        scene_v2_release.stage(
            ROOT,
            assets,
            tmp_path / "public",
            v1_root=v1_export,
            internal_output=tmp_path / "public/internal",
        )


def test_dropped_fields_have_no_runtime_readers(tmp_path):
    assert not archive_compat.dropped_key_reads(ROOT)
    planted = tmp_path / "packages/tower-sim/src/tower_sim/planted.py"
    planted.parent.mkdir(parents=True)
    planted.write_text(
        'def read(candidate):\n    return candidate["provenance"]["reference_commands_loaded"]\n'
    )
    assert archive_compat.dropped_key_reads(tmp_path)
    contract = archive_compat.field_contract()
    assert not any("measured_asset_id" in path for path in contract["dropped"])


def test_public_reader_is_independent_of_archive(tmp_path):
    import subprocess
    import sys

    # A process-level import blocker covers transitive imports as well as names.
    code = """
import importlib.abc, sys
class NoArchive(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if "archive_compat" in fullname:
            raise RuntimeError("public reader imported archival conversion")
sys.meta_path.insert(0, NoArchive())
from tower_sim.bench import scene_format, data_release
scene_format.scene_schema()
assert not any("archive_compat" in name for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "value", ["worker.ecs." + "internal", "worker." + "internal", ".".join(("10", "8", "7", "6"))]
)
def test_private_string_validation(value):
    with pytest.raises(ValueError, match="Private"):
        scene_format.validate_public({"label": value})


@pytest.mark.archive
def test_canonical_resource_and_unknown_field_rejection(v1_export):
    original = json.loads((v1_export / "scenes/task1-2.json").read_bytes())["scenes"][0]
    body = archive_compat.public_scene(original, "scene-000")
    body["episode"]["episode_spec"]["blocks"][0]["asset_reference"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="Noncanonical"):
        scene_format.decode_scene(body)
    body = archive_compat.public_scene(original, "scene-000")
    body["episode"]["episode_spec"]["layout_metadata"]["generator_seed"] = 1
    with pytest.raises(ValueError, match="unrecognized"):
        scene_format.decode_scene(body)


@pytest.mark.archive
def test_archive_selector_bounds_and_hashes(v1_export, monkeypatch):
    for selector in (-1, 50, True, "scene-050", "scene-999"):
        with pytest.raises(ValueError):
            archive_compat.public_scene_id(ROOT, "task1-2", selector)
    original = json.loads((v1_export / "scenes/task1-2.json").read_bytes())["scenes"][0]
    assert (
        archive_compat.public_scene_id(ROOT, "task1-2", original["seed"], v1_root=v1_export)
        == "scene-000"
    )
    assert (
        archive_compat.archived_scene(ROOT, "task1-2", "scene-000", v1_root=v1_export) == original
    )
    monkeypatch.setattr(
        archive_compat,
        "equivalence_map",
        lambda root: [dict(task="task1-2", scene_id="scene-000", old_sha256="0" * 64)],
    )
    with pytest.raises(ValueError, match="unique archive-to-public"):
        archive_compat.public_scene_id(ROOT, "task1-2", original["seed"], v1_root=v1_export)
    with pytest.raises(ValueError, match="unique archived scene"):
        archive_compat.archived_scene(ROOT, "task1-2", "scene-000", v1_root=v1_export)


@pytest.mark.public_assets
def test_locator_independent_identity(release_data, tmp_path, monkeypatch):
    assets, data = release_data
    root = _metadata_only_checkout(tmp_path / "checkout")
    monkeypatch.setattr(
        verification,
        "data_manifest_path",
        lambda benchmark="2026.11.1": (
            root / "packages/tower-sim/src/tower_sim/bench/public-data.json"
        ),
    )
    before = verification.verify_data(root, assets=assets, data_root=data)
    lock_path = root / "benchmarks/2026.11.1/assets.lock.json"
    lock = json.loads(lock_path.read_bytes())
    lock.update(
        repository="owner/evaluation", revision="a" * 40, tag="owner-release", directory="data"
    )
    lock_path.write_text(json.dumps(lock))
    after = verification.verify_data(root, assets=assets, data_root=data)
    assert before["data_manifest_sha256"] == after["data_manifest_sha256"]
    assert before == after
    lock["manifest_sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="manifest_sha256"):
        verification.verify_data(root, assets=assets, data_root=data)


@pytest.mark.public_assets
@pytest.mark.parametrize("directory", [".", "", "/", "/./", "data", "/nested/data/"])
@pytest.mark.parametrize("transient", [False, True])
def test_data_download_is_pinned_and_checks_every_byte(
    release_data, tmp_path, monkeypatch, directory, transient
):
    import io
    from urllib.error import URLError
    from urllib.parse import unquote

    from tower_sim.bench import download

    monkeypatch.setattr(download.time, "sleep", lambda delay: None)
    _, data = release_data
    root = _metadata_only_checkout(tmp_path / "checkout")
    lock_path = root / "benchmarks/2026.11.1/assets.lock.json"
    lock = json.loads(lock_path.read_bytes())
    lock["revision"] = "TODO(owner): unpublished test fixture"
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(data_release, "urlopen", lambda *a, **kw: pytest.fail("unapproved network"))
    with pytest.raises(ValueError, match="data release not yet pinned; pass --data-root <dir>"):
        data_release.download(root, tmp_path / "unpublished")
    assert data_release.download(root, tmp_path / "local", data_root=data)
    lock_path = root / "benchmarks/2026.11.1/assets.lock.json"
    lock = json.loads(lock_path.read_bytes())
    lock.update(repository="owner/evaluation", revision="a" * 40, directory=directory)
    lock_path.write_text(json.dumps(lock))
    fetched = []

    def fetch(url, **kwargs):
        assert "/./" not in url
        name = unquote(url.split("/" + "a" * 40 + "/")[1])
        prefix = directory.strip("/")
        if prefix not in ("", "."):
            assert name.startswith(prefix + "/")
            name = name.removeprefix(prefix + "/")
        fetched.append(name)
        if transient and len(fetched) == 1:
            raise URLError("temporary")
        return io.BytesIO((data / name).read_bytes())

    monkeypatch.setattr(data_release, "urlopen", fetch)
    downloaded = tmp_path / "download"
    data_release.download(root, downloaded)
    assert set(fetched) == {p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file()}
    assert "assets.lock.json" not in fetched
    monkeypatch.setattr(data_release, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="checksum/size"):
        data_release.download(root, tmp_path / "corrupt")


@pytest.mark.public_assets
def test_lineage_rejected_on_all_public_surfaces(release_data, tmp_path):
    import shutil

    from tower_sim.bench.public_surface import check_lineage

    _, data = release_data
    for schema in ("tower-scene-" + "lineage-v1", "tower-internal-" + "lineage-manifest-v1"):
        payload = json.dumps(dict(schema=schema)).encode()
        for name in ("public.json", "tower_sim/hidden.json"):
            with pytest.raises(ValueError, match="Internal lineage"):
                check_lineage(name, payload)
        check_lineage("scripts/maintainer/scene_v2_release.py", payload)
    root = _metadata_only_checkout(tmp_path / "checkout")
    (root / "benchmarks/2026.11.1/extra.json").write_text("{}")
    assert scan_scene_v2(root)
    staged = tmp_path / "stage"
    shutil.copytree(data, staged)
    (staged / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="Unexpected"):
        data_release.verify_release(staged)
    (staged / "extra.json").unlink()
    (staged / "README.md").write_bytes(payload)
    with pytest.raises(ValueError, match="Internal lineage"):
        data_release.verify_release(staged)


@pytest.mark.public_assets
def test_v2_runtime_plans_and_clone_authoring(release_data, monkeypatch):
    """Exercise actual plans, clone wrapper and block authoring without Isaac.

    Storage hashing and SDK primitives are doubles; field access and construction
    code are real. All-scene equivalence separately proves complete runtime DTOs.
    """
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock

    from tower_sim.bench import scene, tasks

    assets, data = release_data
    pxr = ModuleType("pxr")
    for name in ("Gf", "Sdf", "Usd", "UsdGeom", "UsdPhysics", "UsdShade"):
        setattr(pxr, name, MagicMock())
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    hashes = {
        str((assets / row["path"]).resolve()): row["sha256"]
        for row in assets_api.manifest(assets)["files"]
    }
    monkeypatch.setattr(scene, "_file_hash", lambda path: hashes[str(Path(path).resolve())])
    for path in sorted((data / "scenes").glob("*.json")):
        row = assets_api.resolve_scene(path.stem, 0, assets, data_root=data)
        plans = scene.plans_for_candidates([row["candidate"]], tasks.validate_spec)
        assert len(plans) == 1 and plans[0]["blocks"]
        stage = MagicMock()
        prims = {item["root"]: MagicMock() for item in plans[0]["blocks"]}
        stage.GetPrimAtPath.side_effect = prims.__getitem__
        pxr.Usd.PrimRange.side_effect = lambda root, *args: [root]

        def author(stage, selected, bond, prims=prims):
            for plan in selected:
                for block in plan["blocks"]:
                    scene._author_block(stage, block, 1000.0)
                    expected = block["block"]["measured_asset_id"]
                    prims[block["root"]].CreateAttribute.return_value.Set.assert_called_once_with(
                        -1 if expected is None else expected
                    )
            return dict(passed=True)

        class Cloner:
            def __init__(self, stage):
                pass

            def clone(self, *args, **kwargs):
                return "cloned"

        hook = scene.CloneHook(plans, "unused", author=author)
        assert (
            hook.factory(Cloner, stage).clone("/World/envs/env_0", [p["env_path"] for p in plans])
            == "cloned"
        )
        hook.verify()
        assert hook.report["usd_readback"]["passed"]
        unverified = scene_format.decode_scene(
            data_release.scene(path.stem, "scene-000", assets=assets, data_root=data)
        )
        with pytest.raises(ValueError, match="Unverified candidate"):
            scene.plans_for_candidates([unverified["candidate"]], tasks.validate_spec)


def test_planted_leak_in_bound_metadata_is_rejected(tmp_path):
    root = _metadata_only_checkout(tmp_path / "checkout")
    target = root / "benchmarks/2026.11.1/validation.json"
    target.write_text(json.dumps(dict(path="/" + "mnt" + "/n" + "as/planted")))
    assert scan_scene_v2(root)


@pytest.mark.public_assets
def test_upload_manifest_byte_pin(release_data, tmp_path):
    import shutil

    _, data = release_data
    root = tmp_path / "stage"
    shutil.copytree(data, root)
    path = root / "MANIFEST.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Changed data release MANIFEST"):
        data_release.verify_release(root)


def test_wheel_rejects_private_lineage():
    import io
    import zipfile

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as wheel:
        wheel.writestr(
            "tower_sim/hidden.json", json.dumps(dict(schema="tower-scene-" + "lineage-v1"))
        )
    with pytest.raises(ValueError, match="Internal lineage"):
        verification.wheel_payload(payload.getvalue(), "tower-sim", "0.1.0")


@pytest.mark.archive
def test_physical_map_hash_must_match_payload(
    archive_release_data, tmp_path, monkeypatch, v1_export
):
    from scripts.maintainer import scene_equivalence

    assets, data = archive_release_data
    root = _metadata_only_checkout(tmp_path / "checkout")
    path = root / "benchmarks/2026.11.1/scene-equivalence.json"
    body = json.loads(path.read_bytes())
    body["scenes"][0]["physical_sha256"] = "0" * 64
    path.write_text(json.dumps(body))
    monkeypatch.setattr(scene_equivalence, "expected_receipt", lambda root: {})
    with pytest.raises(ValueError, match="Scene equivalence mismatch"):
        scene_equivalence.prove(root, assets, data, v1_root=v1_export)


def test_metadata_file_set_is_exact(tmp_path):
    root = _metadata_only_checkout(tmp_path / "checkout")
    assert not scan_scene_v2(root)
    (root / "benchmarks/2026.11.1/extra.json").write_text("{}")
    assert scan_scene_v2(root)


@pytest.mark.public_assets
def test_download_retry_removes_stale_temporary_files(release_data, tmp_path, monkeypatch):
    import shutil

    _, data = release_data
    root = _metadata_only_checkout(tmp_path / "checkout")
    lock_path = root / "benchmarks/2026.11.1/assets.lock.json"
    lock = json.loads(lock_path.read_bytes())
    lock.update(repository="owner/evaluation", revision="a" * 40)
    lock_path.write_text(json.dumps(lock))
    assets = tmp_path / "assets"
    destination = data_release.payload_root(assets)
    shutil.copytree(data, destination)
    stale = [destination / ".download-killed", destination / "scenes/.download-interrupted"]
    for path in stale:
        path.write_bytes(b"partial download")
    monkeypatch.setattr(data_release, "urlopen", lambda *a, **kw: pytest.fail("already cached"))
    assert len(data_release.download(root, assets)) == 20
    assert not any(path.exists() for path in stale)
    # Cleanup must not hide unrelated extras from the exact file-set check.
    (destination / "unrelated.txt").write_text("preserve")
    with pytest.raises(ValueError, match="Unexpected data release files"):
        data_release.download(root, assets)
    assert (destination / "unrelated.txt").read_text() == "preserve"


def test_readme_post_upload_commands_validate_pin_and_download(tmp_path, monkeypatch):
    import io
    import shlex
    import tempfile

    data = tmp_path / "fixture-release"
    data.mkdir()
    content = b"local download fixture"
    (data / "fixture.txt").write_bytes(content)
    entries = {"fixture.txt": dict(sha256=sha(data / "fixture.txt"), size=len(content))}
    (data / "MANIFEST.json").write_text(
        json.dumps(dict(schema="tower-staged-release-v1", files=entries))
    )
    document = dict(manifest_sha256=sha(data / "MANIFEST.json"), upload_files=entries)
    monkeypatch.setattr(data_release, "manifest", lambda: document)
    root = _metadata_only_checkout(tmp_path / "checkout")
    lock_path = root / "benchmarks/2026.11.1/assets.lock.json"
    lock = json.loads(lock_path.read_bytes())
    lock["manifest_sha256"] = document["manifest_sha256"]
    lock["revision"] = "TODO(owner): unpublished test fixture"
    lock_path.write_text(json.dumps(lock))
    readme = scene_v2_release.README
    for phrase in (
        "$PLAN",
        "check_p706_mutations",
        "test_release_placeholders",
        "private lineage",
        "SOURCE_MANIFEST",
        "internal 2026.11.0 predecessor",
    ):
        assert phrase not in readme
    commands = [
        shlex.split(line)[2] for line in readme.splitlines() if line.startswith("python -c ")
    ]
    assert len(commands) == 2
    monkeypatch.chdir(root)
    monkeypatch.setattr(data_release, "urlopen", lambda *a, **kw: pytest.fail("unapproved network"))
    with pytest.raises(ValueError, match="data release not yet pinned"):
        exec(commands[0], {})
    lock_path = root / "benchmarks/2026.11.1/assets.lock.json"
    lock = json.loads(lock_path.read_bytes())
    lock.update(repository="owner/evaluation", revision="a" * 40)
    lock_path.write_text(json.dumps(lock))
    exec(commands[0], {})
    fetched, verified, temporary = [], [], []
    real_verify, real_temporary = data_release.verify_release, tempfile.TemporaryDirectory

    def fetch(url, **kwargs):
        name = url.split("/" + "a" * 40 + "/")[1]
        prefix = lock["directory"].strip("/")
        if prefix not in ("", "."):
            name = name.removeprefix(prefix + "/")
        fetched.append(name)
        return io.BytesIO((data / name).read_bytes())

    def verify(path, **kwargs):
        verified.append(Path(path))
        return real_verify(path, **kwargs)

    def scratch(*args, **kwargs):
        result = real_temporary(*args, **kwargs)
        temporary.append(Path(result.name))
        return result

    monkeypatch.setattr(data_release, "urlopen", fetch)
    monkeypatch.setattr(data_release, "verify_release", verify)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", scratch)
    exec(commands[1], {})
    assert len(fetched) == 2
    assert len(verified) == 2  # downloader verification plus the documented explicit check
    assert temporary and all(not path.exists() for path in temporary)
    monkeypatch.setattr(data_release, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="checksum/size"):
        exec(commands[1], {})


@pytest.mark.public_assets
def test_assets_verify_cli_checks_explicit_data_root(release_data, tmp_path, monkeypatch, capsys):
    import shutil
    import sys

    assets, data = release_data
    staged = tmp_path / "stage"
    shutil.copytree(data, staged)
    monkeypatch.setattr(assets_api, "verify", lambda root: {"files": 7})
    monkeypatch.setattr(
        sys, "argv", ["assets", "verify", "--assets", str(assets), "--data-root", str(staged)]
    )
    assets_api.main()
    assert json.loads(capsys.readouterr().out) == {"files": 7, "data_files": 20}
    (staged / "README.md").write_text("changed")
    with pytest.raises(ValueError, match="Changed data release file"):
        assets_api.main()


@pytest.mark.archive
def test_runtime_labels_reject_undeclared_changes(v1_export, archive_release_data, monkeypatch):
    assets, _ = archive_release_data
    original = json.loads((v1_export / "scenes/task7-2.json").read_bytes())["scenes"][0]
    body = archive_compat.public_scene(original, "scene-000")
    manifest = assets_api.manifest(assets)
    before = archive_compat.remove_declared_fields(
        archive_compat.relocate_v1(original, assets, manifest)
    )
    expected = archive_compat.expected_runtime(original, body, assets, manifest)
    archive_compat.assert_runtime_labels(before, expected)
    contract = archive_compat.field_contract()
    # Removing a declared label must fail, proving the list actually binds the check.
    contract["runtime_labels"].remove("/seed")
    with monkeypatch.context() as patch:
        patch.setattr(archive_compat, "field_contract", lambda: contract)
        with pytest.raises(ValueError, match="Undeclared runtime label"):
            archive_compat.assert_runtime_labels(before, expected)
    altered = copy.deepcopy(expected)
    altered["candidate"]["episode_spec"]["blocks"][0]["pose"][0] += 0.1
    assert archive_compat.changed_pointers(expected, altered) == {
        "/candidate/episode_spec/blocks/0/pose/0"
    }
    with pytest.raises(ValueError, match="Undeclared runtime label"):
        archive_compat.assert_runtime_labels(before, altered)
    # Even a coordinated edit to both the relabel adapter and regenerated v2 data
    # cannot disguise a physical edit as a mechanical label change.
    real_relabel = archive_compat.relabel

    def drift(row, scene_id):
        value = real_relabel(row, scene_id)
        value["candidate"]["episode_spec"]["blocks"][0]["pose"][0] += 0.1
        return value

    monkeypatch.setattr(archive_compat, "relabel", drift)
    changed = archive_compat.public_scene(original, "scene-000")
    with pytest.raises(ValueError, match="Undeclared runtime label"):
        archive_compat.assert_runtime_equal(original, changed, assets, manifest)


@pytest.mark.archive
@pytest.mark.parametrize("manifest", [{}, {"files": []}])
def test_empty_asset_manifest_cannot_authorize_scene(v1_export, manifest):
    original = json.loads((v1_export / "scenes/task1-2.json").read_bytes())["scenes"][0]
    body = archive_compat.public_scene(original, "scene-000")
    with pytest.raises(ValueError, match="Missing asset content hash"):
        scene_format.decode_scene(body, manifest)


@pytest.mark.archive
def test_neutral_task7_frame_label_and_archive_cleanup(v1_export):
    assert not hasattr(archive_compat, "GENERATION_FIELDS")
    for task in ("task7-2", "task7-3"):
        original = json.loads((v1_export / f"scenes/{task}.json").read_bytes())["scenes"][0]
        body = archive_compat.public_scene(original, "scene-000")
        assert body["initial_state"]["coordinate_frame"] == "encoder_with_single_J2_conversion"
        assert (
            "/initial_state/coordinate_frame" in archive_compat.field_contract()["runtime_labels"]
        )


def test_missing_public_scene_names_task_scene(monkeypatch):
    monkeypatch.setattr(
        data_release,
        "read_payload",
        lambda *a, **kw: dict(schema="tower-scenes-v2", task="task1-2", scenes=[]),
    )
    with pytest.raises(ValueError, match="exactly one scene task/scene"):
        data_release.scene("task1-2", "scene-000", assets="unused")


def test_remove_declared_fields_preserves_every_other_value():
    row = {
        "candidate": {
            "episode_spec": {
                "layout_metadata": {"runtime_sampling_manifest": "neutral provenance", "keep": 3},
                "blocks": [{"measured_asset_id": "retained measured identity", "mass": 2.0}],
            },
            "provenance": {"physical_fingerprint": "neutral provenance", "keep": [1, 2]},
        },
        "initial_state": {"qpos": [0.0, 1.0]},
    }
    before = copy.deepcopy(row)
    expected = copy.deepcopy(row)
    del expected["candidate"]["episode_spec"]["layout_metadata"]["runtime_sampling_manifest"]
    del expected["candidate"]["provenance"]["physical_fingerprint"]
    assert archive_compat.remove_declared_fields(row) == expected
    assert row == before
