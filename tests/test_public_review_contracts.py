"""Reviewed archive tools and station metadata identity keep their fail-closed contracts."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tower_sim.bench.asset_identity import (
    CANONICAL_STATION_SHA256,
    PUBLIC_STATION_SHA256,
    station_identity,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("failure", [None, "hash", "size"])
def test_archive_asset_cli_verifies_hash_and_size(tmp_path, monkeypatch, failure):
    from scripts.maintainer import archive_assets

    payload = b"archived bytes"
    row = dict(path="fixture.bin", sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))
    if failure == "hash":
        row["sha256"] = "0" * 64
    if failure == "size":
        row["size"] += 1
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "fixture.bin").write_bytes(payload)
    manifest = sources / "manifest.json"
    manifest.write_text(json.dumps(dict(files=[row])))
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            dict(
                repository="tower-benchmark/TOWER-Assets-archive",
                revision="b" * 40,
                directory="benchmarks/202611-v1",
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
        )
    )
    monkeypatch.setattr(archive_assets, "LOCK", lock)
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        assert kwargs["token"] is True and kwargs["repo_id"].endswith("-archive")
        assert kwargs["revision"] == "b" * 40
        return sources / Path(kwargs["filename"]).name

    monkeypatch.setattr(
        archive_assets.importlib,
        "import_module",
        lambda name: SimpleNamespace(hf_hub_download=fetch),
    )
    destination = tmp_path / "output"
    if failure:
        with pytest.raises(ValueError, match="Changed archived asset"):
            archive_assets.main(["--destination", str(destination)])
        assert not (destination / "fixture.bin").exists()
    else:
        assert archive_assets.main(["--destination", str(destination)]) == 0
        assert (destination / "fixture.bin").read_bytes() == payload
    assert len(calls) == 2


@pytest.mark.parametrize(
    "digest,profile_digest,accepted",
    [
        (PUBLIC_STATION_SHA256, CANONICAL_STATION_SHA256, True),
        (CANONICAL_STATION_SHA256, CANONICAL_STATION_SHA256, True),
        ("0" * 64, CANONICAL_STATION_SHA256, False),
        (PUBLIC_STATION_SHA256, PUBLIC_STATION_SHA256, False),
    ],
)
def test_replay_profile_accepts_public_station_without_override(digest, profile_digest, accepted):
    # Execute the actual worker's identity and profile admission statements without Kit.
    path = ROOT / "packages/tower-sim/src/tower_sim/replay/dual_replay_worker.py"
    tree = ast.parse(path.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    names = {
        "stage_identity_before",
        "provenance_hash",
        "profile_source_stage_verified",
        "profile_stage_mismatch_acknowledged",
    }
    nodes = [
        n
        for n in ast.walk(run)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
    ]
    gate = next(
        n
        for n in ast.walk(run)
        if isinstance(n, ast.If)
        and ast.unparse(n.test)
        == "not profile_source_stage_verified and (not profile_stage_mismatch_acknowledged)"
    )
    nodes.append(gate)
    nodes.sort(key=lambda n: n.lineno)
    namespace = dict(
        station_identity=station_identity,
        stage_hash_before=digest,
        config={},
        profile={"provenance": {"source_stage_sha256": profile_digest}},
    )
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")
    if accepted:
        exec(code, namespace)
        assert namespace["profile_source_stage_verified"] is True
        assert namespace["profile_stage_mismatch_acknowledged"] is False
    else:
        with pytest.raises(ValueError, match="provenance does not match"):
            exec(code, namespace)
    # The alignment parameter verifier consumes the same canonical identity.
    argument = next(
        n for n in ast.walk(run) if isinstance(n, ast.keyword) and n.arg == "runtime_stage_sha256"
    )
    assert eval(
        compile(ast.Expression(argument.value), str(path), "eval"),
        dict(namespace, reference_report={}),
    ) == station_identity(digest)
    assert namespace["stage_hash_before"] == digest  # Receipts retain actual public bytes.


def test_legacy_asset_manifest_names_private_archive():
    from tower_eval.cli import ASSET_REPOSITORIES

    manifest = json.loads((ROOT / "hf-assets/manifest.json").read_text())
    assert manifest["archived"] is True and manifest["revision"] == "0.0.2"
    assert manifest["repository"] == "tower-benchmark/TOWER-Assets-archive"
    assert manifest["repository"] in ASSET_REPOSITORIES


def test_exact_root_does_not_inherit_public_golden_environment(monkeypatch):
    import argparse

    from scripts.maintainer import check_golden_exact

    class Parsed(Exception):
        pass

    defaults = []

    def parse(parser):
        defaults.append(parser.get_default("root"))
        raise Parsed

    monkeypatch.setenv("TOWER_GOLDEN_ROOT", "public-golden")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    with pytest.raises(Parsed):
        check_golden_exact.main()
    assert defaults == [check_golden_exact.GOLDEN]


@pytest.mark.parametrize("kind", ["checkout", "nested", "symlink"])
def test_archive_cli_refuses_checkout_destinations(tmp_path, monkeypatch, capsys, kind):
    from scripts.maintainer import archive_assets

    destination = ROOT if kind == "checkout" else ROOT / "archived-assets"
    if kind == "symlink":
        destination = tmp_path / "checkout-link"
        destination.symlink_to(ROOT, target_is_directory=True)
        destination /= "archived-assets"
    downloads = []
    monkeypatch.setattr(archive_assets, "download", downloads.append)
    with pytest.raises(SystemExit) as error:
        archive_assets.main(["--destination", str(destination)])
    assert error.value.code == 2
    assert "outside the checkout" in capsys.readouterr().err
    assert not downloads
