"""Public data fetches fail closed on tampering and distinguish unavailable inputs."""

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest
from tower_sim.bench import asset_identity

from scripts.maintainer import fetch_golden as golden


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    payload = b"pinned public recording"
    name = "task1-2-mme.tar.xz"
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {"files": {name: {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}}}
        )
    )
    lock = tmp_path / "assets.lock.json"
    lock.write_text(
        json.dumps({"repository": "tower-benchmark/TOWER-Assets", "revision": "a" * 40})
    )
    monkeypatch.setattr(golden, "MANIFEST", manifest)
    monkeypatch.setattr(golden, "LOCK", lock)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for key in ("TOWER_GOLDEN_ROOT", "HF_HUB_OFFLINE", "TOWER_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    return name, payload


@pytest.mark.parametrize("nested", [False, True])
def test_explicit_golden_root(corpus, tmp_path, monkeypatch, nested):
    name, payload = corpus
    directory = tmp_path / "assets"
    target = directory / golden.DIRECTORY / name if nested else directory / name
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    monkeypatch.setattr(
        golden, "urlopen", lambda *a, **k: pytest.fail("local data fetched from network")
    )
    assert golden.golden_archive(name, directory) == target
    monkeypatch.setenv("TOWER_GOLDEN_ROOT", str(directory))
    assert golden.golden_archive(name) == target
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed public golden"):
        golden.golden_archive(name)
    target.unlink()
    with pytest.raises(ValueError, match="Incomplete explicit"):
        golden.golden_archive(name)


@pytest.mark.parametrize("transient", [False, True])
def test_download_pins_commit_hash_size_and_cache(corpus, monkeypatch, transient):
    from tower_sim.bench import download

    monkeypatch.setattr(download.time, "sleep", lambda delay: None)
    name, payload = corpus
    urls = []

    def fetch(url, **kwargs):
        urls.append(url)
        if transient and len(urls) == 1:
            raise URLError("temporary")
        return io.BytesIO(payload)

    monkeypatch.setattr(golden, "urlopen", fetch)
    path = golden.golden_archive(name)
    assert urls[-1:] == [
        "https://huggingface.co/datasets/tower-benchmark/TOWER-Assets/resolve/"
        + "a" * 40
        + "/"
        + golden.DIRECTORY
        + "/"
        + name
    ]
    assert path.parent.name == hashlib.sha256(payload).hexdigest()
    assert golden.golden_archive(name) == path and len(urls) == 1 + transient
    path.write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="Changed public golden"):
        golden.golden_archive(name)


@pytest.mark.parametrize("mode", ["offline", "network"])
def test_unavailable_is_an_explicit_skip_signal(corpus, monkeypatch, mode):
    from tower_sim.bench import download

    monkeypatch.setattr(download.time, "sleep", lambda delay: None)
    name, _ = corpus
    if mode == "offline":
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def fetch(*args, **kwargs):
        if mode != "network":
            pytest.fail("offline or unpublished input attempted network")
        raise URLError("network unavailable")

    monkeypatch.setattr(golden, "urlopen", fetch)
    with pytest.raises(golden.GoldenUnavailable, match="SKIP: public golden data unavailable"):
        golden.golden_archive(name)


@pytest.mark.parametrize("payload", [b"short", b"x" * len(b"pinned public recording")])
def test_download_mismatch_is_failure_and_leaves_no_cache(corpus, monkeypatch, payload):
    name, _ = corpus
    monkeypatch.setattr(golden, "urlopen", lambda *a, **k: io.BytesIO(payload))
    with pytest.raises(ValueError, match="checksum/size"):
        golden.golden_archive(name)
    cache = Path(__import__("os").environ["XDG_CACHE_HOME"])
    assert not [p for p in cache.rglob("*") if p.is_file()]


@pytest.mark.parametrize("name", ["../task1-2-mme.tar.xz", "unknown.tar.xz", "MANIFEST.json"])
def test_only_manifest_recordings_are_fetchable(corpus, name):
    with pytest.raises(ValueError, match="manifest-listed"):
        golden.golden_archive(name)


def test_station_identity_maps_only_pinned_public_bytes():
    assert (
        asset_identity.station_identity(asset_identity.PUBLIC_STATION_SHA256)
        == asset_identity.CANONICAL_STATION_SHA256
    )
    assert (
        asset_identity.station_identity(asset_identity.CANONICAL_STATION_SHA256)
        == asset_identity.CANONICAL_STATION_SHA256
    )
    for digest in ("", "0" * 64, "f" * 64, asset_identity.PUBLIC_STATION_SHA256[:-1] + "0"):
        assert asset_identity.station_identity(digest) == digest


def test_asset_manifest_identity_maps_only_sealed_distribution():
    assert (
        asset_identity.asset_manifest_identity(asset_identity.PUBLIC_ASSET_MANIFEST_SHA256)
        == asset_identity.CANONICAL_ASSET_MANIFEST_SHA256
    )
    assert asset_identity.asset_manifest_identity("0" * 64) == "0" * 64


@pytest.mark.public_assets
def test_public_station_configuration_is_exactly_the_packaged_cache():
    from tower_sim.backend.resources import DATA
    from tower_sim.bench.drives import CalibratedDrives
    from tower_sim.bench.j2comp import J2Compensation

    from scripts.maintainer.fetch_public_test_data import public_asset_root

    root = public_asset_root()
    station = root / "station/station-dual-j2-wrist-shared-v3.usdz"
    cache = DATA / "benchmark/runtime/ik-capsule/runtime/scene-cache.json"
    configuration = json.loads(cache.read_bytes())["configuration"]
    assert hashlib.sha256(station.read_bytes()).hexdigest() == asset_identity.PUBLIC_STATION_SHA256
    compensation = J2Compensation.from_cache(cache, station)
    assert configuration == compensation.CONFIG
    stage = CalibratedDrives(station, configuration).stage()
    actual = json.loads(stage.GetPrimAtPath("/RSSUnified").GetAttribute("rss:configuration").Get())
    assert actual == configuration


def test_public_assets_offline_skip_and_explicit_failure(tmp_path, monkeypatch):
    from scripts.maintainer import fetch_public_test_data as assets

    monkeypatch.delenv("TOWER_ASSET_ROOT", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(assets.PublicAssetsUnavailable, match="SKIP: public assets unavailable"):
        assets.public_asset_root()
    with pytest.raises(ValueError, match="Incomplete explicit"):
        assets.public_asset_root(tmp_path / "missing")


def test_public_asset_manifest_tamper_fails_before_scene_read(tmp_path, monkeypatch):
    from tower_sim.bench import data_release

    from scripts.maintainer import fetch_public_test_data as assets

    monkeypatch.setattr(
        data_release, "verify_release", lambda *a: pytest.fail("unbound assets read")
    )
    (tmp_path / "manifest.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed public asset manifest"):
        assets.public_asset_root(tmp_path)


def test_recording_size_is_checked_independently_of_hash(corpus, tmp_path):
    name, payload = corpus
    (tmp_path / name).write_bytes(payload)
    manifest = json.loads(golden.MANIFEST.read_bytes())
    manifest["files"][name]["size"] += 1
    golden.MANIFEST.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Changed public golden"):
        golden.golden_archive(name, tmp_path)


@pytest.mark.parametrize(
    "marker,module,error",
    [
        ("golden", "fetch_golden", "GoldenUnavailable"),
        ("public_assets", "fetch_public_test_data", "PublicAssetsUnavailable"),
    ],
)
@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize("required", [False, True])
def test_unavailable_public_data_is_only_a_marked_skip(
    tmp_path, marker, module, error, marked, required
):
    (tmp_path / "conftest.py").write_text((golden.ROOT / "conftest.py").read_text())
    (tmp_path / "pytest.ini").write_text(f"[pytest]\nmarkers = {marker}: public input\n")
    (tmp_path / "test_probe.py").write_text(
        f"import pytest\nfrom scripts.maintainer.{module} import {error}\n"
        + (f"@pytest.mark.{marker}\n" if marked else "")
        + f"def test_probe():\n    raise {error}('SKIP: public data unavailable (offline)')\n"
    )
    env = dict(
        os.environ, PYTHONPATH=str(golden.ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    env.pop("TOWER_ARCHIVE_ROOT", None)
    env["TOWER_REQUIRE_PUBLIC_DATA"] = "1" if required else "0"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
    )
    skipped = marked and not required
    assert result.returncode == (0 if skipped else 1), result.stdout + result.stderr
    assert ("1 skipped" if skipped else "1 failed") in result.stdout
    assert "1 passed" not in result.stdout
    assert f"Public data skips: {int(skipped)}" in result.stdout


@pytest.mark.parametrize("corrupt_peer", [False, True])
def test_concurrent_asset_download_verifies_the_published_winner(
    tmp_path, monkeypatch, corrupt_peer
):
    from tower_sim.bench import data_release

    from scripts.maintainer import fetch_public_test_data as assets

    payload = b'{"files": []}'
    digest = hashlib.sha256(payload).hexdigest()
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"manifest_sha256": digest, "revision": "a" * 40}))
    monkeypatch.setattr(assets, "LOCK", lock)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for variable in ("TOWER_ASSET_ROOT", "TOWER_OFFLINE", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(variable, raising=False)
    winner = tmp_path / "cache/tower/assets" / digest

    def download(destination):
        destination.mkdir()
        (destination / "manifest.json").write_bytes(payload)
        winner.mkdir()
        (winner / "manifest.json").write_bytes(b"changed" if corrupt_peer else payload)

    monkeypatch.setattr(
        assets.importlib, "import_module", lambda name: SimpleNamespace(download=download)
    )
    verified = []

    def verify_release(path):
        verified.append(path)
        return {}

    monkeypatch.setattr(data_release, "verify_release", verify_release)
    if corrupt_peer:
        with pytest.raises(ValueError, match="Changed public asset manifest"):
            assets.public_asset_root()
        assert verified == [winner.with_name(".partial-" + digest) / "benchmark-2026.11.1"]
    else:
        assert assets.public_asset_root() == winner
        assert verified == [
            winner.with_name(".partial-" + digest) / "benchmark-2026.11.1",
            winner / "benchmark-2026.11.1",
        ]
