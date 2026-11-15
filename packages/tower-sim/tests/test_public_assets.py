"""Public assets and request preflight cannot depend on private archive readers."""

import importlib
import io
import json
from pathlib import Path
from urllib.parse import unquote

import pytest
from tower_sim.backend.resources import DATA
from tower_sim.bench import data_release
from tower_sim.bench.resources import sha

from scripts.maintainer import archive_root
from scripts.maintainer.golden_support import benchmark

assets = importlib.import_module("benchmark_assets")
public = importlib.import_module("benchmark_candidate")


@pytest.fixture
def public_assets(tmp_path, monkeypatch):
    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Public route read the private archive")

    monkeypatch.setattr(archive_root, "archive_bundle", forbidden)
    assert assets.LOCK == DATA / "benchmark/assets.lock.json"
    assert (
        assets.SCENE_INDEX_SHA256
        == "0bad41c69640cd51b702fde8527e3b354c7c619743f18d66b080f929a7d46cc3"
    )
    root = tmp_path / "assets"
    station = root / "station/station-dual-j2-wrist-shared-v3.usdz"
    station.parent.mkdir(parents=True)
    station.write_bytes(b"synthetic station")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema="tower-benchmark-assets-v1",
                release="fixture",
                scene_index_sha256=assets.SCENE_INDEX_SHA256,
                files=[
                    dict(
                        path=station.relative_to(root).as_posix(),
                        size=station.stat().st_size,
                        sha256=sha(station),
                    )
                ],
            )
        )
    )
    # Substitute only synthetic content hashes, preserving the actual package reader paths.
    read = benchmark.read
    accessed = []

    def fixture_read(path):
        value = read(path)
        accessed.append(Path(path))
        if Path(path) == DATA / "benchmark/assets.lock.json":
            value["manifest_sha256"] = sha(manifest)
            value["revision"] = "a" * 40  # Synthetic immutable commit for the fake HTTPS server.
        if Path(path) == DATA / "benchmark/runtime/ik-capsule/runtime/scene-cache.json":
            value["scene_sha256"] = sha(station)
        return value

    monkeypatch.setattr(benchmark, "read", fixture_read)
    data = tmp_path / "public-data"
    data.mkdir()
    (data / "MANIFEST.json").write_text(
        json.dumps(dict(schema="tower-staged-release-v1", files={}))
    )
    document = dict(
        data_release.manifest(),
        payload_files={},
        upload_files={},
        manifest_sha256=sha(data / "MANIFEST.json"),
    )
    monkeypatch.setattr(data_release, "manifest", lambda: document)
    return root, data, accessed


def assert_package_reads(accessed):
    assert DATA / "benchmark/assets.lock.json" in accessed
    assert DATA / "benchmark/runtime/ik-capsule/runtime/scene-cache.json" in accessed


def test_verify_without_private_archive(public_assets):
    root, _, accessed = public_assets
    assert assets.verify(root)["files"] == 1
    assert_package_reads(accessed)


@pytest.mark.parametrize("transient", [False, True])
def test_download_without_private_archive(public_assets, monkeypatch, tmp_path, transient):
    from urllib.error import URLError

    from tower_sim.bench import download

    monkeypatch.setattr(download.time, "sleep", lambda delay: None)
    root, data, accessed = public_assets
    calls = []
    lock = benchmark.read(assets.LOCK)
    prefix = f"https://huggingface.co/datasets/{lock['repository']}/resolve/{lock['revision']}/"

    def urlopen(url, *, timeout):
        assert url.startswith(prefix)
        name = unquote(url.removeprefix(prefix))
        calls.append(name)
        if transient and len(calls) == 1:
            raise URLError("temporary")
        return io.BytesIO((root / name).read_bytes())

    monkeypatch.setattr(assets, "urlopen", urlopen)
    result = assets.download(tmp_path / "download", data_root=data)
    assert result["files"] == 1
    assert set(calls) == {"manifest.json", "station/station-dual-j2-wrist-shared-v3.usdz"}
    assert_package_reads(accessed)


def test_candidate_verify_without_private_archive(public_assets):
    root, data, accessed = public_assets
    result = public.verify(assets=root, data_root=data)
    assert result["tasks"] == 12 and result["scenes"] == 600 and result["assets"]["files"] == 1
    assert_package_reads(accessed)


def test_check_request_reaches_release_check_without_archive(public_assets, tmp_path):
    root, data, accessed = public_assets
    release = tmp_path / "absent-release.json"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            dict(
                task="task1-2",
                assets=str(root),
                data_root=str(data),
                policy_url="ws://localhost:19000",
                release_manifest=str(release),
                release_manifest_sha256="0" * 64,
            )
        )
    )
    with pytest.raises(FileNotFoundError) as error:
        benchmark.main(["check-request", "--request", str(request)])
    assert error.value.filename == str(release)
    assert_package_reads(accessed)


def test_verify_rejects_mismatched_scene_index(public_assets):
    root, _, _ = public_assets
    path = root / "manifest.json"
    body = json.loads(path.read_text())
    body["scene_index_sha256"] = "0" * 64
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="different frozen scene suite"):
        assets.verify(root)


def test_verify_rejects_station_outside_calibration(public_assets, monkeypatch):
    root, _, _ = public_assets
    station = root / "station/station-dual-j2-wrist-shared-v3.usdz"
    expected = sha(station)
    read = benchmark.read

    def unchanged_calibration(path):
        value = read(path)
        if Path(path) == DATA / "benchmark/runtime/ik-capsule/runtime/scene-cache.json":
            value["scene_sha256"] = expected
        return value

    monkeypatch.setattr(benchmark, "read", unchanged_calibration)
    station.write_bytes(b"changed synthetic station")
    path = root / "manifest.json"
    body = json.loads(path.read_text())
    body["files"][0].update(size=station.stat().st_size, sha256=sha(station))
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="Station differs from the campaign calibration"):
        assets.verify(root)


def test_scene_preparation_defaults_to_public():
    import inspect

    assert inspect.signature(assets.prepare_scenes).parameters["reference"].default is False


def test_scene_manifest_requires_benchmark_before_external_io(tmp_path):
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps(dict(schema="tower-frozen-initial-states-v1", assets="missing")))
    with pytest.raises(ValueError, match="must declare its benchmark revision"):
        assets.load_rows(path)
