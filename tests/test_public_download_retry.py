"""Transport failures cannot publish partial bytes or discard completed public files."""

import hashlib
import io
import json
import ssl
import sys
from http.client import IncompleteRead
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from tower_sim.bench import download


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda delay: None)


@pytest.mark.parametrize("stream_error", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        URLError("offline"),
        ConnectionError(),
        TimeoutError(),
        ssl.SSLError(),
        IncompleteRead(b"partial"),
    ],
)
def test_retry_includes_open_and_read(tmp_path, error, stream_error):
    target = tmp_path / "payload"
    payload = b"complete pinned bytes"
    calls = []

    class Broken(io.BytesIO):
        def read(self, size=-1):
            if self.tell() == 0:
                return super().read(3)
            raise error

    def opener(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            if stream_error:
                return Broken(payload)
            raise error
        return io.BytesIO(payload)

    download.download_file(
        "https://example.invalid/payload",
        target,
        hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        opener=opener,
    )
    assert len(calls) == 2 and target.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("code", [400, 401, 403, 404, 429])
def test_http_4xx_never_retries(tmp_path, code):
    calls = []

    def opener(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.invalid/payload", code, "denied", {}, None)

    with pytest.raises(HTTPError):
        download.download_file(
            "https://example.invalid/payload", tmp_path / "payload", "0" * 64, opener=opener
        )
    assert len(calls) == 1 and not list(tmp_path.iterdir())


def test_retries_are_bounded_and_never_publish(tmp_path):
    calls = []

    def opener(*args, **kwargs):
        calls.append(1)
        raise URLError("offline")

    with pytest.raises(URLError):
        download.download_file(
            "https://example.invalid/payload", tmp_path / "payload", "0" * 64, opener=opener
        )
    assert len(calls) == 5 and not list(tmp_path.iterdir())


def test_verified_files_resume_and_bad_bytes_are_not_retried(tmp_path):
    payload = b"verified"
    target = tmp_path / "payload"
    target.write_bytes(payload)
    calls = []

    def opener(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(payload)

    download.download_file(
        "https://example.invalid/payload",
        target,
        hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        opener=opener,
    )
    assert not calls
    target.write_bytes(b"tampered")
    download.download_file(
        "https://example.invalid/payload",
        target,
        hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        opener=opener,
    )
    assert len(calls) == 1 and target.read_bytes() == payload
    calls.clear()
    with pytest.raises(ValueError, match="checksum/size"):
        download.download_file(
            "https://example.invalid/payload",
            target,
            hashlib.sha256(payload).hexdigest(),
            size=len(payload) + 1,
            opener=opener,
        )
    assert len(calls) == 1 and target.read_bytes() == payload


def test_public_asset_cache_keeps_verified_files_after_interruption(tmp_path, monkeypatch):
    from tower_sim.bench import data_release

    from scripts.maintainer import fetch_public_test_data as assets

    manifest = b'{"files": []}'
    digest = hashlib.sha256(manifest).hexdigest()
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(dict(revision="a" * 40, manifest_sha256=digest)))
    monkeypatch.setattr(assets, "LOCK", lock)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for key in ("TOWER_ASSET_ROOT", "TOWER_OFFLINE", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    partial = tmp_path / "cache/tower/assets" / (".partial-" + digest)
    published = partial.with_name(digest)
    downloaded = []
    attempts = []

    def fetch(root):
        attempts.append(root)
        assert root == partial and not published.exists()

        def opener(*args, **kwargs):
            downloaded.append(1)
            return io.BytesIO(manifest)

        download.download_file(
            "https://example.invalid/manifest", root / "manifest.json", digest, opener=opener
        )
        if len(attempts) == 1:
            raise ssl.SSLError("interrupted later file")

    monkeypatch.setattr(
        assets.importlib, "import_module", lambda name: SimpleNamespace(download=fetch)
    )
    monkeypatch.setattr(data_release, "verify_release", lambda root: {})
    with pytest.raises(assets.PublicAssetsUnavailable):
        assets.public_asset_root()
    assert partial.is_dir() and not published.exists()
    assert assets.public_asset_root() == published
    assert len(downloaded) == 1 and not partial.exists()


@pytest.mark.parametrize("module", ["fetch_golden", "fetch_public_test_data"])
@pytest.mark.parametrize("revision", ["TODO(owner): immutable HF commit", "main", "x" * 40])
def test_malformed_lock_is_error_even_offline(tmp_path, monkeypatch, module, revision):
    import importlib

    helper = importlib.import_module("scripts.maintainer." + module)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(dict(revision=revision)))
    monkeypatch.setattr(helper, "LOCK", lock)
    monkeypatch.setenv("TOWER_OFFLINE", "1")
    with pytest.raises(ValueError, match="immutable HF commit"):
        if module == "fetch_golden":
            helper.golden_archive("task1-2-mme.tar.xz")
        else:
            helper.public_asset_root()


@pytest.mark.parametrize("fault", ["graceful", "reset", 500, 502, 503, 404, "hash"])
def test_real_http_transport_failures(tmp_path, fault):
    import socket
    import struct
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.request import urlopen

    payload = b"pinned" * 40000
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(self.path)
            first = len(requests) == 1
            status = fault if first and isinstance(fault, int) else 200
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload) if status == 200 else 0))
            self.end_headers()
            if status != 200:
                return
            if first and fault in ("graceful", "reset"):
                self.wfile.write(payload[:50000])
                self.wfile.flush()
                if fault == "reset":
                    self.connection.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                    )
                else:
                    self.connection.shutdown(socket.SHUT_WR)
                self.close_connection = True
                return
            self.wfile.write(b"x" * len(payload) if fault == "hash" else payload)

    server = ThreadingHTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = tmp_path / "payload"
    before = len(download.RETRY_EVENTS)
    try:

        def fetch():
            download.download_file(
                f"http://localhost:{server.server_port}/payload",
                target,
                hashlib.sha256(payload).hexdigest(),
                size=len(payload),
                opener=urlopen,
            )

        if fault in (404, "hash"):
            with pytest.raises(HTTPError if fault == 404 else ValueError):
                fetch()
            assert len(requests) == 1 and not target.exists()
            assert len(download.RETRY_EVENTS) == before
        else:
            fetch()
            assert len(requests) == 2 and target.read_bytes() == payload
            assert len(download.RETRY_EVENTS) == before + 1
        assert set(tmp_path.iterdir()) == ({target} if target.exists() else set())
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
        assert not thread.is_alive()


def test_two_processes_share_partial_cache(tmp_path):
    import os
    import subprocess
    import threading
    import time

    from scripts.maintainer import fetch_public_test_data as assets

    payload = b'{"files": []}'
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "lock.json").write_text(json.dumps(dict(revision="a" * 40, manifest_sha256=digest)))
    partial = tmp_path / "cache/tower/assets" / (".partial-" + digest)
    (partial / "nested").mkdir(parents=True)
    (partial / "nested/.download-interrupted").write_bytes(b"partial")
    script = tmp_path / "peer.py"
    script.write_text("""
import fcntl, json, sys, time
from pathlib import Path
from types import SimpleNamespace
from scripts.maintainer import fetch_public_test_data as assets
from tower_sim.bench import data_release
root, role = Path(sys.argv[1]), sys.argv[2]
assets.LOCK = root / "lock.json"
data_release.verify_release = lambda path: {}
original_flock = fcntl.flock

def observe_flock(*args):
    (root / (role + "-locking")).touch()
    return original_flock(*args)
fcntl.flock = observe_flock

def fetch(partial):
    assert not list(partial.rglob(".download-*"))
    partial.mkdir(exist_ok=True)
    (root / (role + "-downloaded")).touch()
    if role == "first":
        deadline = time.monotonic() + 20
        while not (root / "release").exists():
            assert time.monotonic() < deadline, "parent did not release first downloader"
            time.sleep(0.01)
    (partial / "manifest.json").write_bytes(b'{"files": []}')
sys.modules["benchmark_assets"] = SimpleNamespace(download=fetch)
(root / (role + "-result")).write_text(str(assets.public_asset_root()))
""")
    env = dict(os.environ, XDG_CACHE_HOME=str(tmp_path / "cache"))
    env["PYTHONPATH"] = os.pathsep.join([str(assets.ROOT), *map(str, sys.path)])
    for key in ("TOWER_ASSET_ROOT", "TOWER_OFFLINE", "HF_HUB_OFFLINE", "TOWER_ARCHIVE_ROOT"):
        env.pop(key, None)
    children = []

    def wait_for(name):
        deadline = time.monotonic() + 15
        while not (tmp_path / name).exists():
            assert time.monotonic() < deadline, f"missing peer event: {name}"
            assert all(child.poll() is None for child in children), (
                "peer failed before synchronization"
            )
            threading.Event().wait(0.01)

    try:
        children.append(
            subprocess.Popen(
                [sys.executable, str(script), str(tmp_path), "first"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
        wait_for("first-downloaded")
        children.append(
            subprocess.Popen(
                [sys.executable, str(script), str(tmp_path), "peer"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
        wait_for("peer-locking")
        (tmp_path / "release").touch()
        for child in children:
            stdout, stderr = child.communicate(timeout=20)
            assert child.returncode == 0, stdout + stderr
        published = partial.with_name(digest)
        assert not partial.exists()
        assert sorted(p.name for p in tmp_path.glob("*-downloaded")) == ["first-downloaded"]
        assert all(
            (tmp_path / (role + "-result")).read_text() == str(published)
            for role in ("first", "peer")
        )
        assert {
            p.relative_to(published).as_posix() for p in published.rglob("*") if p.is_file()
        } == {"manifest.json"}
    finally:
        (tmp_path / "release").touch()
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.communicate(timeout=20)


@pytest.mark.parametrize("extra", [False, True])
def test_asset_publish_checks_manifest_and_exact_file_set(tmp_path, monkeypatch, extra):
    from tower_sim.bench import data_release

    from scripts.maintainer import fetch_public_test_data as assets

    payload = b'{"files": []}'
    digest = hashlib.sha256(payload).hexdigest()
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(dict(revision="a" * 40, manifest_sha256=digest)))
    monkeypatch.setattr(assets, "LOCK", lock)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for key in ("TOWER_ASSET_ROOT", "TOWER_OFFLINE", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    published = tmp_path / "cache/tower/assets" / digest
    published.mkdir(parents=True)  # An empty directory is not a published release.
    partial = published.with_name(".partial-" + digest)

    def fetch(root):
        root.mkdir()
        (root / "manifest.json").write_bytes(payload)
        if extra:
            (root / "unexpected.bin").write_bytes(b"unlisted")

    monkeypatch.setitem(sys.modules, "benchmark_assets", SimpleNamespace(download=fetch))
    monkeypatch.setattr(data_release, "verify_release", lambda root: {})
    if extra:
        with pytest.raises(ValueError, match="file set differs"):
            assets.public_asset_root()
        assert not (published / "manifest.json").exists()
    else:
        assert assets.public_asset_root() == published
        assert not partial.exists()
