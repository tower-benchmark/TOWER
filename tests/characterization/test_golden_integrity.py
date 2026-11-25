"""Inspect all committed scoring payloads, including binary metadata, for leaks."""

from __future__ import annotations

import io
import json
import lzma
import re
import tarfile
from typing import Any

import h5py
import numpy as np
import pytest

from scripts.maintainer.golden_support import GOLDEN, ROOT, golden_archive, read, sha

# Construct fragments so the checker can inspect its own source text too.
FORBIDDEN = re.compile(
    "|".join(
        (
            r"/" + "mnt/",
            r"/" + r"root(?:/|\b)",
            r"/" + r"home/[^/\s]+",
            r"/" + "Users/",
            "aliy" + "un",
            "jenga" + "_sim",
            r"(?<![\w.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
            r"172\.(?:1[6-9]|2[0-9]|3[01])(?:\.\d{1,3}){2})(?![\w]|\.\d)",
            "gh" + "p_[A-Za-z0-9]+",
            "github" + "_pat_[A-Za-z0-9]+",
        )
    )
)


def check_text(value: Any, name: str) -> None:
    assert not FORBIDDEN.search(str(value)), f"Private metadata in {name}"


@pytest.mark.parametrize("version", ["10.3.3", "v10.3.3", "192.168.1", "172.16.1"])
def test_privacy_scan_accepts_three_component_versions(version):
    check_text(f"uses: astral-sh/setup-uv@commit # {version}", "workflow")


@pytest.mark.parametrize(
    "octets", [(10, 0, 0, 1), (192, 168, 1, 2), (172, 16, 0, 1), (172, 31, 255, 254)]
)
@pytest.mark.parametrize("wrapper", ["{}", "http://{}:8080/path", '"{}"', "host={}", "Address {}."])
def test_privacy_scan_rejects_private_ipv4(octets, wrapper):
    address = ".".join(map(str, octets))
    with pytest.raises(AssertionError, match="Private metadata"):
        check_text(wrapper.format(address), "fixture")


@pytest.mark.parametrize(
    "octets", [(8, 8, 8, 8), (172, 15, 0, 1), (172, 32, 0, 1), (192, 169, 1, 2)]
)
def test_privacy_scan_accepts_public_ipv4(octets):
    check_text(".".join(map(str, octets)), "fixture")


@pytest.mark.golden
def test_golden_payload_integrity_and_privacy() -> None:
    manifest = read(GOLDEN / "MANIFEST.json")
    assert len(manifest["trials"]) == 24
    assert len({row["task"] for row in manifest["trials"]}) == 12
    assert any(
        row.get("observation_history", {}).get("frames_collected", 0) > 0
        for row in manifest["trials"]
    )
    for row in manifest["trials"]:
        path = golden_archive(row["archive"])
        assert path.stat().st_size == row["size"] and sha(path) == row["sha256"]
        with tarfile.open(path, "r:xz") as archive:
            names = archive.getnames()
            assert set(names) == {
                "episode_spec.json",
                "status.json",
                "score.json",
                "finger_geometry.json",
                "episode/states.hdf5",
                "finger-link-poses.npz",
                "finger-feedback.npz",
            }
            for member in archive:
                assert member.isfile()
                check_text(member.name, path.name)
                stream = archive.extractfile(member)
                assert stream is not None
                payload = stream.read()
                if member.name.endswith(".json"):
                    check_text(payload.decode(), member.name)
                    if member.name == "score.json":
                        import hashlib

                        assert hashlib.sha256(payload).hexdigest() == row["source_score_sha256"]
                elif member.name.endswith(".hdf5"):
                    with h5py.File(io.BytesIO(payload), "r") as data:
                        assert not data.attrs
                        assert set(data) == {"time", "blocks", "block_velocities", "qpos"}
                        for name in data:
                            assert not data[name].attrs
                else:
                    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                        assert set(data.files) in ({"time", "poses_m"}, {"time", "forces_n"})
                        assert all(data[name].dtype.kind == "f" for name in data.files)


def test_new_committed_text_has_no_private_paths() -> None:
    roots = (ROOT / "scripts", ROOT / "tests", ROOT / ".github/workflows")
    for root in roots:
        paths = root.rglob("*") if root.is_dir() else [root]
        for path in paths:
            if path.suffix in (".py", ".json", ".md", ".yml"):
                assert "PROVENANCE.local" not in path.name
                check_text(path.read_text(), str(path.relative_to(ROOT)))


@pytest.mark.golden
def test_compressed_online_streams_and_total_size() -> None:
    # The cap includes recordings, full streams and every other golden file.
    assert (
        sum(path.stat().st_size for path in GOLDEN.parent.rglob("*") if path.is_file())
        <= 15_000_000
    )
    for path in GOLDEN.glob("*.online.json.xz"):
        payload = lzma.decompress(path.read_bytes())
        check_text(payload.decode(), path.name)
        snapshot = json.loads(payload)
        assert (
            payload
            == json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        assert len(snapshot["events"]) == snapshot["updates"]
        assert re.fullmatch(r"[0-9a-f]{64}", snapshot["archived_stream_sha256"])
        # This digest is deliberately verified only by check_golden_exact.py.
