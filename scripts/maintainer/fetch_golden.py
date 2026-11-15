"""Fetch public scoring recordings by immutable commit, SHA-256 and size."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from tower_sim.bench.download import RETRY_ERRORS, download_file

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/golden/benchmark-2026.11/MANIFEST.json"
LOCK = ROOT / "packages/tower-sim/src/tower_sim/data/benchmark/assets.lock.json"
DIRECTORY = "golden/benchmark-2026.11.1"


class GoldenUnavailable(RuntimeError):
    """Public data is unavailable; this is a marked skip, never a pass."""


def verify(path, expected):
    payload = path.read_bytes()
    if (
        len(payload) != expected["size"]
        or hashlib.sha256(payload).hexdigest() != expected["sha256"]
    ):
        raise ValueError(f"Changed public golden recording: {path.name}")
    return path


def golden_archive(name, root=None):
    lock = json.loads(LOCK.read_bytes())
    if not isinstance(lock.get("revision"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", lock["revision"]
    ):
        raise ValueError("Public golden release must use an immutable HF commit")
    manifest = json.loads(MANIFEST.read_bytes())
    if Path(name).name != name or name not in manifest["files"] or not name.endswith(".tar.xz"):
        raise ValueError("Expected a manifest-listed golden recording filename")
    expected = manifest["files"][name]
    explicit = root or os.environ.get("TOWER_GOLDEN_ROOT")
    if explicit is not None:
        directory = Path(explicit)
        if (directory / DIRECTORY).is_dir():
            directory /= DIRECTORY
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Incomplete explicit public golden root: {name}")
        return verify(path, expected)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tower/golden"
    target = cache / expected["sha256"] / name
    if target.is_file():
        return verify(target, expected)
    reason = "SKIP: public golden data unavailable; set TOWER_GOLDEN_ROOT or --golden-root"
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TOWER_OFFLINE") == "1":
        raise GoldenUnavailable(reason + " (offline)")
    base = f"https://huggingface.co/datasets/{lock['repository']}/resolve/{lock['revision']}/"
    try:
        download_file(
            base + DIRECTORY + "/" + quote(name),
            target,
            expected["sha256"],
            size=expected["size"],
            opener=urlopen,
        )
    except RETRY_ERRORS as error:
        raise GoldenUnavailable(reason + f" ({type(error).__name__})") from error
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-root", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_bytes())
    try:
        for name in manifest["files"]:
            if name.endswith(".tar.xz"):
                golden_archive(name, args.golden_root)
    except GoldenUnavailable as error:
        print(str(error))
        return 77
    print("Verified all 24 public golden recordings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
