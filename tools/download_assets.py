# SPDX-License-Identifier: Apache-2.0
"""Download assets listed in a TOWER manifest with checksum verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(manifest: Path, destination: Path) -> None:
    document = json.loads(manifest.read_text())
    if document.get("schema") != "tower-assets-v1":
        raise ValueError("unsupported asset manifest schema")
    for item in document.get("assets", []):
        if not item.get("url"):
            raise ValueError(
                "This legacy manifest has no download URL. For the public "
                "tower-benchmark/TOWER-Assets dataset, run: "
                "python tools/simulator/benchmark_assets.py download --assets simulator-assets"
            )
        path = destination / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(item["url"], path)
        actual = sha256(path)
        if actual != item["sha256"]:
            path.unlink()
            raise ValueError(f"checksum mismatch for {item['path']}: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("assets"))
    args = parser.parse_args()
    download(args.manifest, args.destination)


if __name__ == "__main__":
    main()
