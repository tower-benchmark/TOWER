# SPDX-License-Identifier: Apache-2.0
"""Verify local USD-family references and manifest checksums before release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ASSET_REFERENCE = re.compile(r"@([^@]+)@")
SUPPORTED = {".usd", ".usda", ".usdz", ".png", ".jpg", ".jpeg"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(
    root: Path, manifest: Path | None = None, *, check_manifest_files: bool = True
) -> list[str]:
    root = root.resolve()
    failures: list[str] = []
    candidates = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix in SUPPORTED
    )
    for path in candidates:
        if path.suffix not in {".usda"}:
            continue
        for reference in ASSET_REFERENCE.findall(path.read_text(errors="ignore")):
            if "://" in reference:
                continue
            target = (path.parent / reference).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                failures.append(f"{path.relative_to(root)} references outside root: {reference}")
                continue
            if not target.is_file():
                failures.append(f"{path.relative_to(root)} missing reference: {reference}")
    if manifest is not None:
        document = json.loads(manifest.read_text())
        if document.get("schema") != "tower-assets-v1":
            failures.append("manifest has unsupported schema")
        seen: set[str] = set()
        for item in document.get("assets", []):
            relative_name = str(item.get("path", ""))
            relative = Path(relative_name)
            if not relative_name or relative.is_absolute() or ".." in relative.parts:
                failures.append(f"manifest asset path is not relative: {relative_name!r}")
                continue
            if relative_name in seen:
                failures.append(f"manifest asset path is duplicated: {relative_name}")
                continue
            seen.add(relative_name)
            if not check_manifest_files:
                continue
            target = root / relative
            if not target.is_file():
                failures.append(f"manifest asset missing: {relative}")
                continue
            expected = item.get("sha256")
            if expected and sha256(target) != expected:
                failures.append(f"manifest checksum mismatch: {relative}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="validate manifest structure without requiring staged asset bytes",
    )
    args = parser.parse_args()
    failures = verify(args.root, args.manifest, check_manifest_files=not args.manifest_only)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print(f"asset closure OK: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
