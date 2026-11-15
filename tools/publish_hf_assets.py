# SPDX-License-Identifier: Apache-2.0
"""Publish versioned USD/USDZ and referenced texture assets to a Hugging Face dataset repository.

This tool never adds binary assets to Git. It uploads the files, computes their
checksums locally, and emits a manifest that can be reviewed and committed.
Install the optional client with ``uv run --with huggingface_hub``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(
    files: list[Path],
    *,
    repo_id: str,
    version: str,
    license_name: str,
    station_license: str | None = None,
    manifest_path: Path,
    root: Path | None = None,
    repo_type: str = "dataset",
    revision: str | None = None,
    tag: str | None = None,
    metadata: list[Path] | None = None,
    source_repository: str = "https://github.com/tower-benchmark/TOWER",
    github_revision: str | None = None,
    evaluator_version: str = "0.1.0",
    dry_run: bool = False,
) -> dict[str, Any]:
    if not files:
        raise ValueError("at least one USD/USDZ file is required")
    for path in files:
        if path.suffix.lower() not in {".usd", ".usda", ".usdz", ".png", ".jpg", ".jpeg"}:
            raise ValueError(f"unsupported asset type (expected USD or image): {path}")
        if not path.is_file():
            raise FileNotFoundError(path)

    # ``revision`` is the upload target (normally the default branch).  A
    # release ``tag`` is created only after all files and the manifest are
    # uploaded, so the tag points at one complete immutable snapshot.
    manifest_revision = tag or revision or version
    entries = []
    metadata_entries = []

    def relative_path(path: Path, *, allow_outside_root: bool = False) -> str:
        path = path.resolve()
        if root is None:
            return path.name
        root_path = root.resolve()
        try:
            return path.relative_to(root_path).as_posix()
        except ValueError:
            if allow_outside_root:
                return path.name
            raise ValueError(f"asset is outside --root: {path}") from None

    for path in files:
        path = path.resolve()
        relative = relative_path(path)
        entry_license = license_name
        if station_license and relative.startswith("station/"):
            entry_license = station_license
        entries.append(
            {
                "id": relative,
                "path": relative,
                "url": f"https://huggingface.co/datasets/{repo_id}/resolve/{manifest_revision}/{relative}",
                "sha256": sha256(path),
                "version": version,
                "license": entry_license,
            }
        )
    for path in metadata or []:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = relative_path(path, allow_outside_root=True)
        metadata_entries.append(
            {
                "path": relative,
                "url": f"https://huggingface.co/datasets/{repo_id}/resolve/{manifest_revision}/{relative}",
                "sha256": sha256(path),
            }
        )
    if github_revision is None:
        try:
            github_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            github_revision = None
    manifest = {
        "schema": "tower-assets-v1",
        "repository": repo_id,
        "revision": manifest_revision,
        "license": "other" if station_license else license_name,
        "source": {
            "repository": source_repository,
            "github_commit": github_revision,
            "evaluator_version": evaluator_version,
        },
        "assets": entries,
        "metadata": metadata_entries,
    }
    if dry_run:
        return manifest

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise RuntimeError(
            "install publishing support with: uv run --with huggingface_hub ..."
        ) from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=entry["path"], path_or_fileobj=path)
        for path, entry in zip(files, entries)
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for path, entry in zip(metadata or [], metadata_entries):
        operations.append(CommitOperationAdd(path_in_repo=entry["path"], path_or_fileobj=path))
    manifest_relative = relative_path(manifest_path, allow_outside_root=True)
    operations.append(
        CommitOperationAdd(path_in_repo=manifest_relative, path_or_fileobj=manifest_path)
    )
    api.create_commit(
        operations=operations,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        commit_message=f"Publish TOWER manifest {version}",
    )
    if tag:
        api.create_tag(
            repo_id=repo_id,
            repo_type=repo_type,
            tag=tag,
            revision=revision or "main",
            tag_message=f"TOWER-Assets release {version}",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--repo-id", default="tower-benchmark/TOWER-Assets", help="Hugging Face dataset identifier"
    )
    parser.add_argument("--version", required=True, help="immutable asset release, e.g. 0.0.1")
    parser.add_argument("--license", required=True, dest="license_name")
    parser.add_argument(
        "--station-license",
        help="license expression for composite station assets (for example, 'MIT AND CC-BY-4.0')",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="local staging root; preserve each asset path relative to this directory",
    )
    parser.add_argument("--manifest", type=Path, default=Path("hf-assets/manifest.json"))
    parser.add_argument(
        "--metadata",
        action="append",
        type=Path,
        default=[],
        help="metadata file to upload; may be repeated",
    )
    parser.add_argument(
        "--revision", help="existing HF branch to upload; defaults to the repository default branch"
    )
    parser.add_argument(
        "--tag",
        help="create this immutable HF tag after uploading (for example, 0.0.1)",
    )
    parser.add_argument("--source-repository", default="https://github.com/tower-benchmark/TOWER")
    parser.add_argument("--github-revision", help="source Git commit; defaults to current HEAD")
    parser.add_argument("--evaluator-version", default="0.1.0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = publish(
        args.files,
        repo_id=args.repo_id,
        version=args.version,
        license_name=args.license_name,
        station_license=args.station_license,
        manifest_path=args.manifest,
        root=args.root,
        revision=args.revision,
        tag=args.tag,
        metadata=args.metadata,
        source_repository=args.source_repository,
        github_revision=args.github_revision,
        evaluator_version=args.evaluator_version,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
