"""Explicit private archive selection for maintainer evidence, never inferred."""

import hashlib
import json
import os
from pathlib import Path


class ArchiveUnavailable(FileNotFoundError):
    """No archive was requested; an evidence check must report SKIP, never PASS."""


def archive_bundle(root=None):
    selected = root if root is not None else os.environ.get("TOWER_ARCHIVE_ROOT")
    if selected is None:
        raise ArchiveUnavailable(
            "SKIP: supply --archive-root or TOWER_ARCHIVE_ROOT for archived evidence"
        )
    root = Path(selected).resolve()
    bundle = root if (root / "protocol.json").is_file() else root / "simulator/benchmark-202611"
    if not (bundle / "protocol.json").is_file():
        raise ValueError("Invalid explicit archive root: protocol.json is missing")
    return bundle


def archive_checkout(root=None):
    bundle = archive_bundle(root)
    if bundle.parent.name != "simulator" or bundle.name != "benchmark-202611":
        raise ValueError(
            "Archive checkout requires a full export containing simulator/benchmark-202611; a bundle-only root cannot supply checkout evidence"
        )
    checkout = bundle.parents[1]
    if not (checkout / "tests/golden/benchmark-2026.11").is_dir():
        raise ValueError("Invalid archive checkout: tests/golden/benchmark-2026.11 is missing")
    return checkout


class ArchivedPath:
    """Lazy path into the explicitly selected archive for old evidence adapters.

    Constructing adapters does no I/O. Accessing archived bytes requires the root;
    no checkout, tag or current-working-directory fallback is permitted.
    """

    def __init__(self, relative=""):
        self.relative = Path(relative)

    def __truediv__(self, other):
        return type(self)(self.relative / other)

    def resolve_root(self):
        return archive_bundle()

    def __fspath__(self):
        return str(self.resolve_root() / self.relative)

    def __str__(self):
        return self.__fspath__()

    def __repr__(self):
        return f"ArchivedPath({self.relative!r})"

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.resolve_root() / self.relative, name)


def archive_tools(root=None):
    """Locate frozen tools from the private manifest, without publishing its layout."""
    bundle = archive_bundle(root)
    manifest = json.loads((bundle / "SOURCE_MANIFEST.json").read_text())
    anchors = [name for name in manifest["files"] if Path(name).name == "v4_evaluation.py"]
    if len(anchors) != 1:
        raise ValueError("Archive must contain one evaluation tool anchor")
    anchor = (bundle / anchors[0]).resolve()
    if not anchor.is_relative_to(bundle):
        raise ValueError("Archive tool anchor escapes its bundle")
    if hashlib.sha256(anchor.read_bytes()).hexdigest() != manifest["files"][anchors[0]]["sha256"]:
        raise ValueError("Archive tool anchor hash differs")
    return anchor.parent


class ArchivedToolPath(ArchivedPath):
    """Lazy private tool path, including during public test collection."""

    def resolve_root(self):
        return archive_tools()


def archived_file(name, digest, root=None):
    """Select an archived original by basename and pinned bytes, never by an internal path."""
    matches = [
        path
        for path in archive_checkout(root).rglob(name)
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one pinned archived original: {name}")
    return matches[0]


def golden_root(root=None):
    """Original corpus metadata accompanies the explicitly supplied archive."""
    return archive_checkout(root) / "tests/golden/benchmark-2026.11"
