# SPDX-License-Identifier: Apache-2.0
"""Expose every unresolved owner field in the release source inventory."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.xfail(
    strict=False, reason="owner must fill release placeholders before public release"
)
def test_release_placeholders():
    marker = "TODO" + "(owner):"
    paths = (
        subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT).decode().split("\0")
    )
    hits = []
    for entry in filter(None, paths):
        # Gitlinks are independently versioned repositories, not source files.
        if entry.startswith("160000 "):
            continue
        name = entry.split("\t", 1)[1]
        try:
            lines = (ROOT / name).read_text().splitlines()
        except UnicodeError:
            continue
        hits.extend(
            f"{name}:{number}: {line.strip()}"
            for number, line in enumerate(lines, 1)
            if marker in line
        )
    assert not hits, "Release placeholders:\n" + "\n".join(hits)
