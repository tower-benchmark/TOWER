# SPDX-License-Identifier: Apache-2.0
"""Offline Markdown link validation; no network or third-party dependencies."""

import json
import re
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def prose(text):
    return re.sub(r"(?ms)^\s*(`{3,}|~{3,}).*?^\s*\1\s*$", "", text)


def anchors(text):
    text = prose(text)
    found = set(re.findall(r"(?:id|name)=[\"\']([^\"\']+)", text))
    counts = {}
    headings = re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", text)
    headings += re.findall(r"(?m)^([^\n]+)\n(?:===+|---+)\s*$", text)
    for heading in headings:
        heading = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", heading)
        slug = unicodedata.normalize("NFKD", heading).encode("ascii", "ignore").decode()
        slug = re.sub(r"[^\w\s-]", "", slug.lower()).strip()
        slug = re.sub(r"[-\s]+", "-", slug)
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        found.add(slug if not count else f"{slug}_{count}")
        found.add(slug if not count else f"{slug}-{count}")
    return found


def links(text):
    text = prose(text)
    refs = dict(re.findall(r"(?m)^\s*\[([^]]+)\]:\s*<?([^\s>]+)>?", text))
    refs = {key.lower(): value for key, value in refs.items()}
    for match in re.finditer(r"!?\[[^]]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", text):
        yield match.group(1).strip("<>")
    for label, key in re.findall(r"!?\[([^]]+)\]\[([^]]*)\]", text):
        yield refs.get((key or label).lower(), "missing-reference:" + (key or label))
    yield from re.findall(r"(?:href|src)=[\"\']([^\"\']+)", text)


def check_page(path):
    errors = []
    for link in links(path.read_text()):
        url = urlsplit(link)
        if link.startswith("missing-reference:"):
            errors.append(link)
            continue
        if url.scheme or url.netloc or link.startswith("/"):
            continue
        target = (path.parent / unquote(url.path)).resolve() if url.path else path
        if not target.exists() or (
            url.fragment
            and target.suffix == ".md"
            and unquote(url.fragment) not in anchors(target.read_text())
        ):
            errors.append(link)
    return sorted(set(errors))


def findings():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    return {
        name: errors
        for name in names
        if name.endswith(".md")
        and (
            name.startswith("docs/")
            or name
            in {
                "README.md",
                "CONTRIBUTING.md",
                "src/tower-eval/README.md",
                "packages/nero-kinematics/README.md",
            }
        )
        and (ROOT / name).is_file()
        and (errors := check_page(ROOT / name))
    }


def test_internal_links():
    assert not findings(), json.dumps(findings(), indent=2)


def test_link_checker_detects_missing_file_and_anchor(tmp_path):
    target = tmp_path / "target.md"
    target.write_text("# Present\n")
    page = tmp_path / "index.md"
    page.write_text("[ok](target.md#present) [bad](target.md#absent) [missing](gone.md)")
    assert check_page(page) == ["gone.md", "target.md#absent"]
    page.write_text("[reference][ref]\n\n[ref]: target.md#absent\n")
    assert check_page(page) == ["target.md#absent"]


if __name__ == "__main__":
    test_internal_links()
    print("Internal documentation links passed")  # noqa: T201 — standalone check


def test_readme_uses_public_multi_scene_benchmark():
    text = (ROOT / "README.md").read_text()
    assert "benchmark_run.py run" in text
    assert "benchmark_diagnostic.py" in text
    assert "--pilot-review" not in text
