# SPDX-License-Identifier: Apache-2.0
"""Scan tracked text and compressed evidence; freeze exceptions must not grow silently."""

import io
import lzma
import subprocess
import tarfile
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
from tower_sim.bench.public_surface import violations

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "simulator/benchmark-202611/"

# A match is exempt only if it is ALSO in the proven byte-bound set below.
# Newly added files within these directories are therefore still scanned.
FROZEN_ALLOWLIST = {}


def scan_text(name, data):
    if b"\0" in data:
        return []
    try:
        text = data.decode("utf-8-sig")
    except UnicodeError:
        return []
    return [
        f"{name}:{number}: {', '.join(hits)}"
        for number, line in enumerate(text.splitlines(), 1)
        if (hits := violations(line, name=name))
    ]


def scan_path(root, name):
    path = root / name
    if name.endswith(".xz") and tarfile.is_tarfile(path):
        findings = []
        # Never extract archive members to disk (including symlinks or traversal paths).
        with tarfile.open(path, "r|xz") as archive:
            for member in archive:
                if member.isfile():
                    stream = archive.extractfile(member)
                    assert stream is not None
                    findings.extend(scan_text(f"{name}!{member.name}", stream.read()))
        return findings
    if name.endswith(".xz"):
        with lzma.open(path) as stream:
            return scan_text(name + "!payload", stream.read())
    # Gitlinks are independently versioned repositories, not tracked file blobs.
    if name == "third_party/XPolicyLab":
        entry = subprocess.check_output(["git", "ls-files", "--stage", "--", name], cwd=root)
        assert entry.startswith(b"160000 "), "XPolicyLab must be a pinned gitlink"
        return []
    return scan_text(name, path.read_bytes())


def frozen_paths():
    return set()


def test_no_internal_leaks():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    names = [name for name in names if name]
    frozen = frozen_paths()
    for pattern, evidence in FROZEN_ALLOWLIST.items():
        assert evidence["reason"].strip()
        assert any(fnmatchcase(name, pattern) and name in frozen for name in names), (
            f"Stale frozen evidence exception: {pattern}"
        )
    findings = []
    from tower_sim.bench.public_surface import check_lineage

    for name in names:
        if (ROOT / name).is_file():
            check_lineage(name, (ROOT / name).read_bytes())
        if name in frozen and any(fnmatchcase(name, pattern) for pattern in FROZEN_ALLOWLIST):
            continue
        findings.extend(scan_path(ROOT, name))
    findings.extend(scan_scene_v2(ROOT))
    assert not findings, "Internal leaks:\n" + "\n".join(findings)


@pytest.mark.parametrize(
    "value",
    [
        "/" + "mnt/private/data",
        "/" + "root/cache",
        "/" + "home/person/code",
        "/" + "Users/person/code",
        "ali" + "yun",
        "worker.ecs." + "internal",
        "worker." + "internal",
        "jenga" + "_sim_task1-run",
        "host-" + ".".join(("10", "1", "2", "3")),
        "jenga" + "_sim",
        "jenga" + "_sim_bench",
        "anonymous" + "-review-assets",
        "submission " + "1378" + "4",
        "double" + "-blind",
        "pypi.nvidia" + ".cn",
        ".".join(("10", "1", "2", "3")),
        ".".join(("8", "8", "8", "8")),
        "The console is at " + ".".join(("8", "130", "188", "198")) + ".",
        "TOWER paper " + "1378" + "4",
        "submission #" + "1378" + "4",
        "ICL" + "R 2027",
        "double" + " blind",
        "Anonymous" + " Authors",
        "after paper " + "acceptance",
        "source_path: " + "mnt/" + "na" + "s/jenga",
    ],
)
def test_detects_leak_classes(value):
    assert violations(value)


@pytest.mark.parametrize(
    "value",
    [
        "http://127.0.0.1:8080",
        "0.0.0.0",
        "http://192.0.2.1:8080",
        "198.51.100.25",
        "203.0.113.2",
        'version = "4.11.0.86"',
        '"isaacsim": "6.1.0.0"',
        "Isaac Sim 6.1.0.0",
        "isaacsim[all,extscache]==6.1.0.0",
        "jenga" + "_sim_v1",
    ],
)
def test_recognizes_documentation_addresses_and_non_address_versions(value):
    assert not violations(value)


@pytest.mark.parametrize("archive", ["evidence.tar.xz", "evidence.xz", "text.xz"])
def test_compressed_text_is_scanned(tmp_path, archive):
    payload = ("private host: " + ".".join(("10", "1", "2", "3"))).encode()
    path = tmp_path / archive
    if archive != "text.xz":
        with tarfile.open(path, "w:xz") as stream:
            info = tarfile.TarInfo("nested/evidence.json")
            info.size = len(payload)
            stream.addfile(info, io.BytesIO(payload))
    else:
        path.write_bytes(lzma.compress(payload))
    assert scan_path(tmp_path, archive)


def scan_scene_v2(root):
    """Every new revision file is scanned, including untracked staging additions."""
    directory = root / "benchmarks/2026.11.1"
    from tower_sim.bench.data_release import manifest
    from tower_sim.bench.public_surface import check_lineage

    findings = []
    actual = {p.relative_to(root).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest()["files"]):
        findings.append("Public metadata file set differs from public-data.json")
    for path in directory.rglob("*"):
        if path.is_file():
            name = path.relative_to(root).as_posix()
            assert not any(fnmatchcase(name, pattern) for pattern in FROZEN_ALLOWLIST)
            findings.extend(scan_path(root, name))
            check_lineage(name, path.read_bytes())
    return findings


def test_scene_v2_zero_exemptions():
    assert not scan_scene_v2(ROOT)


@pytest.mark.parametrize(
    "value",
    [
        "paper-" + "isaac51",
        "refs/" + "tags/example",
        "tag " + "benchmark-2026.11.0",
        "benchmark-2026.11.0" + ":path",
        "jenga" + "-desktop",
    ],
)
def test_tag_labels_in_compressed_members_and_wheels(tmp_path, value):
    import zipfile

    from tower_sim.bench.verification import wheel_payload

    data = value.encode()
    archive = tmp_path / "evidence.tar.xz"
    with tarfile.open(archive, "w:xz") as stream:
        info = tarfile.TarInfo("notes.txt")
        info.size = len(data)
        stream.addfile(info, io.BytesIO(data))
    assert scan_path(tmp_path, archive.name)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as stream:
        stream.writestr("notes.txt", data)
    with pytest.raises(ValueError, match="Private identifiers"):
        wheel_payload(output.getvalue(), "tower-sim", "0.1.0")
