"""Archive opt-in cannot turn unmarked or private-gate failures into skips."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("marked,private", [(False, False), (True, False), (True, True)])
def test_archive_skip_requires_marker_and_absent_root(tmp_path, marked, private):
    (tmp_path / "conftest.py").write_text((ROOT / "conftest.py").read_text())
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = archive: private evidence\n")
    (tmp_path / "test_probe.py").write_text(
        "import pytest\nfrom scripts.maintainer.archive_root import ArchiveUnavailable\n"
        + ("@pytest.mark.archive\n" if marked else "")
        + "def test_probe():\n    raise ArchiveUnavailable('SKIP: supply --archive-root')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "TOWER_ARCHIVE_ROOT"}
    if private:
        (tmp_path / "protocol.json").write_text("{}")
        env["TOWER_ARCHIVE_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == (0 if marked and not private else 1), result.stdout + result.stderr
    if marked and not private:
        assert "test_probe.py:3" in result.stdout
    else:
        assert "1 failed" in result.stdout


def test_unmarked_archive_reader_assertion_stays_failed(tmp_path):
    (tmp_path / "conftest.py").write_text((ROOT / "conftest.py").read_text())
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "test_probe.py").write_text(
        "from scripts.maintainer.archive_root import archive_bundle\n"
        "def test_probe():\n"
        f"    assert (archive_bundle({str(tmp_path)!r}) / 'protocol.json').read_text() == 'wrong'\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 1 and "1 failed" in result.stdout, result.stdout + result.stderr


def test_private_gate_rejects_explicit_archive_skip(tmp_path):
    (tmp_path / "conftest.py").write_text((ROOT / "conftest.py").read_text())
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "test_probe.py").write_text(
        "import pytest\n"
        '@pytest.mark.archive\ndef test_probe():\n    pytest.skip("SKIP: supply --archive-root")\n'
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(tmp_path), "--archive-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "zero archive skips" in result.stdout, (
        result.stdout + result.stderr
    )


def test_documented_archive_option_with_out_of_tree_root(tmp_path, monkeypatch):
    checkout, archive = tmp_path / "checkout", tmp_path / "private-evidence"
    checkout.mkdir()
    archive.mkdir()
    (archive / "protocol.json").write_text("{}")
    (checkout / "conftest.py").write_text((ROOT / "conftest.py").read_text())
    (checkout / "pytest.ini").write_text("[pytest]\nmarkers = archive: private evidence\n")
    (checkout / "test_probe.py").write_text(
        "from scripts.maintainer.archive_root import archive_bundle\n"
        "def test_probe():\n"
        f"    assert archive_bundle() == __import__('pathlib').Path({str(archive)!r})\n"
    )
    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--archive-root=" + str(archive), "-q"],
        cwd=checkout,
        env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout and "Archive evidence skips: 0" in result.stdout
