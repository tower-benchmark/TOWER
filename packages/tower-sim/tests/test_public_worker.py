"""Public worker setup and complete source collection never need private archives."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from public_worker_fixture import PROBE
from tower_sim.bench.import_guard import ARCHIVE_STEM_SHA256

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("rows_mode", ["supplied", "prepare"])
def test_public_setup_and_collection_without_archive(tmp_path, monkeypatch, rows_mode):
    monkeypatch.delenv("TOWER_ARCHIVE_ROOT", raising=False)
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(ROOT), str(tmp_path), rows_mode],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=""),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt["passed"] and receipt["public_closure"]
    assert receipt["bootstrap_calls"] == 1 and receipt["filename_and_alias_rejected"]
    assert not receipt["archive_environment_present"]
    assert receipt["prepared_rows"] == int(rows_mode == "prepare")


def test_native_expat_model_requires_exact_stdlib_payload(tmp_path, monkeypatch):
    from importlib.machinery import EXTENSION_SUFFIXES
    from types import SimpleNamespace

    from tower_sim.bench.source_receipt import validate_imports

    from scripts.maintainer import candidate_replay

    stdlib = tmp_path / "stdlib"
    native = stdlib / "lib-dynload"
    native.mkdir(parents=True)
    extension = native / ("pyexpat" + EXTENSION_SUFFIXES[0])
    extension.write_bytes(b"native payload fixture")
    monkeypatch.setattr(candidate_replay.sysconfig, "get_path", lambda name: str(stdlib))
    module = SimpleNamespace(__file__=str(extension))
    monkeypatch.setitem(sys.modules, "pyexpat", module)

    def validate(name, path):
        validate_imports(
            {"modules": {name: {"path": str(path)}}},
            forbidden_stems={"model"},
            allowed_packages=candidate_replay.expected_import_roots(ROOT),
        )

    for name in ("pyexpat.model", "xml.parsers.expat.model"):
        validate(name, extension)
    for name, path in (
        ("unbound.model", extension),
        ("pyexpat.model", native / "model.py"),
        ("xml.parsers.expat.model", tmp_path / extension.name),
    ):
        with pytest.raises(ValueError, match="Forbidden candidate import"):
            validate(name, path)
    # A forged parent, wrong filename and symlink escape cannot admit a copy.
    outside = tmp_path / extension.name
    outside.write_bytes(extension.read_bytes())
    wrong_name = native / "other.so"
    wrong_name.write_bytes(extension.read_bytes())
    for path in (outside, wrong_name):
        module.__file__ = str(path)
        with pytest.raises(ValueError, match="Forbidden candidate import"):
            validate("pyexpat.model", path)
    extension.unlink()
    extension.symlink_to(outside)
    module.__file__ = str(extension)
    with pytest.raises(ValueError, match="Forbidden candidate import"):
        validate("pyexpat.model", outside)


@pytest.mark.archive
def test_pinned_archive_stems_match_explicit_archive():
    from scripts.maintainer.archive_root import archive_bundle
    from scripts.maintainer.candidate_replay import frozen_module_stems

    archive = archive_bundle()
    expected = {
        hashlib.sha256(stem.encode("utf-8")).hexdigest()
        for stem in frozen_module_stems(ROOT, v1_root=archive)
    }
    assert len(expected) == 89 and expected == ARCHIVE_STEM_SHA256
