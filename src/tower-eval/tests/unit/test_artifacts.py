# SPDX-License-Identifier: Apache-2.0
from tower_eval.artifacts import source_identity


def test_source_identity_supports_reference_archive_without_git(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "module.py").write_text("value = 1\n")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"ignored")

    identity = source_identity(source, output)

    assert identity["source_format"] == "archive"
    assert identity["head"] is None
    assert list(identity["file_sha256"]) == ["module.py"]
    assert (output / "source.patch").read_bytes() == b""
