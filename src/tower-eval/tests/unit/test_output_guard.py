# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest
from tower_eval import cli


def mark_run(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps({"schema": "tower-eval-report-v1"}))


@pytest.mark.parametrize("protected", ["cwd", "ancestor", "home", "root"])
def test_guard_refuses_protected_paths_even_with_report(tmp_path, monkeypatch, protected):
    cwd = tmp_path / "project/work"
    cwd.mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    output = {"cwd": cwd, "ancestor": cwd.parent, "home": home, "root": Path(cwd.anchor)}[protected]
    if protected != "root":
        mark_run(output)
    removed = []
    monkeypatch.setattr(cli.shutil, "rmtree", removed.append)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        cli._remove_output_safely(output)
    assert removed == []


@pytest.mark.parametrize("report", [None, "not json", "[]", '{"schema": "unrelated-v1"}'])
def test_guard_refuses_unrecognized_directory(tmp_path, report):
    output = tmp_path / "valuable"
    output.mkdir()
    (output / "notes.txt").write_text("keep")
    if report is not None:
        (output / "summary.json").write_text(report)
    with pytest.raises(ValueError, match="TOWER run"):
        cli._remove_output_safely(output)
    assert output.exists()


def test_guard_is_independent_of_package_install_location(tmp_path, monkeypatch):
    cwd = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(
        cli, "__file__", str(tmp_path / ".venv/lib/python3.11/site-packages/tower_eval/cli.py")
    )
    output = cwd / "outputs/run"
    mark_run(output)
    cli._remove_output_safely(output)
    assert not output.exists()
    assert cwd.exists()


def test_guard_does_not_follow_output_or_report_symlinks(tmp_path):
    run = tmp_path / "run"
    mark_run(run)
    alias = tmp_path / "alias"
    alias.symlink_to(run, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        cli._remove_output_safely(alias)
    assert run.exists()
    unrecognized = tmp_path / "unrecognized"
    unrecognized.mkdir()
    (unrecognized / "summary.json").symlink_to(run / "summary.json")
    with pytest.raises(ValueError, match="TOWER run"):
        cli._remove_output_safely(unrecognized)
    assert unrecognized.exists()


def test_guard_refuses_files_and_accepts_absent_output(tmp_path):
    output = tmp_path / "file"
    output.touch()
    with pytest.raises(ValueError, match="not a directory"):
        cli._remove_output_safely(output)
    cli._remove_output_safely(tmp_path / "absent")


def test_guard_allows_recognized_run_inside_source_checkout(tmp_path, monkeypatch):
    repository = tmp_path / "checkout"
    repository.mkdir()
    monkeypatch.chdir(repository)
    monkeypatch.setattr(cli, "__file__", str(repository / "src/tower-eval/src/tower_eval/cli.py"))
    output = repository / "outputs/run"
    mark_run(output)
    cli._remove_output_safely(output)
    assert not output.exists()


def test_guard_accepts_interrupted_run_marker_and_empty_directory(tmp_path):
    interrupted = tmp_path / "interrupted"
    (interrupted / "episodes").mkdir(parents=True)
    (interrupted / "episodes/episode_0000.json").write_text("{}")
    (interrupted / cli.RUN_MARKER).write_text(json.dumps({"schema": cli.RUN_MARKER_SCHEMA}))
    cli._remove_output_safely(interrupted)
    assert not interrupted.exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    cli._remove_output_safely(empty)
    assert not empty.exists()
