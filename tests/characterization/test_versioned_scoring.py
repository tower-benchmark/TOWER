"""New library paths must preserve full historical documents, without ignores."""

from pathlib import Path
from typing import Any

import pytest
from tower_eval.scoring.api import score_recording, score_run
from tower_eval.scoring.recording import from_campaign_files, from_files
from tower_eval.tasks import benchmark_task

from scripts.maintainer.golden_support import GOLDEN, json_value, read
from tests.characterization.golden_compare import assert_equivalent


@pytest.mark.golden
def test_library_campaign(recording: Path, trial: dict[str, Any], capsys):
    result = score_recording(from_campaign_files(recording, trial["task"]), trial["task"])
    with capsys.disabled():
        assert_equivalent(read(recording / "score.json"), json_value(result))


@pytest.mark.golden
def test_library_files(recording: Path, trial: dict[str, Any], capsys):
    result = score_recording(from_files(recording), trial["task"], ruleset="bench-2026.11")
    with capsys.disabled():
        assert_equivalent(
            read(GOLDEN / (trial["id"] + ".benchmark-score-cli.json")), json_value(result)
        )


@pytest.mark.golden
def test_library_paper(recording: Path, trial: dict[str, Any], capsys):
    r = from_campaign_files(recording, trial["task"])
    if trial["task"].startswith("task5-"):
        with pytest.raises(
            ValueError, match="Insertion slot must have two authored side blocks"
        ) as error:
            score_recording(r, trial["task"], ruleset="paper-v1")
        result = {"exception": type(error.value).__name__, "message": str(error.value)}
    else:
        result = json_value(score_recording(r, trial["task"], ruleset="paper-v1"))
    with capsys.disabled():
        assert_equivalent(read(GOLDEN / (trial["id"] + ".paper-v1.json")), result)


@pytest.mark.golden
def test_run_report_provenance(recording: Path, trial: dict[str, Any], capsys):
    from tower_eval import __version__

    result = score_run(recording, trial["task"], input_path="campaign")
    expected_provenance = dict(
        ruleset="bench-2026.11",
        input_path="campaign",
        tower_eval_version=__version__,
        canonical_task_id=benchmark_task(trial["task"]).id,
    )
    assert {key: result.pop(key) for key in expected_provenance} == expected_provenance
    with capsys.disabled():
        assert_equivalent(read(recording / "score.json"), json_value(result))


@pytest.mark.golden
@pytest.mark.parametrize(
    "status",
    [
        {"status": "infrastructure_failure", "valid_trial": False},
        {"status": "duration_reached", "valid_trial": False},
    ],
)
def test_campaign_files_rejects_invalid_trials(
    recording: Path, trial: dict[str, Any], tmp_path, status
):
    import json
    import shutil

    copy = tmp_path / "run"
    shutil.copytree(recording, copy)
    persisted = json.loads((copy / "status.json").read_text())
    (copy / "status.json").write_text(json.dumps({**persisted, **status}))
    with pytest.raises(ValueError, match="Not a valid campaign trial"):
        from_campaign_files(copy, trial["task"])
