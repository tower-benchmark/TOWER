"""Characterize separate scorer contracts without reconciling their semantics.

Authoritative archive tier: exact canonical JSON in the manifest's archived
Linux numerical environment. Portable CI tier: non-float leaves and JSON types
remain exact; floats use golden_compare's 1e-12 absolute/relative tolerances.
Only explicit Qhull vertex-list pointers allow cyclic rotation, never reflection
or arbitrary reordering. Every accepted float difference and rotation is logged.

Archive input is finalize_trial -> from_physical_rows -> score_recording.
The package intentionally differs: Task5 destination/source geometry and the
measured insertion gate, Task6 untouched/final roof rules, Task1-2 four-place
hold, Task4 final-state scoring, Task7 yaw directions and Task7-3 collapse veto.
It has its own paper-v1 snapshots (including Task5's existing insertion-slot
ValueError). The generic benchmark score CLI is another
contract: input file hashes, distal finger corners and persisted status differ
from campaign inputs. StackScorer is an older stack diagnostic and rejects
unsupported task topology. Its complete update streams use the portable
comparator; original stream digests are metadata verified only in the exact tier.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts.maintainer.golden_support import (
    GOLDEN,
    benchmark,
    campaign_score,
    json_value,
    online_score,
    online_snapshot,
    package_score,
    read,
)
from tests.characterization.golden_compare import assert_equivalent


@pytest.fixture
def compare(capsys: pytest.CaptureFixture[str]) -> Callable[[Any, Any], None]:
    def check(expected: Any, actual: Any) -> None:
        # Disable capture around the comparison itself, not fixture setup:
        # accepted differences must remain visible even without pytest -s.
        with capsys.disabled():
            assert_equivalent(expected, actual)

    return check


@pytest.mark.archive
@pytest.mark.golden
def test_frozen_campaign_score(
    recording: Path, trial: dict[str, Any], compare: Callable[[Any, Any], None]
) -> None:
    compare(read(recording / "score.json"), campaign_score(recording, trial["task"]))


@pytest.mark.golden
def test_package_score(
    recording: Path, trial: dict[str, Any], compare: Callable[[Any, Any], None]
) -> None:
    compare(
        read(GOLDEN / (trial["id"] + ".paper-v1.json")), package_score(recording, trial["task"])
    )


@pytest.mark.archive
@pytest.mark.golden
def test_generic_benchmark_score_cli(
    recording: Path, trial: dict[str, Any], compare: Callable[[Any, Any], None]
) -> None:
    compare(
        read(GOLDEN / (trial["id"] + ".benchmark-score-cli.json")),
        json_value(benchmark.score_run(recording, trial["task"])),
    )


@pytest.mark.golden
def test_online_score(
    recording: Path, trial: dict[str, Any], compare: Callable[[Any, Any], None]
) -> None:
    expected = online_snapshot(trial["id"])
    expected.pop("archived_stream_sha256", None)
    compare(expected, online_score(recording))
