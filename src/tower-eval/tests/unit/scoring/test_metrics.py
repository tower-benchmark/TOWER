# SPDX-License-Identifier: Apache-2.0
import pytest
from tower_eval.scoring import BenchmarkMetrics, EpisodeRecord


def test_metrics_follow_episode_denominators_and_ignore_invalid_trials():
    metrics = BenchmarkMetrics()
    metrics.extend(
        [
            EpisodeRecord(
                task_success=True,
                progress_completed=4,
                progress_total=5,
                grasp_successes=2,
                grasp_targets=3,
                place_successes=1,
                place_targets=2,
                disturbed_block_count=3,
            ),
            EpisodeRecord(
                progress_completed=1,
                progress_total=5,
                grasp_successes=1,
                grasp_targets=3,
                place_targets=2,
                disturbed_block_count=2,
            ),
            EpisodeRecord(valid=False, invalid_reason="camera failure"),
        ]
    )
    assert metrics.summary() == {
        "valid_episodes": 2,
        "invalid_episodes": 1,
        "task_success_rate": 0.5,
        "long_horizon_progress_rate": 0.5,
        "grasp_success_rate": 0.5,
        "place_success_rate": 0.25,
        "disturbed_block_count": 2.5,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"progress_completed": 2, "progress_total": 1},
        {"grasp_successes": 2, "grasp_targets": 1},
        {"place_successes": 2, "place_targets": 1},
        {"disturbed_block_count": -1},
        {"disturbed_block_count": True},
    ],
)
def test_episode_record_rejects_inconsistent_counts(kwargs):
    with pytest.raises(ValueError):
        EpisodeRecord(**kwargs)


def test_invalid_episode_requires_reason():
    with pytest.raises(ValueError, match="invalid_reason"):
        EpisodeRecord(valid=False)


def test_legacy_odr_and_missing_evidence_do_not_become_zero_dbc():
    legacy = EpisodeRecord(disturbed_operations=1, executed_operations=2)
    assert BenchmarkMetrics([legacy]).summary()["disturbed_block_count"] is None
    assert BenchmarkMetrics([]).summary()["disturbed_block_count"] is None
    assert (
        BenchmarkMetrics([EpisodeRecord(disturbed_block_count=0)]).summary()[
            "disturbed_block_count"
        ]
        == 0
    )
