"""Episode-level benchmark metrics.

The metric accumulator deliberately consumes event counts rather than simulator
internals. A backend or task adapter can emit one :class:`EpisodeRecord` when
an episode ends, and the same aggregation is then used for CPU and RTX runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _bounded(value: int, upper: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if value > upper:
        raise ValueError(f"{name} cannot exceed {upper}")


@dataclass(frozen=True)
class EpisodeRecord:
    """One terminal episode record used by the five public benchmark metrics."""

    valid: bool = True
    task_success: bool = False
    progress_completed: int = 0
    progress_total: int = 0
    grasp_successes: int = 0
    grasp_targets: int = 0
    place_successes: int = 0
    place_targets: int = 0
    disturbed_operations: int = 0
    executed_operations: int = 0
    invalid_reason: str | None = None
    # None means legacy/missing evidence; ODR counts cannot determine DBC.
    disturbed_block_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool) or not isinstance(self.task_success, bool):
            raise ValueError("valid and task_success must be booleans")
        _bounded(self.progress_total, 2**31 - 1, "progress_total")
        _bounded(self.progress_completed, self.progress_total, "progress_completed")
        _bounded(self.grasp_targets, 2**31 - 1, "grasp_targets")
        _bounded(self.grasp_successes, self.grasp_targets, "grasp_successes")
        _bounded(self.place_targets, 2**31 - 1, "place_targets")
        _bounded(self.place_successes, self.place_targets, "place_successes")
        _bounded(self.executed_operations, 2**31 - 1, "executed_operations")
        _bounded(self.disturbed_operations, self.executed_operations, "disturbed_operations")
        if self.disturbed_block_count is not None:
            _bounded(self.disturbed_block_count, 2**31 - 1, "disturbed_block_count")
        if self.valid and self.invalid_reason is not None:
            raise ValueError("valid episodes cannot have invalid_reason")
        if not self.valid and not self.invalid_reason:
            raise ValueError("invalid episodes require invalid_reason")


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


@dataclass
class BenchmarkMetrics:
    """Aggregate valid episodes using the TOWER metric definitions.

    Invalid trials are excluded from all rates. Progress and operation rates
    use the sum of their per-episode denominators, while task success is an
    episode count as specified by the benchmark protocol.
    """

    episodes: list[EpisodeRecord] = field(default_factory=list)

    def add(self, episode: EpisodeRecord) -> None:
        self.episodes.append(episode)

    def extend(self, episodes: list[EpisodeRecord]) -> None:
        for episode in episodes:
            self.add(episode)

    def summary(self) -> dict[str, Any]:
        valid = [episode for episode in self.episodes if episode.valid]
        return {
            "valid_episodes": len(valid),
            "invalid_episodes": len(self.episodes) - len(valid),
            "task_success_rate": _ratio(sum(episode.task_success for episode in valid), len(valid)),
            "long_horizon_progress_rate": _ratio(
                sum(episode.progress_completed for episode in valid),
                sum(episode.progress_total for episode in valid),
            ),
            "grasp_success_rate": _ratio(
                sum(episode.grasp_successes for episode in valid),
                sum(episode.grasp_targets for episode in valid),
            ),
            "place_success_rate": _ratio(
                sum(episode.place_successes for episode in valid),
                sum(episode.place_targets for episode in valid),
            ),
            # Mean of episode-local unique counts; never deduplicate across trials.
            "disturbed_block_count": (
                sum(episode.disturbed_block_count for episode in valid) / len(valid)
                if valid and all(episode.disturbed_block_count is not None for episode in valid)
                else None
            ),
        }
