# SPDX-License-Identifier: Apache-2.0
"""Per-task detector registry, shared by both versioned rulesets."""

from collections.abc import Callable
from dataclasses import dataclass

from ...tasks import benchmark_task, load_tasks
from .grasp import score_transfer
from .push import score_push


@dataclass(frozen=True)
class TaskDetectors:
    number: int
    operation: Callable = score_transfer
    push: bool = False
    tool: bool = False
    extraction: bool = False
    insertion: bool = False
    paired: bool = False
    geometry_task: int = 1
    finalizer: str | None = None


_FAMILIES = {
    1: TaskDetectors(1, paired=True),
    2: TaskDetectors(2, geometry_task=2),
    4: TaskDetectors(4, operation=score_push, push=True, finalizer="task4_final_state"),
    5: TaskDetectors(5, insertion=True, geometry_task=5),
    6: TaskDetectors(6, extraction=True, geometry_task=6, finalizer="task6_completion"),
    7: TaskDetectors(7, tool=True, extraction=True, paired=True),
}
DETECTORS = {
    task.id: _FAMILIES[int(task.legacy_id[4])]
    for task in load_tasks()
    if task.status == "benchmark"
}


def for_task(task):
    return DETECTORS[benchmark_task(task).id]
