# SPDX-License-Identifier: Apache-2.0
"""Task identity for benchmark 2026.11 and experimental Isaac 5.1 layouts.

The canonical source is ``benchmarks/2026.11/tasks.json``. Builds include that
same file as package data; resolving a name never imports a simulator or scorer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TaskCounts:
    """Planned scoring units; only layerwise pushdown counts every unit as a push."""

    planned_g: int
    planned_p: int
    total_pushes: int | None = None


@dataclass(frozen=True)
class Task:
    """One layer-specific task, with stable identities at each compatibility boundary."""

    id: str
    name: str
    paper: str | None
    legacy_id: str
    legacy: tuple[str, ...]
    generator: str | None
    layers: int
    deadline_s: int | None
    success_rule: str | None
    scorer_ruleset: str | None
    status: Literal["benchmark", "experimental-legacy"]
    counts: TaskCounts | None = None

    @property
    def number(self) -> int:
        """Historical family number at the scene/scorer boundary."""
        return int(self.legacy_id.split("-", 1)[0][4:])

    @property
    def block_count(self) -> int:
        """Six foundations, task-specific movable layers, and Task7's two tools."""
        if self.status != "benchmark":
            raise ValueError("Construction topology requires a benchmark task")
        per_layer = 2 if self.number == 1 else 1 if self.number == 2 else 3
        return 6 + per_layer * self.layers + (2 if self.number == 7 else 0)

    @property
    def generator_id(self) -> str:
        """Legacy layout name used by evaluator configs and asset descriptors."""
        if self.generator is None:
            raise ValueError(f"Task {self.id!r} has no generator layout in this release")
        return f"{self.generator}_{self.layers}"

    @property
    def aliases(self) -> tuple[str, ...]:
        """All accepted names, including the canonical and paper identifiers."""
        generated = (self.generator_id,) if self.generator else ()
        return tuple(
            dict.fromkeys(
                (self.id, *self.legacy, *generated, *((self.paper,) if self.paper else ()))
            )
        )


def load_tasks(path: str | Path | None = None) -> tuple[Task, ...]:
    """Load and validate registry identities from source, a wheel, or an explicit path."""
    if path is not None:
        source = Path(path).read_text(encoding="utf-8")
    else:
        resource = files("tower_eval").joinpath("_data/tasks-2026.11.json")
        # Wheels and sdists carry the registry; editable source reads its single
        # canonical checkout copy. Path traversal also works in shallow installs.
        checkout = Path(__file__).resolve().parent / "../../../../benchmarks/2026.11/tasks.json"
        source = (
            resource.read_text(encoding="utf-8")
            if resource.is_file()
            else checkout.read_text(encoding="utf-8")
        )
    document = json.loads(source)
    if document.get("schema") != "tower-task-registry-v1" or document.get("benchmark") != "2026.11":
        raise ValueError("Expected the tower-task-registry-v1 registry for benchmark 2026.11")
    result = []
    names: dict[str, str] = {}
    ids: set[str] = set()
    for row in document["tasks"]:
        task = Task(
            **{k: v for k, v in row.items() if k not in {"legacy", "counts"}},
            legacy=tuple(row["legacy"]),
            counts=TaskCounts(**row["counts"]) if row.get("counts") else None,
        )
        if task.layers not in (2, 3) or not task.id.endswith(f"-{task.layers}"):
            raise ValueError(f"Invalid layer-specific task id: {task.id}")
        if task.id in ids or task.legacy_id not in task.legacy:
            raise ValueError(f"Duplicate task or missing legacy id: {task.id}")
        ids.add(task.id)
        if task.status == "benchmark":
            if not (task.paper and task.deadline_s and task.success_rule and task.scorer_ruleset):
                raise ValueError(f"Incomplete benchmark task: {task.id}")
        elif task.status != "experimental-legacy" or any(
            value is not None
            for value in (task.paper, task.deadline_s, task.success_rule, task.scorer_ruleset)
        ):
            raise ValueError(f"Invalid experimental task: {task.id}")
        for alias in task.aliases:
            key = alias.casefold()
            if key in names and names[key] != task.id:
                raise ValueError(f"Ambiguous task alias {alias!r}: {names[key]}, {task.id}")
            names[key] = task.id
        result.append(task)
    return tuple(result)


def _key(alias: str) -> str:
    """Case- and separator-insensitive lookup key (`Task 1-2` == `task1_2` == `task1-2`)."""
    key = re.sub(r"[\s_-]+", "-", alias.strip().casefold())
    return re.sub(r"^task-(?=\d)", "task", key)


@lru_cache(maxsize=1)
def _registry() -> dict[str, Task]:
    registry: dict[str, Task] = {}
    for task in load_tasks():
        for alias in task.aliases:
            key = _key(alias)
            if registry.get(key, task) is not task:
                raise ValueError(f"Ambiguous task alias {alias!r}: {registry[key].id}, {task.id}")
            registry[key] = task
    return registry


def resolve(alias: str) -> Task:
    """Resolve a canonical, paper, generator, or legacy name, ignoring case.

    Unknown names raise ValueError with close matches (or canonical choices).
    A family without a layer count is intentionally ambiguous and not accepted.
    """
    if not isinstance(alias, str):
        raise ValueError(f"Task alias must be a string, got {alias!r}")
    registry = _registry()
    key = _key(alias)
    if key in registry:
        return registry[key]
    # Suggest canonical ids (never internal lookup keys).
    suggestions = dict.fromkeys(
        registry[m].id for m in get_close_matches(key, registry, n=8, cutoff=0.6)
    )
    hint = (
        "Close matches: " + ", ".join(list(suggestions)[:5])
        if suggestions
        else "Choose from: " + ", ".join(task.id for task in load_tasks())
    )
    raise ValueError(f"Unknown task {alias!r}. {hint}")


def benchmark_task(alias: str) -> Task:
    """Resolve a name and reject layouts outside the twelve-task benchmark."""
    task = resolve(alias)
    if task.status != "benchmark":
        raise ValueError(f"Task {task.id!r} is experimental-legacy, not a benchmark 2026.11 task")
    return task


def benchmark_legacy_id(alias: str) -> str:
    """Normalize CLI input to the exact task string required by the frozen runtime."""
    return benchmark_task(alias).legacy_id
