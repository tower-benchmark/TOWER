# SPDX-License-Identifier: Apache-2.0
"""Render the task tables from the canonical registry (or check for documentation drift)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/tower-eval/src"))

from tower_eval.tasks import Task, load_tasks

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- task-registry:start -->"
END = "<!-- task-registry:end -->"
DOCUMENTS = {
    "docs/concepts/tasks.md": False,
    "docs/tasks.md": False,
}


def render_table(tasks: tuple[Task, ...], *, benchmark_only: bool = False) -> str:
    """Render identities, timing, scorer routing and planned operations deterministically."""
    lines = [
        "| Task | Canonical id | Legacy id | Deadline | Success rule | Planned G/P | Pushes | Status |",
        "|---|---|---|---:|---|---:|---:|---|",
    ]
    for task in tasks:
        if benchmark_only and task.status != "benchmark":
            continue
        label = f"Task {task.paper[0]} ({task.name})" if task.paper else task.name.capitalize()
        counts = f"{task.counts.planned_g}/{task.counts.planned_p}" if task.counts else "—"
        pushes = task.counts.total_pushes if task.counts else None
        rule = f"`{task.success_rule}`" if task.success_rule else "—"
        deadline = f"{task.deadline_s} s" if task.deadline_s is not None else "—"
        lines.append(
            f"| {label}, {task.layers} layers | `{task.id}` | `{task.legacy_id}` | "
            f"{deadline} | {rule} | {counts} | {pushes if pushes is not None else '—'} | {task.status} |"
        )
    return "\n".join(lines)


def render_document(document: str, tasks: tuple[Task, ...], *, benchmark_only: bool = False) -> str:
    """Replace only the explicitly marked generated table."""
    before, rest = document.split(START)
    _, after = rest.split(END)
    return (
        before
        + START
        + "\n"
        + render_table(tasks, benchmark_only=benchmark_only)
        + "\n"
        + END
        + after
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tasks = load_tasks(ROOT / "benchmarks/2026.11/tasks.json")
    stale = []
    for relative, benchmark_only in DOCUMENTS.items():
        path = ROOT / relative
        original = path.read_text(encoding="utf-8")
        rendered = render_document(original, tasks, benchmark_only=benchmark_only)
        if rendered != original:
            stale.append(relative)
            if not args.check:
                path.write_text(rendered, encoding="utf-8")
    if args.check and stale:
        print("Stale task tables: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
