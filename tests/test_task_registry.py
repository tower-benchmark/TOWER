# SPDX-License-Identifier: Apache-2.0
"""Identity, protocol, packaging source and documentation contracts."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from tower_eval.config import load_config
from tower_eval.tasks import benchmark_task, load_tasks, resolve

from reproduction.paper_metrics.protocol import TASKS as PAPER_TASKS
from scripts.maintainer import archived_benchmark, render_task_table
from scripts.maintainer.golden_support import benchmark

ROOT = Path(__file__).resolve().parents[1]
TASKS = load_tasks()


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.id)
def test_every_alias_resolves_case_insensitively(task):
    for alias in task.aliases:
        assert resolve(alias) == task
        assert resolve(alias.swapcase()) == task
    if task.generator is not None:
        assert resolve(task.generator_id) == task


@pytest.mark.archive
def test_frozen_protocol_is_authoritative():
    protocol = benchmark.read(archived_benchmark.BUNDLE / "protocol.json")
    active = {task.legacy_id: task for task in TASKS if task.status == "benchmark"}
    assert len(TASKS) == 14
    assert set(active) == set(protocol["tasks"])
    for legacy, settings in protocol["tasks"].items():
        task = active[legacy]
        assert task.deadline_s == settings["duration_s"]
        assert task.success_rule == settings["success_rule"]
        assert task.scorer_ruleset == "bench-2026.11"
        paper = PAPER_TASKS[task.paper[0]]
        assert legacy == f"task{paper['legacy']}-{task.layers}"
        assert task.name == paper["name"].lower()
        assert task.id == f"{paper['name'].lower().replace(' ', '-')}-{task.layers}"
    # The paper JSON specifies a separate human-annotation protocol, not an A-F map.
    paper_protocol = json.loads((ROOT / "reproduction/paper_metrics/protocol.json").read_text())
    assert paper_protocol["duration_limit_seconds"] == 420
    assert "tasks" not in paper_protocol


@pytest.mark.archive
def test_counts_equal_frozen_scorer_and_paper():
    scorer = benchmark.load_scorer()
    trajectory = importlib.import_module(scorer.__package__ + ".offline.trajectory")
    for task in TASKS:
        if task.status != "benchmark":
            continue
        number, layers, g, p = trajectory.task_counts(task.legacy_id)
        assert task.layers == layers
        assert (task.counts.planned_g, task.counts.planned_p) == (g, p)
        assert PAPER_TASKS[task.paper[0]]["counts"][layers] == (g, p)
        assert task.counts.total_pushes == (g + p if number == 4 else None)
    assert resolve("E2").counts.total_pushes == 6
    assert resolve("E3").counts.total_pushes == 9


@pytest.mark.parametrize("alias", ["task3-2", "aligned_stack_3", "task_000016"])
def test_experimental_layouts_have_no_benchmark_claim(alias):
    task = resolve(alias)
    assert task.status == "experimental-legacy"
    assert (task.paper, task.deadline_s, task.success_rule, task.scorer_ruleset, task.counts) == (
        None,
    ) * 5
    with pytest.raises(ValueError, match="experimental-legacy"):
        benchmark_task(alias)


def test_unknown_alias_suggests_close_matches():
    with pytest.raises(ValueError, match="Unknown task.*Close matches:.*two-block-transfer-3"):
        resolve("two-block-tranfer-3")
    with pytest.raises(ValueError, match="Choose from:"):
        resolve("")
    with pytest.raises(ValueError, match="must be a string"):
        resolve(None)


def test_duplicate_alias_cannot_silently_change_identity(tmp_path):
    data = json.loads((ROOT / "benchmarks/2026.11/tasks.json").read_text())
    data["tasks"][1]["legacy"].append(data["tasks"][0]["id"].upper())
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Ambiguous task alias"):
        load_tasks(path)


@pytest.mark.parametrize("relative,benchmark_only", render_task_table.DOCUMENTS.items())
def test_generated_tables_are_current(relative, benchmark_only):
    document = (ROOT / relative).read_text()
    assert document == render_task_table.render_document(
        document, TASKS, benchmark_only=benchmark_only
    )


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.id)
def test_cli_config_aliases_preserve_legacy_asset_names(task, tmp_path):
    path = tmp_path / "evaluation.toml"
    path.write_text(f'[environment]\ntask = "{task.id}"\n')
    if task.generator is None:
        with pytest.raises(ValueError, match="no generator layout"):
            load_config(path)
        return
    assert load_config(path).environment["task"] == task.generator_id
    assert (
        load_config(path, [f'environment.task="{task.legacy_id.upper()}"']).environment["task"]
        == task.generator_id
    )
    with pytest.raises(ValueError, match="Unknown task"):
        load_config(path, ['environment.task="no-such-task"'])


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("Task 2-2", "cross-transfer-2"),
        ("task 7-3", "tool-assisted-extraction-3"),
        ("task7_2", "tool-assisted-extraction-2"),
        ("TASK1-2", "two-block-transfer-2"),
        ("b-3", "two-block-transfer-3"),
    ],
)
def test_separator_variants_resolve(alias, expected):
    assert resolve(alias).id == expected


def test_unknown_task_suggests_canonical_ids_only():
    with pytest.raises(ValueError) as caught:
        resolve("Two-Block Transfer")
    message = str(caught.value)
    assert "two-block-transfer-2" in message
    assert all(
        token.strip() in {t.id for t in TASKS}
        for token in message.split("Close matches: ")[1].split(",")
    )
