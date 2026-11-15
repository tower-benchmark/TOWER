# SPDX-License-Identifier: Apache-2.0
"""Versioned library APIs and provenance-bearing run reports."""

from importlib import import_module
from typing import cast

from .. import __version__
from ..tasks import benchmark_task
from . import engine
from .computation import REFERENCE
from .detectors import for_task
from .recording import from_campaign_files, from_files
from .rules import PROTOCOL
from .rulesets import select
from .rulesets.bench_2026_11 import FINAL_STATE_REVISIONS

INPUT_PATHS = ("campaign", "files")


def score_recording(
    recording,
    task,
    *,
    ruleset=None,
    rules=None,
    success_rule=None,
    include_disturbances=True,
    computation=REFERENCE,
):
    """Return historical metric JSON, without adding output-file provenance.

    Pure metric documents are used by exact equivalence and in-memory detectors.
    score_run/CLI add the four provenance keys to every newly exported report.
    """
    identity = benchmark_task(task)
    selected = select(ruleset or identity.scorer_ruleset, rules)
    detector = for_task(identity.id)
    rule = success_rule or (
        cast(str, identity.success_rule) if selected.four_place_hold else PROTOCOL
    )
    revision = (
        FINAL_STATE_REVISIONS.get(detector.number) if selected.ruleset == "bench-2026.11" else None
    )
    if revision:
        # Historical final-state wrappers call the trajectory baseline with v5.
        rule = PROTOCOL
    report = engine.score_recording(
        recording,
        identity.legacy_id,
        selected,
        success_rule=rule,
        include_disturbances=include_disturbances,
        computation=computation,
    )
    if revision:
        module = import_module(f".detectors.{detector.finalizer}", __package__)
        return module.finalize(recording, identity.legacy_id, selected, report)
    return report


def provenance(task, ruleset, input_path):
    if input_path not in INPUT_PATHS:
        raise ValueError(f"Unknown scoring input path: {input_path!r}")
    identity = benchmark_task(task)
    selected = select(ruleset or identity.scorer_ruleset)
    return dict(
        ruleset=selected.ruleset,
        input_path=input_path,
        tower_eval_version=__version__,
        canonical_task_id=identity.id,
    )


def score_run(run, task, *, ruleset=None, input_path="files", rules=None):
    """Score a run with explicit construction and report provenance.

    Default ruleset comes from Task.scorer_ruleset. The files path is the
    backwards-compatible CLI default, including its benchmark deadline check.
    """
    identity = benchmark_task(task)
    origin = provenance(identity.id, ruleset, input_path)
    selected = select(origin["ruleset"], rules)
    recording = (
        from_files(run, selected)
        if input_path == "files"
        else from_campaign_files(run, identity.id)
    )
    if identity.deadline_s is not None and recording.times[-1] > identity.deadline_s + 1e-8:
        raise ValueError("Recording exceeds this benchmark's task deadline")
    return {
        **score_recording(recording, identity.id, ruleset=selected.ruleset, rules=selected),
        **origin,
    }
