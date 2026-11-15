# SPDX-License-Identifier: Apache-2.0
"""Paper workbook scoring CLI: python -m reproduction.paper_metrics.scoring_cli --help"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tower_eval.scoring.offline.trajectory import PROTOCOL, digest, score_run
from tower_eval.tasks import benchmark_legacy_id

from ._paper_annotations import read_annotations, resolve_run, validate


def _score(job):
    source, run, task = job
    try:
        return source, score_run(run, task)
    except (ValueError, KeyError, OSError, IndexError) as exc:
        return source, dict(
            protocol=PROTOCOL,
            task=task,
            valid=False,
            scores=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score SIM trajectories for Tasks 1, 2, 4, 5, 6, 7; workbook comparisons exclude Task 7."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--workbook", type=Path, help="SIM source annotations in XLSX format")
    inputs.add_argument("--run", type=Path, help="One recorded run directory")
    parser.add_argument("--task", help="Required with --run, e.g. task6-3")
    parser.add_argument(
        "--runs-root", type=Path, help="Remap workbook paths to a relocated runs directory"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory outside the input recordings"
    )
    parser.add_argument(
        "--push-columns",
        choices=["paper-phase", "arm"],
        default="paper-phase",
        help="Meaning of Task 4 annotation G/P columns (arm: simulator arms 0/1)",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.task is not None:
        try:
            args.task = benchmark_legacy_id(args.task)
        except ValueError as exc:
            parser.error(str(exc))
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workbook and args.task:
        parser.error("--task is only used with --run")
    if args.run and not args.task:
        parser.error("--run requires --task")
    if args.run and args.runs_root:
        parser.error("--runs-root is only used with --workbook")
    if args.workbook:
        episodes, ratings, skipped = read_annotations(args.workbook)
        jobs = [
            (e["source"], resolve_run(e["source"], args.runs_root), e["task"]) for e in episodes
        ]
    else:
        episodes, ratings, skipped = [], [], {}
        jobs = [(str(args.run), args.run, args.task)]
    out = args.output.resolve()
    if any(out.is_relative_to(Path(run).resolve()) for _, run, _ in jobs):
        parser.error("--output must be outside the source run directories")
    out.mkdir(parents=True, exist_ok=False)
    predictions = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_score, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            source, report = future.result()
            predictions[source] = report
            if i % 20 == 0 or i == len(jobs):
                print(f"Scored {i}/{len(jobs)}", file=sys.stderr, flush=True)
    # Deterministic ordering regardless of worker completion order.
    predictions = dict(sorted(predictions.items()))
    (out / "scores.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    if args.workbook:
        validation: dict = validate(episodes, ratings, predictions, args.push_columns)
        validation.update(protocol=PROTOCOL, workbook_sha256=digest(args.workbook), skipped=skipped)
        (out / "validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    k: validation[k]
                    for k in ("episodes", "source_ratings", "scored_episodes", "agreed")
                },
                indent=2,
            )
        )
    failures = {
        source: report["error"] for source, report in predictions.items() if "error" in report
    }
    if failures:
        print(json.dumps(failures, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
