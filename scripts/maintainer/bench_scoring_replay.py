"""CPU campaign replay; frozen code is a test oracle only.

The committed online snapshots are StackScorer's separate diagnostic contract.
Campaign latch sequences are reconstructed at every recorded tick from the
immutable rows and frozen detector, then checked against archived terminals.
"""

import sys

sys.dont_write_bytecode = True
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import patch

from tower_sim.bench.scoring import BenchScorer
from tower_sim.bench.tasks import BenchTask
from tower_sim.bench.termination import terminal_evidence

from scripts.maintainer.golden_support import arrays, json_value, read, reference_scorer
from tests.characterization.runtime_scenarios import runtime_module


def replay(run, task, resources, compare, *, reference=True):
    data = arrays(run)
    rows = [{key: value[i] for key, value in data.items()} for i in range(len(data["time"]))]
    spec, fingers, status = (
        read(run / p) for p in ("episode_spec.json", "finger_geometry.json", "status.json")
    )
    contract = BenchTask.from_resources(task, resources)
    if reference:
        frozen = reference_scorer()
    else:
        from tower_eval.scoring import early_success as frozen
    # Literal frozen batch dispatch, independently of candidate registry/protocol.
    enabled = not task.startswith("task7-") and contract.success_rule in (
        "tower-task12-four-place-hold-v1",
        "tower-sim-trajectory-v5",
    )
    original = (
        (
            frozen.FourPlaceHoldStop()
            if task == "task1-2"
            else frozen.Task2HoldStop()
            if task == "task2-2"
            else frozen.TaskAwareHoldStop(task)
        )
        if enabled
        else None
    )
    scorers = {mode: BenchScorer(contract, mode=mode) for mode in ("off", "fast", "audit")}
    streams = {mode: [] for mode in ("frozen", *scorers)}
    failed = status["status"] == "task_failure"
    for i, row in enumerate(rows):
        tick = round(float(row["time"]) * 400)
        # Recorded policy failures happen at a request boundary after the prior
        # tick's early-stop check. None of these archives fail an observed-joint
        # check before early stop; the terminal comparison catches that mismatch.
        expected = None
        recording = None
        if enabled and tick >= 4000 and tick % 400 == 0:
            record = frozen.from_physical_rows(spec, rows[: i + 1], fingers)
            assert original is not None
            expected = original.check(record, tick)
            if expected is not None:
                assert tick == round(float(rows[-1]["time"]) * 400), (
                    "Stop preceded archived terminal"
                )
            recording = scorers["off"].recording(spec, rows[: i + 1], fingers)
        streams["frozen"].append(dict(tick=tick, success=deepcopy(expected)))
        for mode, scorer in scorers.items():
            actual = scorer.check(recording, tick)
            streams[mode].append(dict(tick=tick, success=deepcopy(actual)))
            # Dict/booleans/ticks are exact, including portable runs.
            assert actual == expected, (task, mode, tick)
    results = {}
    for mode, scorer in scorers.items():
        recording = scorer.recording(spec, rows, fingers, failed=failed)
        stats_before = dict(getattr(scorer.computation, "stats", {}))
        report = json_value(scorer.score(recording))
        assert stats_before == getattr(scorer.computation, "stats", {})
        compare(read(run / "score.json"), report)
        success = streams[mode][-1]["success"]
        tick = round(float(rows[-1]["time"]) * 400)
        failure = (
            dict(
                failure_tick=tick, rejected_action_applied=False, category=status["terminal_reason"]
            )
            if failed
            else None
        )
        terminal = terminal_evidence(
            contract,
            rows,
            [0.0, rows[-1]["time"]],
            [r["time"] for r in rows],
            report,
            tick=tick,
            failure=failure,
            success=success,
        )
        archived_fields = ("status", "simulated_seconds", "terminal_reason")
        assert {key: terminal[key] for key in archived_fields} == {
            key: status[key] for key in archived_fields
        }, (task, mode, terminal)
        if reference and mode == "off":
            original_terminal = frozen_terminal(
                run, contract, spec, rows, fingers, tick, failure, success
            )
        if reference:
            assert terminal == {key: original_terminal[key] for key in terminal}
        results[mode] = dict(
            stream=streams[mode],
            terminal=terminal,
            score=report,
            cache_stats=getattr(scorer.computation, "stats", {}),
        )
    assert streams["off"] == streams["fast"] == streams["audit"] == streams["frozen"]
    return results


def frozen_terminal(run, task, spec, rows, fingers, tick, failure, success):
    """Execute the actual frozen finalizer, doubling persistence/authorization I/O.

    Golden archives retain physical rows and terminal status, not camera images
    or command arrays. Their terminal clock is supplied at those I/O boundaries.
    Separate abort tests execute real held-tick callback order, including t=0.
    """
    module = runtime_module("v4_evaluation")
    context = module.EvaluationContext.__new__(module.EvaluationContext)
    context.task, context.success_rule = task.identity.legacy_id, task.success_rule
    context.duration_ticks = task.duration_ticks
    context.rows = [dict(candidate=dict(seed=1, episode_spec=spec), output=run)]
    context.completed, context.closed_archives = {}, set()
    context.package = dict(checkpoint_sha256="test-only", package_digest="test-only")
    context.request = dict(protocol_sha256="test-only")
    context.write_evidence = lambda *a: run
    evidence = NS(records=[rows], env=NS(finger_geometry=fingers))
    archive = NS(times=[0.0, rows[-1]["time"]], close=lambda **kw: None)
    saved = {}
    with (
        patch.object(
            module, "immutable", lambda path, body: saved.__setitem__(path.name, json_value(body))
        ),
        patch.object(module.formal, "success_scorer", reference_scorer),
        patch.object(module.formal, "file_sha256", lambda path: "test-only"),
    ):
        context.finalize_trial(
            0,
            None,
            evidence,
            [dict(time=r["time"]) for r in rows],
            archive,
            {},
            {},
            tick=tick,
            failure=failure,
            success=success,
        )
    return saved["status.json"]


def main():
    import argparse
    import hashlib
    import json
    import platform
    import tempfile
    from pathlib import Path

    import h5py
    import numpy as np
    import scipy
    from tower_sim.bench.resources import Resources

    from scripts.maintainer.golden_support import ROOT, sha, unpack, write
    from tests.characterization.golden_compare import assert_equivalent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument(
        "--golden-root", type=Path, help="Public golden folder or staged asset root"
    )
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="Allow missing archive evidence (still reported as skipped)",
    )
    args = parser.parse_args()
    import os

    from scripts.maintainer.archive_root import ArchiveUnavailable, archive_bundle, golden_root

    try:
        bundle = archive_bundle(args.archive_root)
    except ArchiveUnavailable as error:
        result = dict(passed=False, skipped=True, reason=str(error))
        write(args.receipt, result)
        print(json.dumps(result))
        return 0 if args.allow_skip else 77
    os.environ["TOWER_ARCHIVE_ROOT"] = str(bundle)
    golden = golden_root(args.archive_root)
    manifest = read(golden / "MANIFEST.json")
    if args.exact:
        if (
            platform.system(),
            platform.machine(),
            platform.python_version(),
            platform.libc_ver(),
        ) != ("Linux", "x86_64", "3.12.3", ("glibc", "2.39")):
            raise RuntimeError("Exact replay requires the archived Linux numerical environment")
        for name, actual in (
            ("numpy", np.__version__),
            ("scipy", scipy.__version__),
            ("h5py", h5py.__version__),
        ):
            if actual != manifest["exact_environment"][name]:
                raise RuntimeError(f"Exact replay version mismatch: {name}")
    resources = Resources.from_roots(checkout=ROOT, assets=ROOT)
    receipts = []
    for trial in manifest["trials"]:
        archive = golden / trial["archive"]
        if sha(archive) != trial["sha256"] or archive.stat().st_size != trial["size"]:
            raise RuntimeError("Golden archive identity differs")
        original_archive_sha256 = sha(archive)
        if args.golden_root is not None:
            from scripts.maintainer.fetch_golden import golden_archive

            archive = golden_archive(trial["archive"], args.golden_root)
        with tempfile.TemporaryDirectory(prefix="tower-bench-replay-") as directory:
            run = Path(directory)
            unpack(archive, run)
            result = replay(
                run,
                trial["task"],
                resources,
                lambda a, b: assert_equivalent(a, b, portable=not args.exact),
            )
            if args.exact:
                expected_bytes = (run / "score.json").read_bytes()
                for mode, value in result.items():
                    actual_bytes = (
                        json.dumps(value["score"], indent=2, allow_nan=False) + "\n"
                    ).encode()
                    assert actual_bytes == expected_bytes, (
                        trial["id"],
                        mode,
                        "serialized score.json bytes",
                    )
        from scripts.maintainer.check_golden_exact import canonical_sha

        receipts.append(
            dict(
                trial=trial["id"],
                archive_sha256=sha(archive),
                original_archive_sha256=original_archive_sha256,
                modes={
                    mode: dict(
                        stream_sha256=canonical_sha(value["stream"]),
                        score_sha256=canonical_sha(value["score"]),
                        terminal=value["terminal"],
                        cache_stats=value["cache_stats"],
                    )
                    for mode, value in result.items()
                },
            )
        )
        print(
            f"{'EXACT' if args.exact else 'PORTABLE'} PASS {trial['id']} frozen+off+fast+audit",
            flush=True,
        )
    write(
        args.receipt,
        dict(
            schema="tower-bench-scoring-replay-v1",
            exact=args.exact,
            corpus_sha256=hashlib.sha256((golden / "MANIFEST.json").read_bytes()).hexdigest(),
            trials=receipts,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
