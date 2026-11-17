"""Isaac-free aggregation of hash-bound benchmark episodes."""

import json
import re
from pathlib import Path

from tower_eval.posture_flags import flags
from tower_eval.tasks import benchmark_task

from .resources import save_exclusive, sha
from .verification import read as read_bound

UNREVIEWED = "tower-public-benchmark-protocol-unreviewed-v1"


def qualification(protocol):
    expected = "unreviewed" if protocol.get("schema") == UNREVIEWED else "posture-reviewed"
    value = protocol.get("qualification", "posture-reviewed")
    if value != expected:
        raise ValueError("Qualification differs from bound protocol schema")
    return value


def aggregate(runs, output):
    """Read batch directories (or run roots containing runs/) and publish once."""
    scenes, excluded, seen = [], [], set()
    identity = None
    totals = dict(g=0, p=0, s=0, D=0, planned_g=0, planned_p=0)
    batches = []
    for directory in map(Path, runs):
        if (directory / "runs").is_dir():
            batches.extend(
                sorted(
                    path
                    for path in (directory / "runs").iterdir()
                    if re.fullmatch(r"batch-\d+", path.name) and path.is_dir()
                )
            )
        else:
            batches.append(directory)
    for batch in batches:
        binding = json.loads((batch / "benchmark-request.json").read_text())
        protocol = read_bound(binding["protocol"], binding["protocol_sha256"])
        current = dict(
            benchmark=protocol["public"]["benchmark"],
            ruleset="bench-2026.11",
            task=benchmark_task(protocol["task"]).id,
            package_digest=protocol["package_digest"],
            qualification=qualification(protocol),
        )
        if identity is not None and current != identity:
            raise ValueError("Mixed task/package/benchmark/ruleset/qualification")
        identity = current
        for seed in protocol["seeds"]:
            scene = f"scene-{seed:03d}"
            if scene in seen:
                raise ValueError(f"Duplicate scene: {scene}")
            seen.add(scene)
            episode = batch / "simulation" / f"seed-{seed}"
            status_path = episode / "status.json"
            if not status_path.exists():
                excluded.append(
                    dict(scene=scene, status="infrastructure_failure", reason="Missing status")
                )
                continue
            status = json.loads(status_path.read_text())
            if not status.get("valid_trial") or not status.get("evidence_complete"):
                excluded.append(
                    dict(
                        scene=scene,
                        status="infrastructure_failure",
                        reason=status.get("reason", status["status"]),
                    )
                )
                continue
            if (
                status["protocol_sha256"] != binding["protocol_sha256"]
                or status["package_digest"] != current["package_digest"]
                or status["seed"] != seed
            ):
                raise ValueError("Episode differs from bound protocol")
            score_path = episode / "score.json"
            score_sha = sha(score_path)
            if score_sha != status["source_hashes"]["score.json"]:
                raise ValueError("Aggregate score SHA mismatch")
            states = episode / "episode/states.hdf5"
            if sha(states) != status["source_hashes"]["episode/states.hdf5"]:
                raise ValueError("Aggregate states SHA mismatch")
            report = json.loads(score_path.read_text())
            if (
                report.get("valid") is not True
                or benchmark_task(report["task"]).id != current["task"]
            ):
                raise ValueError("Invalid scene score")
            posture = flags(states, score_path)
            flag_path = episode / "posture-flags.json"
            if flag_path.exists():
                if json.loads(flag_path.read_text()) != posture:
                    raise ValueError("Changed posture flags")
            else:
                save_exclusive(flag_path, posture)
            scenes.append(
                dict(
                    scene=scene,
                    status=status["status"],
                    scores=report["scores"],
                    score_sha256=score_sha,
                    posture=posture,
                )
            )
            for key in totals:
                totals[key] += report[key] if key.startswith("planned_") else report["scores"][key]
    if identity is None:
        raise ValueError("No bound benchmark requests")
    c = totals["g"] + totals["p"]
    e = c + len(scenes) - totals["s"]

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    metrics = dict(
        GSR=ratio(totals["g"], totals["planned_g"]),
        PSR=ratio(totals["p"], totals["planned_p"]),
        LHPR=ratio(c, totals["planned_g"] + totals["planned_p"]),
        TSR=ratio(totals["s"], len(scenes)),
        ODR=ratio(totals["D"], e),
        C=c,
        E=e,
        trials=len(scenes),
    )
    result = dict(
        schema="tower-benchmark-score-v1",
        **identity,
        scenes=sorted(scenes, key=lambda row: row["scene"]),
        metrics=metrics,
        excluded=sorted(excluded, key=lambda row: row["scene"]),
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_exclusive(output / "score.json", result)
    return result
