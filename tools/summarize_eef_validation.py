# SPDX-License-Identifier: Apache-2.0
"""Summarize durable Isaac events; never infer task success from completion."""

import argparse
import json
from pathlib import Path

import numpy as np
from tower_eval.artifacts import sha256, write_json


def percentiles(values):
    return None if not len(values) else np.percentile(values, [50, 95, 100], axis=0).tolist()


def matching_implementations(first, second):
    required = {
        "tower_eval/controllers/controller.py",
        "tower_eval/controllers/dual_mode.py",
        "tower_eval/scoring/__init__.py",
        "tools/validate_eef_isaac.py",
    }
    return required <= first.keys() and first == second


def matching_simulators(first, second):
    """Compare effective runtime settings, excluding run output locations."""
    fields = (
        "usd_sha256",
        "scene",
        "backend_config",
        "physics_hz",
        "simulator_hz",
        "instanceable_blocks",
        "coupled_gripper_fingers",
        "render_settings",
        "rendering",
        "camera_rotation_corrections",
        "render_warmup_captures",
        "kinematics",
    )
    identities = []
    for runtime in (first, second):
        source_hash = runtime.get("generator", {}).get("source_sha256")
        if not source_hash or any(runtime.get(key) is None for key in fields):
            return False
        identity = {key: runtime[key] for key in fields}
        identity["generator"] = source_hash
        identity["backend_config"] = {
            key: value for key, value in runtime["backend_config"].items() if key != "output"
        }
        identity["tcp_calibration_cache"] = runtime.get("tcp_calibration_cache", {}).get("sha256")
        identities.append(identity)
    return identities[0] == identities[1]


def summarize(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    records = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    ik = [r for r in records if r["event"] == "ik"]
    executed = [r for r in records if r["event"] == "executed"]
    targets = np.asarray([r["target"] for r in executed])
    scores = [r for r in records if r["event"] == "score"]
    detail = dict(
        output=str(root.resolve()),
        status=summary["status"],
        task_success=summary.get("task_success"),
        summary_sha256=sha256(root / "summary.json"),
        completed_steps=len(executed),
        scored_steps=len(scores),
        last_score=scores[-1]["score"] if scores else None,
        ik_ticks=len(ik),
        ik_failures=[r for r in records if r["event"] == "ik_failure"],
    )
    if len(targets) > 1:
        delta = np.abs(np.diff(targets[:, [*range(7), *range(8, 15)]], axis=0))
        detail["max_command_joint_delta_rad"] = float(delta.max())
        detail["max_command_joint_delta_per_arm_rad"] = [
            float(delta[:, :7].max()),
            float(delta[:, 7:].max()),
        ]
    if ik:
        detail.update(
            backends=sorted({r["backend"] for r in ik}),
            ik_success_rate=float(np.mean([r["success"] for r in ik])),
            fk_errors_p50_p95_max=percentiles(
                [
                    [[a["position_error_m"], a["orientation_error_rad"]] for a in r["arms"]]
                    for r in ik
                ]
            ),
            tracking_p50_p95_max=percentiles([r["tracking_errors"] for r in ik]),
            control_tick_p50_p95_max_s=percentiles([r["tick_wall_s"] for r in ik]),
            control_tick_over_budget_fraction=float(
                np.mean([r["tick_wall_s"] > 1 / 30 for r in ik])
            ),
            rate_limited_ticks=sum(
                any(m["kind"] == "arm_speed_limit" for m in r["modifications"]) for r in ik
            ),
        )
        # These are command-width phases, not assertions that a grasp succeeded.
        # Task success always comes from the independent block-state scorer.
        phases = {}
        inputs = {r["index"]: r["action"] for r in records if r["event"] == "input"}
        for r in ik:
            widths = np.asarray(r["input"]["gripper"])
            previous = inputs.get(r["index"] - 1)
            old = np.asarray(previous["gripper"]) if previous else widths
            if np.any(widths < old - 1e-5):
                phase = "closing_command"
            elif np.any(widths > old + 1e-5):
                phase = "opening_command"
            elif np.any(widths < 0.06):
                phase = "closed_transport_or_contact"
            else:
                phase = "open_approach_or_retreat"
            phases.setdefault(phase, []).append(r["tracking_errors"])
        detail["command_phase_tracking"] = {
            k: dict(count=len(v), errors_p50_p95_max=percentiles(v)) for k, v in phases.items()
        }
    return detail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pair", action="store_true", help="Verify matched Joint/EEF trials")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = [summarize(root) for root in args.runs]
    result = reports
    if args.pair:
        if len(args.runs) != 2:
            raise ValueError("--pair requires Joint then EEF output directories")
        joint, eef = args.runs

        def load(root, name):
            return json.loads((root / name).read_text())

        a, b = load(joint, "initial_state.json"), load(eef, "initial_state.json")
        summaries = [load(r, "summary.json") for r in (joint, eef)]
        runtimes = [load(r, "runtime.json") for r in (joint, eef)]
        input_rows = [
            [
                json.loads(line)
                for line in (r / "events.jsonl").read_text().splitlines()
                if json.loads(line)["event"] == "input"
            ]
            for r in (joint, eef)
        ]
        matched = dict(
            initial_state_hash=a["hash"] == b["hash"],
            implementation=matching_implementations(
                *(r.get("implementation", {}) for r in runtimes)
            ),
            simulator=matching_simulators(*runtimes),
            config=load(joint, "config.json") == load(eef, "config.json"),
            raw_sha256=summaries[0]["raw_sha256"] == summaries[1]["raw_sha256"],
            calibration=summaries[0]["calibration_sha256"] == summaries[1]["calibration_sha256"],
            scorer=summaries[0]["scorer_protocol"] == summaries[1]["scorer_protocol"],
            command_count=len(input_rows[0]) == len(input_rows[1]),
            completed_horizon=all(
                r["status"] == "completed" and r["completed_steps"] == len(rows)
                for r, rows in zip(reports, input_rows)
            ),
            every_eef_tick_logged=reports[1]["ik_ticks"] == reports[1]["completed_steps"],
            no_ik_failures=not reports[1]["ik_failures"],
            gripper_commands=[r["action"]["gripper"] for r in input_rows[0]]
            == [r["action"]["gripper"] for r in input_rows[1]],
            explicit_inputs=all(r["action"]["input_space"] == "joint" for r in input_rows[0])
            and all(r["action"]["input_space"] == "eef" for r in input_rows[1]),
            actual_backend=reports[1].get("backends") == ["pinocchio_casadi"],
        )
        result = dict(
            runs=reports,
            matched_conditions=matched,
            paired_task_acceptance=all(matched.values())
            and all(r["task_success"] is True for r in reports),
        )
    write_json(args.output, result)


if __name__ == "__main__":
    main()
