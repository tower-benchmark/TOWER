# SPDX-License-Identifier: Apache-2.0
"""Run with python -m reproduction.paper_metrics; no simulator dependencies."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from . import VERSION
from .metrics import ARTIFACTS, IDENTITY, LABELS, source_agreement, summarize
from .protocol import TASKS, group_keys, operation_plan

ROOT = Path(__file__).resolve().parent
COLUMNS = [
    *IDENTITY,
    "valid",
    "invalid_reason",
    "s",
    "g",
    "p",
    "d",
    "planned_g",
    "planned_p",
    "annotation_status",
    *ARTIFACTS,
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"CSV columns must be exactly {COLUMNS}")
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row or None in row.values():
                raise ValueError(f"Malformed CSV row {line}")
            for field in ARTIFACTS:
                row[field] = row[field].strip()
            for field in ("layers", "planned_g", "planned_p", *LABELS):
                value = row[field].strip()
                if value and (not value.isascii() or not value.isdecimal()):
                    raise ValueError(f"Line {line}: {field} must be a nonnegative integer or blank")
                row[field] = int(value) if value else None
            if row["valid"] not in ("0", "1"):
                raise ValueError(f"Line {line}: valid must be 0 or 1")
            row["valid"] = row["valid"] == "1"
            rows.append(row)
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_manifest():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("Reference version differs from manifest")
    for name, digest in manifest["files_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != digest:
            raise ValueError(f"Reference checksum mismatch: {name}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Verify the versioned reference and protocol files")
    template = sub.add_parser("template", help="Write 960 empty label rows; not evaluation results")
    template.add_argument("--output", type=Path, required=True)
    agg = sub.add_parser("aggregate", help="Aggregate final human episode labels")
    agg.add_argument("--episodes", type=Path, required=True)
    agg.add_argument(
        "--ratings", type=Path, help="Optional source ratings JSONL for Table 7 procedure"
    )
    agg.add_argument("--output", type=Path, required=True)
    agg.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Diagnostic export only; preserve unavailable metrics",
    )
    args = parser.parse_args(argv)
    try:
        manifest = verify_manifest()
        if args.command == "verify":
            print(json.dumps({"version": VERSION, "verified_files": len(manifest["files_sha256"])}))
            return 0
        if args.output.exists():
            raise ValueError("Output already exists; choose a new path")
        if args.command == "template":
            rows = []
            for policy, task, layers, domain in group_keys():
                ng, np = TASKS[task]["counts"][layers]
                for trial in range(1, 11):
                    rows.append(
                        dict.fromkeys(COLUMNS, "")
                        | {
                            "policy": policy,
                            "task": task,
                            "layers": layers,
                            "domain": domain,
                            "episode_id": f"{domain}-{policy}-{task}{layers}-{trial:02d}",
                            "planned_g": ng,
                            "planned_p": np,
                        }
                    )
            write_csv(args.output, rows, COLUMNS)
            print(
                "Wrote 960 unscored template rows. Set validity, labels, annotation status and artifact references before aggregation."
            )
            return 0
        rows = read_csv(args.episodes)
        result = summarize(rows, strict=not args.allow_incomplete)
        agreement = {"status": "unavailable", "reason": "No source ratings supplied"}
        if args.ratings:
            ratings = [
                json.loads(line) for line in args.ratings.read_text().splitlines() if line.strip()
            ]
            agreement = source_agreement(rows, ratings)
        result["source_agreement"] = agreement
        result["reference_version"] = VERSION
        result["inputs"] = {
            "episodes_sha256": sha256(args.episodes),
            "ratings_sha256": sha256(args.ratings) if args.ratings else None,
            "manifest_sha256": sha256(ROOT / "manifest.json"),
        }
        result["historical_results_verified"] = False
        args.output.mkdir(parents=True)
        write_json(args.output / "summary.json", result)
        table = []
        for group in result["groups"]:
            row = {
                k: group[k]
                for k in (
                    *IDENTITY[:4],
                    "valid_episodes",
                    "invalid_episodes",
                    "odr",
                    "mean_successful_operations",
                )
            }
            row.update(
                {
                    name + "_percent": group[name] * 100 if group[name] is not None else None
                    for name in ("tsr", "gsr", "psr", "lhpr")
                }
            )
            for field, stats in group["statistics"].items():
                row.update({field + "_" + name: value for name, value in stats.items()})
            table.append(row)
        write_csv(args.output / "tables-14-17.csv", table, list(table[0]))
        write_csv(
            args.output / "table-2-pooled-odr.csv",
            result["pooled_model_odr"],
            ["policy", "domain", "valid_episodes", "odr", "rank"],
        )
        write_json(args.output / "table-7-source-agreement.json", agreement)
        write_json(args.output / "result-to-artifact.json", result["result_to_artifact"])
        write_json(args.output / "annotation-status.json", result["annotation_status"])
        write_json(
            args.output / "operation-plans.json",
            {f"{t}{h}": operation_plan(t, h) for t in TASKS for h in (2, 3)},
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "cohort_complete": result["cohort_complete"],
                    "historical_results_verified": False,
                }
            )
        )
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
