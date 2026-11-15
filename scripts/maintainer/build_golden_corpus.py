"""Extract scoring-only recordings from an explicit local provenance selection.

Input: {"trials": [{"id", "source", "task", "seed", "model", "success"}]}.
Sources and local provenance are never copied into the repository. All output
is staged first; --commit-root copies only the budgeted, verified artifacts.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.maintainer.golden_support import arrays, campaign_score, read, sha, write
from tests.characterization.golden_compare import assert_equivalent

BLOCK_FIELDS = ("name", "tower", "layer", "slot", "pose", "dimensions_m", "fixed_on_table")
TRANSFER_FIELDS = ("arm", "block_id", "block_index", "source_layer", "destination_layer", "goal")
STATUS_FIELDS = (
    "status",
    "returncode",
    "valid_trial",
    "simulated_seconds",
    "terminal_reason",
    "seed",
)
METADATA_FIELDS = ("catalog_task", "movable_layers", "evaluation_task", "task_variant")


def select(body: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: body[key] for key in keys if key in body}


def scrub_spec(body: dict[str, Any]) -> dict[str, Any]:
    metadata = select(body["layout_metadata"], METADATA_FIELDS)
    if "push_targets" in body["layout_metadata"]:
        metadata["push_targets"] = [
            select(row, ("block_id", "block_index", "arm", "layer", "slot", "direction", "order"))
            for row in body["layout_metadata"]["push_targets"]
        ]
    if "cycles" in body["layout_metadata"]:
        metadata["cycles"] = [
            {"direction": row["direction"]} for row in body["layout_metadata"]["cycles"]
        ]
    return dict(
        logical_id=body["logical_id"],
        seed=body["seed"],
        blocks=[select(row, BLOCK_FIELDS) for row in body["blocks"]],
        transfers=[select(row, TRANSFER_FIELDS) for row in body["transfers"]],
        layout_metadata=metadata,
    )


def extract(source: Path, destination: Path, row: dict[str, Any]) -> None:
    destination.mkdir()
    (destination / "episode").mkdir()
    status = read(source / "status.json")
    # Verify source-bound archive bytes before changing their representation.
    for name in (
        "episode_spec.json",
        "episode/states.hdf5",
        "finger-link-poses.npz",
        "finger-feedback.npz",
        "finger_geometry.json",
        "score.json",
    ):
        if sha(source / name) != status["source_hashes"][name]:
            raise ValueError(f"Source hash differs: {row['id']}/{name}")
    write(destination / "episode_spec.json", scrub_spec(read(source / "episode_spec.json")))
    write(destination / "status.json", select(status, STATUS_FIELDS))
    geometry = read(source / "finger_geometry.json")
    write(
        destination / "finger_geometry.json",
        [
            {
                "pinch_center_in_flange_m": arm["pinch_center_in_flange_m"],
                "fingers": [
                    {"distal_link_corners_m": finger["distal_link_corners_m"]}
                    for finger in arm["fingers"]
                ],
            }
            for arm in geometry
        ],
    )
    data = arrays(source)
    with h5py.File(destination / "episode/states.hdf5", "w") as stream:
        for key in ("time", "blocks", "block_velocities", "qpos"):
            stream.create_dataset(key, data=data[key], track_times=False)
    np.savez_compressed(
        destination / "finger-link-poses.npz", time=data["time"], poses_m=data["finger_poses"]
    )
    np.savez_compressed(
        destination / "finger-feedback.npz", time=data["time"], forces_n=data["finger_forces"]
    )
    shutil.copyfile(source / "score.json", destination / "score.json")
    assert_equivalent(
        read(source / "score.json"), campaign_score(destination, row["task"]), portable=False
    )


def pack(directory: Path, output: Path) -> None:
    with tarfile.open(output, "w:xz", preset=9) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                payload = path.read_bytes()
                entry = tarfile.TarInfo(path.relative_to(directory).as_posix())
                entry.size, entry.mode, entry.mtime = len(payload), 0o644, 0
                archive.addfile(entry, io.BytesIO(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--raw-root", type=Path, help="Use local RAW_ROOT/<id> instead of original source paths"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--commit-root", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selection = read(args.selection)
    rows = []
    for selected in selection["trials"]:
        trial_id = selected["id"]
        if not re.fullmatch(r"task[124567]-[23]-(openwam|mme)", trial_id):
            raise ValueError("Unexpected trial id")
        source = args.raw_root / trial_id if args.raw_root else Path(selected["source"])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / trial_id
            extract(source, run, selected)
            name = trial_id + ".tar.xz"
            pack(run, args.output / name)
        row = select(selected, ("id", "task", "seed", "model", "success"))
        row.update(
            archive=name,
            sha256=sha(args.output / name),
            size=(args.output / name).stat().st_size,
            scorer="benchmark-2026.11.0:finalize_trial/from_physical_rows/score_recording",
            source_score_sha256=sha(source / "score.json"),
        )
        if (source / "mme-history.json").exists():
            row["observation_history"] = select(
                read(source / "mme-history.json"),
                (
                    "frames_collected",
                    "lossless_history_bytes",
                    "closed",
                    "extra_render_or_physics_calls",
                ),
            )
        rows.append(row)
        print(f"Verified {trial_id}: {row['size']} bytes", flush=True)
    # Maximize whole-trial count within 15 MB, retaining every variant first.
    chosen: set[str] = set()
    used = 0
    for task in sorted({row["task"] for row in rows}):
        row = min((r for r in rows if r["task"] == task), key=lambda r: r["size"])
        chosen.add(row["id"])
        used += row["size"]
    if used > 15_000_000:
        raise ValueError("One recording per task exceeds the commit budget")
    for row in sorted(rows, key=lambda r: (r["size"], r["id"])):
        if row["id"] not in chosen and used + row["size"] <= 15_000_000:
            chosen.add(row["id"])
            used += row["size"]
    for row in rows:
        row["committed"] = row["id"] in chosen
    manifest = dict(
        schema="tower-golden-recordings-v1",
        reference="benchmark-2026.11.0",
        exact_environment=read(args.environment),
        trials=rows,
        committed_compressed_bytes=used,
        files={row["archive"]: select(row, ("sha256", "size")) for row in rows},
    )
    write(args.output / "MANIFEST.json", manifest)
    write(args.output / "PROVENANCE.local.json", selection)
    if args.commit_root:
        args.commit_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output / "MANIFEST.json", args.commit_root / "MANIFEST.json")
        for row in rows:
            if row["committed"]:
                shutil.copyfile(args.output / row["archive"], args.commit_root / row["archive"])
    print(json.dumps({"staged": len(rows), "committed": len(chosen), "bytes": used}))


if __name__ == "__main__":
    main()
