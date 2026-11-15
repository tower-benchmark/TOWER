# SPDX-License-Identifier: Apache-2.0
"""Download and resolve the exact November benchmark assets from a pinned public dataset."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

import benchmark
from tower_sim.backend.resources import DATA
from tower_sim.bench.asset_identity import station_identity
from tower_sim.bench.download import download_file
from tower_sim.bench.scene_format import DEFAULT_BENCHMARK

LOCK = DATA / "benchmark/assets.lock.json"
SCENE_INDEX_SHA256 = "0bad41c69640cd51b702fde8527e3b354c7c619743f18d66b080f929a7d46cc3"


def safe_path(root, name):
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe release path: {name}")
    target = (Path(root) / name).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Release path escapes destination: {name}")
    return target


def manifest(root):
    body = benchmark.read(Path(root) / "manifest.json")
    if body.get("schema") != "tower-benchmark-assets-v1":
        raise ValueError("Unsupported benchmark asset manifest")
    names = [row["path"] for row in body["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate release paths")
    for name in names:
        safe_path(root, name)
    return body


def verify(root, *, check_lock=True):
    root = Path(root).resolve()
    if check_lock:
        lock = benchmark.read(LOCK)
        if benchmark.sha256(root / "manifest.json") != lock["manifest_sha256"]:
            raise ValueError("Asset manifest differs from the pinned release")
    body = manifest(root)
    if body["scene_index_sha256"] != SCENE_INDEX_SHA256:
        raise ValueError("Asset release belongs to a different frozen scene suite")
    for row in body["files"]:
        path = safe_path(root, row["path"])
        if path.stat().st_size != row["size"] or benchmark.sha256(path) != row["sha256"]:
            raise ValueError(f"Changed asset: {row['path']}")
    station = root / "station/station-dual-j2-wrist-shared-v3.usdz"
    expected = benchmark.read(DATA / "benchmark/runtime/ik-capsule/runtime/scene-cache.json")[
        "scene_sha256"
    ]
    if station_identity(benchmark.sha256(station)) != expected:
        raise ValueError("Station differs from the campaign calibration")
    return {
        "files": len(body["files"]),
        "bytes": sum(r["size"] for r in body["files"]),
        "release": body["release"],
        "station_sha256": expected,
    }


def download(destination, *, data_root=None):
    """Unauthenticated, pinned HTTPS downloads; incomplete files never become assets."""
    lock = benchmark.read(LOCK)
    revision = lock["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Release must be pinned to a Hugging Face commit")
    from tower_sim.bench import data_release

    if data_root is None:
        data_release.verify_locator(
            benchmark.read(benchmark.ROOT / "benchmarks/2026.11.1/assets.lock.json"),
            require_pin=True,
        )
    else:
        data_release.verify_release(data_root)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = lock["directory"].strip("/")
    prefix = "" if directory in ("", ".") else quote(directory) + "/"
    base = f"https://huggingface.co/datasets/{lock['repository']}/resolve/{revision}/{prefix}"

    def fetch(name, expected, size=None):
        download_file(
            base + quote(name), safe_path(root, name), expected, size=size, opener=urlopen
        )

    fetch("manifest.json", lock["manifest_sha256"])
    body = manifest(root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda row: fetch(row["path"], row["sha256"], row["size"]), body["files"]))
    receipt = verify(root)
    receipt["data_files"] = len(data_release.download(benchmark.ROOT, root, data_root=data_root))
    return receipt


def frozen_scene(task, seed, *, v1_root=None):
    from scripts.maintainer.archive_scene import v1_scene

    return v1_scene(benchmark.ROOT, task, seed, v1_root=v1_root)


def resolve_scene(
    task,
    seed,
    root,
    *,
    benchmark_version=DEFAULT_BENCHMARK,
    data_root=None,
    compatibility=False,
    v1_root=None,
):
    """Only relocate asset fields; poses, physics, tools and initial joints survive verbatim."""
    from tower_sim.bench.data_release import scene
    from tower_sim.bench.scene_format import resolve_physical, revision, scene_number

    revision(benchmark_version)
    body = manifest(root)
    if benchmark_version == "2026.11.0":
        from scripts.maintainer.archive_scene import relocate_v1

        return relocate_v1(frozen_scene(task, seed, v1_root=v1_root), root, body)
    if isinstance(seed, str):
        scene_number(seed)
        scene_id = seed
    elif type(seed) is int and 0 <= seed < 50:
        scene_id = f"scene-{seed:03d}"
    else:
        from scripts.maintainer.archive_scene import public_scene_id

        scene_id = public_scene_id(benchmark.ROOT, task, seed, v1_root=v1_root)
    encoded = scene(task, scene_id, assets=root, data_root=data_root)
    if compatibility:
        from scripts.maintainer.archive_scene import archived_scene, compatibility_view

        original = archived_scene(benchmark.ROOT, task, scene_id, v1_root=v1_root)
        return compatibility_view(encoded, original, root, body)
    return resolve_physical(encoded, root, body, selector=seed if type(seed) is int else None)


def prepare_scenes(
    task,
    seeds,
    assets,
    destination,
    *,
    reference=False,
    benchmark_version=None,
    data_root=None,
    v1_root=None,
):
    benchmark_version = benchmark_version or ("2026.11.0" if reference else "2026.11.1")
    if reference:
        if benchmark_version != "2026.11.0":
            raise ValueError("Frozen runtime only accepts v1 scenes")
        benchmark.verify_bundle()
    else:
        from tower_sim.bench.verification import verify_data

        verify_data(
            benchmark.ROOT,
            benchmark=benchmark_version,
            assets=assets,
            data_root=data_root,
            v1_root=v1_root,
        )
    verify(assets)
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scenes = []
    for seed in seeds:
        row = resolve_scene(
            task,
            seed,
            assets,
            benchmark_version=benchmark_version,
            data_root=data_root,
            v1_root=v1_root,
        )
        directory = output / f"seed-{seed}"
        directory.mkdir()
        for name, body in (
            ("candidate.json", row["candidate"]),
            ("reference-initial-state.json", row["initial_state"]),
        ):
            (directory / name).write_text(json.dumps(body, indent=2) + "\n")
        scenes.append(
            {
                "candidate": str(directory / "candidate.json"),
                "plan": str(directory),
                "task": task,
                "seed": seed,
            }
        )
    path = output / "scenes.json"
    path.write_text(
        json.dumps(
            {
                "schema": "tower-frozen-initial-states-v1",
                "assets": str(Path(assets).resolve()),
                "benchmark": benchmark_version,
                "data_root": str(Path(data_root).resolve()) if data_root is not None else None,
                "v1_root": str(Path(v1_root).resolve()) if v1_root is not None else None,
                "scenes": scenes,
            },
            indent=2,
        )
        + "\n"
    )
    return path


def load_rows(path):
    """Bind initial states to the shipped scenes without loading any expert motion."""
    import numpy as np

    body = benchmark.read(path)
    if body.get("schema") != "tower-frozen-initial-states-v1":
        raise ValueError("Expected the public frozen-scene manifest")
    if "benchmark" not in body:
        raise ValueError("Scene manifest must declare its benchmark revision")
    verify(body["assets"])
    entries = body["scenes"]
    pairs = [(e["task"], e["seed"]) for e in entries]
    if not 1 <= len(entries) <= 16 or len(set(pairs)) != len(entries):
        raise ValueError("Expected 1..16 distinct frozen scenes")
    rows = []
    for index, entry in enumerate(entries):
        expected = resolve_scene(
            entry["task"],
            entry["seed"],
            body["assets"],
            benchmark_version=body["benchmark"],
            data_root=body.get("data_root"),
            v1_root=body.get("v1_root"),
        )
        candidate = benchmark.read(entry["candidate"])
        plan = Path(entry["plan"])
        initial = benchmark.read(plan / "reference-initial-state.json")
        if candidate != expected["candidate"] or initial != expected["initial_state"]:
            raise ValueError("Portable candidate/initial state differs from the frozen source")
        q = np.asarray(initial["qpos"], float)
        rows.append(
            dict(
                env_id=index,
                candidate=candidate,
                candidate_path=Path(entry["candidate"]),
                plan=plan,
                initial=initial,
                receipt={"initial_state_only": True},
                times=np.array([0.0, 0.05]),
                actions=np.stack([q, q]),
                ticks=20,
                parameters={},
                manifest_entry=entry,
                portable_initial_state_only=True,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--assets", type=Path, required=True)
        command.add_argument("--data-root", type=Path)
    scene = sub.add_parser("prepare-scenes")
    scene.add_argument("--benchmark", choices=("2026.11.1",), default="2026.11.1")
    scene.add_argument("--data-root", type=Path)
    scene.add_argument("--v1-root", type=Path, help="Explicit private v1 archive export")
    scene.add_argument("--assets", type=Path, required=True)
    scene.add_argument("--task", required=True, type=benchmark.task_argument)
    scene.add_argument("--seeds", type=int, nargs="+", required=True)
    scene.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download(args.assets, data_root=args.data_root), indent=2))
    elif args.command == "verify":
        from tower_sim.bench import data_release

        result = verify(args.assets)
        if args.data_root is not None:
            result["data_files"] = len(data_release.verify_release(args.data_root))
        print(json.dumps(result, indent=2))
    else:
        print(
            prepare_scenes(
                args.task,
                args.seeds,
                args.assets,
                args.output,
                reference=False,
                benchmark_version=args.benchmark,
                data_root=args.data_root,
                v1_root=args.v1_root,
            )
        )


if __name__ == "__main__":
    main()
