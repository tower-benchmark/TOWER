"""I/O adapters for characterization; never imported by production code."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import lzma
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import h5py
import numpy as np

from scripts.maintainer.archive_root import ArchivedPath
from scripts.maintainer.fetch_golden import golden_archive as golden_archive

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ArchivedPath("")
RUNTIME = BUNDLE / "runtime"
GOLDEN = ROOT / "tests/golden/benchmark-2026.11"
sys.path.insert(0, str(ROOT / "tools/simulator"))
benchmark: Any = importlib.import_module("benchmark")


def reference_bundle(reference_root: Path | None = None) -> Path:
    from scripts.maintainer.archive_root import archive_bundle

    return archive_bundle(reference_root)


def reference_scorer(reference_root: Path | None = None) -> Any:
    """Load verified frozen code without borrowing candidate module globals.

    This does not use benchmark.load_scorer or require the in-tree runtime.
    Separate root-derived namespaces allow both reference routes in one process.
    """
    bundle = reference_bundle(reference_root)
    manifest = bundle / "SOURCE_MANIFEST.json"
    pinned = read(Path(__file__).with_name("gpu_reference_contract.json"))["reference_identity"]
    if sha(manifest) != pinned["source_manifest_sha256"]:
        raise ValueError("Reference SOURCE_MANIFEST identity differs from benchmark-2026.11.0")
    for relative, binding in read(manifest)["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path) != binding["sha256"]:
            raise ValueError(f"Changed reference source: {relative}")
    directory = bundle / "runtime/src/tower-eval/src/tower_eval/scoring"
    name = "_tower_reference_" + hashlib.sha256(str(bundle.resolve()).encode()).hexdigest()
    if name in sys.modules:
        return importlib.import_module(name + ".early_success")
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return importlib.import_module(name + ".early_success")
    except BaseException:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                sys.modules.pop(key, None)
        raise


def read(path: Path | ArchivedPath) -> Any:
    if path.name.endswith(".json.xz"):
        return json.loads(lzma.decompress(path.read_bytes()))
    return json.loads(path.read_text())


def json_value(value: Any) -> Any:
    """Use the scorer's persisted JSON types (e.g. Rules tuples become arrays)."""
    return json.loads(json.dumps(value, allow_nan=False))


def write(path: Path, value: Any) -> None:
    if path.name.endswith(".json.xz"):
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        path.write_bytes(lzma.compress(payload.encode(), preset=9))
        return
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def sha(path: Path | ArchivedPath) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(path: Path, destination: Path) -> None:
    """Accept only regular, relative files; no links or destination traversal."""
    with tarfile.open(path, "r:xz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not member.isfile() or not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe golden archive member")
            stream = archive.extractfile(member)
            assert stream is not None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stream.read())


def arrays(run: Path) -> dict[str, Any]:
    with h5py.File(run / "episode/states.hdf5", "r") as stream:
        data: dict[str, Any] = {
            key: cast(h5py.Dataset, stream[key])[:]
            for key in ("time", "blocks", "block_velocities", "qpos")
        }
    for name, key, field in (
        ("finger-link-poses.npz", "finger_poses", "poses_m"),
        ("finger-feedback.npz", "finger_forces", "forces_n"),
    ):
        with np.load(run / name, allow_pickle=False) as stream:
            np.testing.assert_array_equal(stream["time"], data["time"])
            data[key] = stream[field]
    return data


def campaign_recording(run: Path, task: str, *, reference_root: Path | None = None) -> Any:
    """Exactly v4_evaluation.finalize_trial's constructor and status mapping."""
    scorer = reference_scorer(reference_root)
    data = arrays(run)
    rows = [
        {key: value[index] for key, value in data.items()} for index in range(len(data["time"]))
    ]
    recording = scorer.from_physical_rows(
        read(run / "episode_spec.json"), rows, read(run / "finger_geometry.json")
    )
    status = read(run / "status.json")
    if status["status"] == "task_failure" and task != "task1-2":
        recording.status["status"] = "task_failure"
    return recording


def campaign_score(run: Path, task: str, *, reference_root: Path | None = None) -> Any:
    scorer = reference_scorer(reference_root)
    rule = read(reference_bundle(reference_root) / "protocol.json")["tasks"][task]["success_rule"]
    return json_value(
        scorer.score_recording(
            campaign_recording(run, task, reference_root=reference_root), task, success_rule=rule
        )
    )


def package_score(run: Path, task: str) -> Any:
    scorer = importlib.import_module("tower_eval.scoring.offline.trajectory")
    # Both dataclasses expose the same inputs. No fork rules are passed in.
    from tower_eval.scoring.recording import from_campaign_files

    record = from_campaign_files(run, task)
    try:
        return json_value(scorer.score_recording(scorer.Recording(**vars(record)), task))
    except ValueError as error:
        # The package's Task5 destination lookup rejects the archived source
        # geometry. Preserve that current observable behavior as a snapshot.
        if task not in ("task5-2", "task5-3"):
            raise
        return {"exception": type(error).__name__, "message": str(error)}


def online_score(run: Path) -> Any:
    from tower_eval.scoring import StackScorer

    data, spec = arrays(run), read(run / "episode_spec.json")
    scene = SimpleNamespace(
        **{
            **spec,
            "blocks": [SimpleNamespace(**b) for b in spec["blocks"]],
            "transfers": [SimpleNamespace(**t) for t in spec["transfers"]],
        }
    )
    try:
        scorer = StackScorer(scene)
    except ValueError as error:
        # Task4/5/6 topology is deliberately unsupported by this old scorer.
        return {"constructor_error": str(error)}
    events = []
    for index, stamp in enumerate(data["time"]):
        result = scorer.update(
            SimpleNamespace(
                time=float(stamp),
                blocks=data["blocks"][index],
                block_velocities=data["block_velocities"][index],
                qpos=data["qpos"][index],
            )
        )
        # Freeze each update before the next call; never retain mutable scorer state.
        events.append(json_value(result))
    result = {
        "updates": len(data["time"]),
        "events": events,
        "final": scorer.result,
        "protocol": scorer.protocol,
    }
    # StackScorer uses integer block-index mapping keys. The snapshot is JSON,
    # whose object keys are strings; serialize once just as the snapshot writer
    # does. This preserves every numeric value and all array order.
    return json.loads(json.dumps(result, allow_nan=False))


def online_stream_sha256(events: list[dict[str, Any]]) -> str:
    """Reproduce the original newline-delimited update digest, in the exact tier.

    StackScorer's layers/support mappings use integer block IDs. Restore those
    two key types before sort_keys=True so IDs 10+ retain the original numeric
    ordering. Stored JSON still uses normal string keys throughout.
    """
    digest = hashlib.sha256()
    for event in events:
        original = {
            **event,
            **{name: {int(k): v for k, v in event[name].items()} for name in ("layers", "support")},
        }
        digest.update(
            json.dumps(original, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def online_snapshot(trial_id: str, *, fixture_root: Path = GOLDEN) -> Any:
    compressed = fixture_root / (trial_id + ".online.json.xz")
    return read(compressed if compressed.exists() else fixture_root / (trial_id + ".online.json"))


def candidate_score(run: Path, task: str, *, implementation: str = "bench_2026_11") -> Any:
    """Explicit candidate route, always constructed independently of frozen code."""
    if implementation != "bench_2026_11":
        raise ValueError(f"Unsupported candidate scorer: {implementation}")
    return library_campaign_score(run, task)


def library_campaign_score(run: Path, task: str) -> Any:
    """New library path; never constructs inputs using the frozen scorer."""
    from tower_eval.scoring.api import score_recording
    from tower_eval.scoring.recording import from_campaign_files

    return json_value(
        score_recording(from_campaign_files(run, task), task, ruleset="bench-2026.11")
    )
