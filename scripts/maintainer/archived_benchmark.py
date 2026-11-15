"""Explicit-archive readers for maintainer comparison tests only."""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

from scripts.maintainer.archive_root import ArchivedPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/simulator"))

benchmark = importlib.import_module("benchmark")
ROOT, benchmark_legacy_id, posture_evidence, read, sha256 = (
    benchmark.ROOT,
    benchmark.benchmark_legacy_id,
    benchmark.posture_evidence,
    benchmark.read,
    benchmark.sha256,
)

BUNDLE = ArchivedPath("")
RUNTIME = BUNDLE / "runtime"


def require_reference_eval():
    import tower_eval

    expected = ROOT / "src/tower-eval/src/tower_eval"
    if Path(tower_eval.__file__).resolve().parent != expected.resolve():
        raise ValueError(
            "Reference route requires checkout tower_eval; start a fresh process with --reference"
        )


def verify_bundle(bundle=BUNDLE):
    """Verify every shipped implementation and frozen scene document before use."""
    manifest = read(bundle / "SOURCE_MANIFEST.json")
    for relative, binding in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha256(path) != binding["sha256"]:
            raise ValueError(f"Changed benchmark source: {relative}")
    protocol = read(bundle / "protocol.json")
    index = read(bundle / "scene-index.json")["scenes"]
    pairs = {(row["task"], row["seed"]) for row in index}
    if len(index) != 600 or len(pairs) != 600:
        raise ValueError("Expected exactly 600 unique frozen task/seed pairs")
    for task, settings in protocol["tasks"].items():
        if sum(row["task"] == task for row in index) != settings["scene_count"]:
            raise ValueError(f"Scene count differs for {task}")
    return {
        "verified_files": len(manifest["files"]),
        "scenes": len(index),
        "tasks": len(protocol["tasks"]),
    }


def load_scorer() -> Any:
    """Keep the latest campaign scorer separate from older public diagnostics."""
    verify_bundle()
    directory = RUNTIME / "src/tower-eval/src/tower_eval/scoring"
    name = "_tower_benchmark_202611_scoring"
    if name not in sys.modules:
        spec = cast(
            Any,
            importlib.util.spec_from_file_location(
                name, directory / "__init__.py", submodule_search_locations=[str(directory)]
            ),
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return importlib.import_module(name + ".early_success")


def score_run(run, task):
    task = benchmark_legacy_id(task)
    protocol = read(BUNDLE / "protocol.json")
    if task not in protocol["tasks"]:
        raise ValueError(f"Unsupported benchmark task: {task}")
    scorer = load_scorer()
    trajectory = importlib.import_module(scorer.__package__ + ".offline.trajectory")
    recording = trajectory.load_recording(run, trajectory.DEFAULT_RULES)
    if recording.times[-1] > protocol["tasks"][task]["duration_s"] + 1e-8:
        raise ValueError("Recording exceeds this benchmark's task deadline")
    return scorer.score_recording(
        recording, task, success_rule=protocol["tasks"][task]["success_rule"]
    )


def preflight_request(path):
    """Reject incompatible or reused requests before starting Isaac or a policy."""
    request = read(path)
    output = Path(request["output"])
    if not output.is_absolute() or output.exists():
        raise ValueError(
            "A new absolute output directory is required; never reuse an archived trial"
        )
    verify_bundle()
    request["task"] = benchmark_legacy_id(request.get("task"))
    protocol = read(BUNDLE / "protocol.json")
    task = protocol["tasks"].get(request.get("task"))
    if task is None or request.get("command_replay"):
        raise ValueError("Expected a learned-policy request for a frozen task")
    if (
        request.get("duration") != task["duration_s"]
        or request.get("success_rule") != task["success_rule"]
    ):
        raise ValueError("Request does not use the frozen deadline/scorer")
    seeds = request["seeds"]
    if not 1 <= len(seeds) <= 16 or len(set(seeds)) != len(seeds):
        raise ValueError("Expected 1..16 unique seeds")
    index = {(r["task"], r["seed"]): r for r in read(BUNDLE / "scene-index.json")["scenes"]}
    if any((request["task"], seed) not in index for seed in seeds):
        raise ValueError("Request contains a scene outside the frozen suite")
    for field in ("protocol", "runtime_manifest", "ik_bounds"):
        if sha256(request[field]) != request[field + "_sha256"]:
            raise ValueError(f"Changed request binding: {field}")
    if sha256(request["ik_bounds"]) != sha256(RUNTIME / "shared/shared-ik-spec.json"):
        raise ValueError("Request changes the shared IK bounds")
    settings = read(request["protocol"])
    for key, value in protocol["policy"].items():
        actual = settings["policy"].get("hz" if key == "control_hz" else key)
        if actual != value:
            raise ValueError(f"Policy contract differs: {key}")
    scenes = read(request["manifest"])["scenes"]
    if len(scenes) != len(seeds):
        raise ValueError("Scene manifest length differs from seed selection")
    portable = request.get("portable_assets")
    if portable:
        benchmark_assets = importlib.import_module("benchmark_assets")

        body = read(request["manifest"])
        if Path(body["assets"]).resolve() != Path(portable).resolve():
            raise ValueError("Scene manifest uses a different asset root")
        rows = benchmark_assets.load_rows(request["manifest"])
        if [(r["manifest_entry"]["task"], r["candidate"]["seed"]) for r in rows] != [
            (request["task"], seed) for seed in seeds
        ]:
            raise ValueError("Portable scene selection differs from request")
        if not Path(request["ik_python"]).is_file():
            raise ValueError("Missing independent IK Python environment")
    for entry, seed in [] if portable else zip(scenes, seeds, strict=True):
        expected = index[(request["task"], seed)]
        candidate = read(entry["candidate"])
        if candidate["seed"] != seed or sha256(entry["candidate"]) != expected["candidate_sha256"]:
            raise ValueError("Candidate differs from the archived frozen scene")
        initial = Path(entry["plan"]) / "reference-initial-state.json"
        if sha256(initial) != expected["initial_state_sha256"]:
            raise ValueError("Initial robot state differs from the frozen scene")
    files = read(request["runtime_manifest"])["files"]
    # Every shipped executable must be bound under its actual checkout path.
    # Historical source hashes cannot silently qualify a different local copy.
    for relative, binding in read(BUNDLE / "SOURCE_MANIFEST.json")["files"].items():
        local = (BUNDLE / relative).resolve()
        if relative.startswith("runtime/") and files.get(str(local)) != binding["sha256"]:
            raise ValueError(f"Request must bind this runtime checkout: {relative}")
    if portable:
        for row in benchmark_assets.manifest(portable)["files"]:
            asset = benchmark_assets.safe_path(portable, row["path"])
            if files.get(str(asset)) != row["sha256"]:
                raise ValueError(f"Unbound public asset: {row['path']}")
        review_binding = settings["pilot_review"]
        if sha256(review_binding["path"]) != review_binding["sha256"]:
            raise ValueError("Changed pilot review")
        evidence = posture_evidence(
            read(review_binding["path"]),
            request["task"],
            read(request["package"])["package_digest"],
        )
        for path in evidence:
            if files.get(str(path.resolve())) != sha256(path):
                raise ValueError("Posture evidence is not bound in the runtime manifest")
    for filename, digest in files.items():
        if sha256(filename) != digest:
            raise ValueError(f"Missing or changed external source/asset: {filename}")
    for field in ("capsule", "live_tools", "startup_environment", "template_runtime", "package"):
        if not Path(request[field]).is_absolute() or not Path(request[field]).exists():
            raise ValueError(f"Missing staged dependency: {field}")
    return request
