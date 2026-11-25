"""Cross-revision CPU gates; synthetic artifacts live only in pytest temp dirs."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.maintainer import golden_support as support
from scripts.maintainer import gpu_baseline, gpu_runner
from scripts.maintainer.gpu_identity import candidate_summary, read_summary, reference_contract
from tests.characterization.golden_compare import assert_equivalent
from tests.characterization.test_gpu_compare import artifact, compare


def candidate_identity() -> dict[str, Any]:
    return dict(
        schema="tower-gpu-candidate-v2",
        implementation="tower-sim",
        benchmark_version="2026.11.0",
        commit="a" * 40,
        wheel_sha256={"tower_sim.whl": "b" * 64},
        dependency_sha256={"numpy.whl": "c" * 64},
        data_version="benchmark-2026.11.0",
        frozen_components=[],
    )


def upgrade(root: Path) -> dict[str, Any]:
    row = support.read(root / "summary.json")["trials"][0]
    del row["source_manifest_sha256"]
    result = candidate_summary(candidate_identity(), reference_contract()["contract"], [row])
    support.write(root / "summary.json", result)
    support.write(root / "task1-2/summary.json", result)
    return result


def test_legacy_and_new_candidate_are_comparable(tmp_path: Path) -> None:
    artifact(tmp_path / "a")
    artifact(tmp_path / "b")
    before = (tmp_path / "a/summary.json").read_bytes()
    upgraded = upgrade(tmp_path / "b")
    result = compare(tmp_path / "a", tmp_path / "b")
    assert result["passed"] and result["acceptance_eligible"]
    assert result["baseline_identity"]["schema"] == "tower-gpu-legacy-v1-identity"
    assert result["baseline_identity"]["commit"] is None
    assert result["candidate_identity"] == upgraded["candidate_identity"]
    assert result["reference_identity"] == reference_contract()["reference_identity"]
    assert (tmp_path / "a/summary.json").read_bytes() == before
    # Different candidate code is allowed; reference/data/physics equality still holds.
    upgraded["candidate_identity"]["commit"] = "d" * 40
    upgraded["candidate_identity"]["wheel_sha256"]["tower_sim.whl"] = "e" * 64
    support.write(tmp_path / "b/summary.json", upgraded)
    support.write(tmp_path / "b/task1-2/summary.json", upgraded)
    assert compare(tmp_path / "a", tmp_path / "b")["passed"]


@pytest.mark.parametrize(
    "path,value,error",
    [
        (
            ("contract", "immutable_data", "assets_manifest_sha256"),
            "0" * 64,
            "immutable_data/assets_manifest_sha256",
        ),
        (("contract", "physics", "simulation_hz"), 120, "physics/simulation_hz"),
        (
            ("contract", "hardware_runtime_profile", "driver"),
            "different",
            "hardware_runtime_profile/driver",
        ),
        (
            ("reference_identity", "source_manifest_sha256"),
            "0" * 64,
            "reference_identity/source_manifest_sha256",
        ),
        (
            ("reference_identity", "corpus_manifest_sha256"),
            "0" * 64,
            "reference_identity/corpus_manifest_sha256",
        ),
        (("candidate_identity", "commit"), "unknown", "candidate identity: commit"),
        (("candidate_identity", "wheel_sha256"), {}, "candidate identity: wheel_sha256"),
        (("candidate_identity", "dependency_sha256"), {}, "candidate identity: dependency_sha256"),
        (("candidate_identity", "data_version"), "other", "candidate identity: data_version"),
        (
            ("candidate_identity", "frozen_components"),
            ["unknown"],
            "candidate identity: frozen_components",
        ),
    ],
)
def test_changed_identity_or_data_rejected(
    tmp_path: Path, path: tuple[str, ...], value: Any, error: str
) -> None:
    artifact(tmp_path / "a")
    artifact(tmp_path / "b")
    document = upgrade(tmp_path / "b")
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    support.write(tmp_path / "b/summary.json", document)
    with pytest.raises(ValueError, match=error):
        compare(tmp_path / "a", tmp_path / "b")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("source_manifest_sha256", "0" * 64, "Legacy v1 source_manifest_sha256"),
        ("scene_id", "scene-049", "task1-2/scene_id"),
        ("expert_sha256", "0" * 64, "task1-2/expert_sha256"),
        ("samples", 3, "task1-2/samples"),
        ("duration_s", 1.0, "task1-2/duration_s"),
        ("task_success", False, "task1-2/task_success"),
        ("policy_action_failure", True, "task1-2/policy_action_failure"),
        ("replay_completed", False, "incomplete replay"),
    ],
)
def test_changed_legacy_trial_rejected(tmp_path: Path, field: str, value: Any, error: str) -> None:
    artifact(tmp_path / "a")
    artifact(tmp_path / "b")
    document = support.read(tmp_path / "b/summary.json")
    document["trials"][0][field] = value
    support.write(tmp_path / "b/summary.json", document)
    with pytest.raises(ValueError, match=error):
        compare(tmp_path / "a", tmp_path / "b")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("time", [0.0, 0.0], "Invalid trace time"),
        ("time", [0.0, float("nan")], "Invalid trace time"),
        ("qpos", np.zeros((2, 15)), "qpos"),
        ("qpos", np.full((2, 16), float("nan")), "qpos"),
        ("blocks", np.zeros((2, 1, 7)), "zero quaternion"),
        ("blocks", np.zeros((1, 1, 7)), "blocks"),
    ],
)
def test_rehashed_malformed_trace_rejected(
    tmp_path: Path, field: str, value: Any, error: str
) -> None:
    artifact(tmp_path / "a")
    artifact(tmp_path / "b")
    path = tmp_path / "b/task1-2/trace.npz"
    with np.load(path) as stream:
        arrays = dict(stream)
    arrays[field] = value
    np.savez_compressed(path, **arrays)
    document = support.read(tmp_path / "b/summary.json")
    document["trials"][0]["trace_sha256"] = support.sha(path)
    support.write(tmp_path / "b/summary.json", document)
    with pytest.raises(ValueError, match=error):
        compare(tmp_path / "a", tmp_path / "b")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("samples", 3, "Trace sample count"),
        ("duration_s", 0.2, "Trace duration"),
    ],
)
def test_matching_but_false_summaries_rejected(
    tmp_path: Path, field: str, value: Any, error: str
) -> None:
    for name in ("a", "b"):
        artifact(tmp_path / name)
        path = tmp_path / name / "summary.json"
        document = support.read(path)
        document["trials"][0][field] = value
        support.write(path, document)
    with pytest.raises(ValueError, match=error):
        compare(tmp_path / "a", tmp_path / "b")


def test_committed_v1_receipt_readable_without_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_baseline, "BUNDLE", Path("/does-not-exist"))
    document = support.read(support.GOLDEN / "gpu-baseline-summary.json")
    for repeat, key in ((False, "baseline_runs"), (True, "repeat_runs")):
        normalized = read_summary(document, repeat=repeat)
        assert normalized["trials"] == document[key]
        assert len(normalized["trials"]) == 12


def test_candidate_invalid_assets_cli_does_not_create_output(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(support.ROOT / "scripts/maintainer/gpu_baseline.py"),
            "run",
            "--implementation",
            "tower-sim",
            "--assets",
            str(tmp_path),
            "--expert-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "manifest.json" in result.stderr
    assert "P7-08 must provide" not in result.stderr
    assert not (tmp_path / "out").exists()


def test_runner_dependency_error_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        raise ModuleNotFoundError("broken dependency", name="real_dependency")

    monkeypatch.setattr(gpu_runner.importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match="broken dependency"):
        gpu_runner.resolve_candidate()


@pytest.mark.archive
def test_candidate_resume_binds_code_and_substitutions(tmp_path: Path) -> None:
    artifact(tmp_path)
    document = upgrade(tmp_path)
    path = tmp_path / "task1-2/summary.json"
    support.write(path, document)
    binding = {key: document[key] for key in ("candidate_identity", "contract")}
    assert gpu_baseline.resume_summary(path, binding)["task"] == "task1-2"
    for field, value in (("commit", "d" * 40), ("frozen_components", ["cameras"])):
        changed = copy.deepcopy(binding)
        changed["candidate_identity"][field] = value
        with pytest.raises(ValueError, match=f"resume/candidate_identity/{field}"):
            gpu_baseline.resume_summary(path, changed)
    with pytest.raises(ValueError, match="changed frozen"):
        gpu_baseline.resume_summary(path, None)


def test_candidate_worker_protocol_and_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fake only the future runner; exercise the harness's real artifact lifecycle."""
    source = tmp_path / "expert.npz"
    np.savez_compressed(source, time_s=[0.0, 0.1], action_qpos=np.zeros((2, 16)))
    pinned = reference_contract()
    expert = pinned["reference_identity"]["experts"]["task1-2"]
    expert["sha256"] = support.sha(source)
    from scripts.maintainer import gpu_identity

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "manifest.json").write_text("{}")
    pinned["contract"]["immutable_data"]["assets_manifest_sha256"] = support.sha(
        assets / "manifest.json"
    )
    monkeypatch.setattr(gpu_identity, "reference_contract", lambda: pinned)
    monkeypatch.setattr(gpu_baseline, "reference_contract", lambda: pinned)
    support.write(
        tmp_path / "MANIFEST.json",
        dict(
            trials=[
                dict(
                    task="task1-2",
                    scene_id=expert["scene_id"],
                    file=source.name,
                    sha256=expert["sha256"],
                )
            ]
        ),
    )
    identity = candidate_identity()
    identity["frozen_components"] = ["cameras"]
    calls = []
    output = tmp_path / "out"

    class Prepared:
        def replay(self, commands: gpu_runner.CommandTrajectory) -> gpu_runner.ReplayEvidence:
            calls.append("replay")
            np.testing.assert_array_equal(commands.time_s, [0.0, 0.1])
            assert commands.action_qpos.shape == (2, 16)
            artifact(tmp_path / "fake")
            for path in (tmp_path / "fake/task1-2").iterdir():
                shutil.copy(path, output / path.name)
            return gpu_runner.ReplayEvidence(
                True,
                True,
                False,
                {name: "first.png" for name in ("left", "right", "front", "back")},
            )

        def close(self) -> None:
            assert (output / "summary.json").is_file()
            calls.append("close")

    class Runner:
        def identity(
            self, assets: Path, gpu: int, components: tuple[str, ...], reference: Path
        ) -> dict:
            assert components == ("cameras",) and reference == tmp_path
            return dict(candidate_identity=identity, contract=pinned["contract"])

        def setup(self, request: gpu_runner.ReplayRequest) -> Prepared:
            assert request.task == "task1-2" and request.seed == int(
                expert["scene_id"].removeprefix("scene-")
            )
            assert request.frozen_components == ("cameras",) and request.output == output
            calls.append("setup")
            return Prepared()

    original_import = gpu_runner.importlib.import_module
    monkeypatch.setattr(
        gpu_runner.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(create_runner=Runner)
            if name == "tower_sim.bench.gpu_baseline"
            else original_import(name)
        ),
    )
    args = SimpleNamespace(
        implementation="tower-sim",
        benchmark_version="2026.11.0",
        task="task1-2",
        assets=assets,
        output=output,
        gpu=0,
        expert_root=tmp_path,
        frozen_component=["cameras"],
        reference_root=tmp_path,
    )
    gpu_baseline.worker(args)
    assert calls == ["setup", "replay", "close"]
    assert read_summary(support.read(output / "summary.json"))["candidate_identity"] == identity
    # The harness checks the asset release itself instead of trusting the runner.
    (assets / "manifest.json").write_text('{"changed": true}')
    args.output = tmp_path / "out-changed"
    with pytest.raises(ValueError, match="Candidate assets differ: manifest.json"):
        gpu_baseline.worker(args)


def test_equal_sample_counts_require_equal_timestamps(tmp_path: Path) -> None:
    for name, middle in (("a", 0.04), ("b", 0.06)):
        artifact(tmp_path / name)
        trace = tmp_path / name / "task1-2/trace.npz"
        with np.load(trace) as stream:
            arrays = {key: np.repeat(stream[key][:1], 3, axis=0) for key in ("qpos", "blocks")}
        np.savez_compressed(
            trace, time=[0.0, middle, 0.1], qpos=arrays["qpos"], blocks=arrays["blocks"]
        )
        path = tmp_path / name / "summary.json"
        document = support.read(path)
        document["trials"][0].update(samples=3, trace_sha256=support.sha(trace))
        support.write(path, document)
    with pytest.raises(ValueError, match="Replay time differs: task1-2/time"):
        compare(tmp_path / "a", tmp_path / "b")


@pytest.fixture(scope="module")
def reference_root(tmp_path_factory: pytest.TempPathFactory):
    from scripts.maintainer.archive_root import archive_bundle

    root = tmp_path_factory.mktemp("reference") / "checkout"
    shutil.copytree(archive_bundle(), root / "simulator/benchmark-202611")
    return root


@pytest.mark.archive
def test_external_reference_without_in_tree_runtime(
    reference_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trial = support.read(support.GOLDEN / "MANIFEST.json")["trials"][0]
    support.unpack(support.golden_archive(trial["archive"]), tmp_path)
    expected = support.campaign_score(tmp_path, trial["task"])
    monkeypatch.setattr(support, "BUNDLE", tmp_path / "absent-runtime")
    monkeypatch.setattr(
        support.benchmark, "load_scorer", lambda: pytest.fail("in-tree loader used")
    )
    actual = support.campaign_score(tmp_path, trial["task"], reference_root=reference_root)
    assert_equivalent(expected, actual, portable=False)
    assert_equivalent(expected, support.candidate_score(tmp_path, trial["task"]))
    assert str(reference_root) in support.reference_scorer(reference_root).__file__
    assert "_tower_reference_" in support.reference_scorer(reference_root).__name__


@pytest.mark.archive
def test_reference_rejects_modified_sources(tmp_path: Path) -> None:
    bundle = tmp_path / "simulator/benchmark-202611"
    shutil.copytree(support.BUNDLE, bundle)
    (bundle / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed reference source: protocol.json"):
        support.reference_scorer(tmp_path)


@pytest.mark.archive
def test_contract_lineage() -> None:
    """The new document copies identities, never rewrites original data/receipts."""
    from scripts.maintainer.archive_root import archive_checkout

    archive = archive_checkout()
    contract = reference_contract()
    reference = contract["reference_identity"]
    assert reference["source_manifest_sha256"] == support.sha(
        support.BUNDLE / "SOURCE_MANIFEST.json"
    )
    assert reference["corpus_manifest_sha256"] == support.sha(
        archive / "tests/golden/benchmark-2026.11/MANIFEST.json"
    )
    receipt = support.read(support.GOLDEN / "gpu-baseline-summary.json")
    assert reference["hardware_runtime_profile"] == receipt["environment"]
    assert reference["experts"] == {
        row["task"]: dict(scene_id=row["scene_id"], sha256=row["expert_sha256"])
        for row in receipt["baseline_runs"]
    }
    data = contract["contract"]["immutable_data"]
    assert (
        data["assets_manifest_sha256"]
        == support.read(support.BUNDLE / "assets.lock.json")["manifest_sha256"]
    )
    for name in ("protocol", "scene-index"):
        assert data[name.replace("-", "_") + "_sha256"] == support.sha(
            support.BUNDLE / (name + ".json")
        )
    groups = {
        "scenes": [p for p in (support.BUNDLE / "scenes").rglob("*") if p.is_file()],
        "physics_resources": [
            p for p in support.RUNTIME.rglob("*") if p.suffix in (".json", ".urdf", ".yaml", ".yml")
        ]
        + list((archive / "src/tower-datagenerator/jenga_generator/data").glob("*.json")),
    }
    for name, paths in groups.items():
        mapping = {p.relative_to(archive).as_posix(): support.sha(p) for p in paths}
        digest = hashlib.sha256(
            json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert data[name + "_sha256"] == digest


def test_substituted_comparison_is_diagnostic_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact(tmp_path / "a")
    artifact(tmp_path / "b")
    upgraded = upgrade(tmp_path / "b")
    upgraded["candidate_identity"]["frozen_components"] = ["cameras"]
    support.write(tmp_path / "b/summary.json", upgraded)
    support.write(tmp_path / "b/task1-2/summary.json", upgraded)
    result = compare(tmp_path / "a", tmp_path / "b")
    assert result["passed"] and not result["acceptance_eligible"]
    # main() requires every protocol task; reuse the one-task result for the CLI gate.
    monkeypatch.setattr(gpu_baseline, "compare", lambda *args, **kwargs: result)
    base = ["gpu_baseline.py", "compare", str(tmp_path / "a"), str(tmp_path / "b")]
    monkeypatch.setattr(sys, "argv", [*base, "--output", str(tmp_path / "refused.json")])
    with pytest.raises(SystemExit) as refused:
        gpu_baseline.main()
    assert refused.value.code == 2 and not (tmp_path / "refused.json").exists()
    monkeypatch.setattr(
        sys, "argv", [*base, "--output", str(tmp_path / "diag.json"), "--diagnostic"]
    )
    gpu_baseline.main()
    assert support.read(tmp_path / "diag.json")["acceptance_eligible"] is False


@pytest.mark.parametrize("field,value", [("commit", "f" * 40), ("frozen_components", ["scoring"])])
def test_compare_rejects_worker_identity_disagreeing_with_root(tmp_path, field, value):
    artifact(tmp_path / "a")
    artifact(tmp_path / "b")
    worker = upgrade(tmp_path / "b")
    worker["candidate_identity"][field] = value
    support.write(tmp_path / "b/task1-2/summary.json", worker)
    with pytest.raises(ValueError, match=f"task1-2/candidate_identity/{field}"):
        compare(tmp_path / "a", tmp_path / "b")
