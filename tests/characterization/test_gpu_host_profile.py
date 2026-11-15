"""Host provenance cannot weaken physics, data, trace or acceptance checks."""

import copy
import json
import os
import sys
from types import SimpleNamespace as NS

import pytest

from scripts.maintainer import gpu_baseline, gpu_identity, kit_safety
from scripts.maintainer import gpu_host_profile as host
from scripts.maintainer.golden_support import read, write
from tests.characterization.test_comparison_routes import candidate_identity
from tests.characterization.test_gpu_compare import artifact


def frozen(root, monkeypatch):
    monkeypatch.setenv("TOWER_GPU_REFERENCE_PROFILE", host.NAME)
    monkeypatch.setattr(host, "runtime_profile", lambda gpu: dict(host.PROFILE))
    document = host.frozen_summary(read(root / "summary.json"), 0)
    write(root / "summary.json", document)
    return document


def test_profile_is_explicit_and_does_not_change_exact_contract(monkeypatch):
    original = gpu_identity.reference_contract()
    monkeypatch.delenv("TOWER_GPU_REFERENCE_PROFILE", raising=False)
    assert host.reference_contract() == original
    monkeypatch.setenv("TOWER_GPU_REFERENCE_PROFILE", host.NAME)
    measured = host.reference_contract()
    assert measured["contract"]["hardware_runtime_profile"] == host.PROFILE
    assert gpu_identity.reference_contract() == original
    monkeypatch.setenv("TOWER_GPU_REFERENCE_PROFILE", "arbitrary")
    with pytest.raises(ValueError, match="Unknown"):
        host.reference_contract()


def test_jenga_acceptance_and_cross_gpu_informational_only(tmp_path, monkeypatch):
    for name in ("p0", "j", "candidate"):
        artifact(tmp_path / name)
    j = frozen(tmp_path / "j", monkeypatch)
    row = copy.deepcopy(j["trials"][0])
    row.pop("source_manifest_sha256")
    document = host.candidate_summary(
        candidate_identity(), host.reference_contract()["contract"], [row]
    )
    for path in (tmp_path / "candidate/summary.json", tmp_path / "candidate/task1-2/summary.json"):
        write(path, document)
    result = gpu_baseline.compare(
        tmp_path / "j", tmp_path / "candidate", required_tasks=["task1-2"]
    )
    assert result["passed"] and result["acceptance_eligible"]
    with pytest.raises(ValueError, match="hardware_runtime_profile"):
        gpu_baseline.compare(tmp_path / "p0", tmp_path / "j", required_tasks=["task1-2"])
    result = gpu_baseline.compare(
        tmp_path / "p0", tmp_path / "j", required_tasks=["task1-2"], informational_cross_gpu=True
    )
    assert result["passed"] and not result["acceptance_eligible"]
    assert result["informational_cross_gpu"]


@pytest.mark.parametrize("field", ["driver", "python", "libc", "gpu_model"])
def test_measured_environment_is_checked(field, monkeypatch):
    monkeypatch.setenv("TOWER_GPU_REFERENCE_PROFILE", host.NAME)
    monkeypatch.setattr(host, "runtime_profile", lambda gpu: dict(host.PROFILE, **{field: "wrong"}))
    with pytest.raises(ValueError, match=field):
        host.verify_runtime(0)


@pytest.mark.parametrize(
    "section,field",
    [("contract", "physics"), ("contract", "immutable_data"), ("reference_identity", "experts")],
)
def test_profile_does_not_admit_other_contract_changes(section, field, monkeypatch):
    monkeypatch.setenv("TOWER_GPU_REFERENCE_PROFILE", host.NAME)
    document = host.candidate_summary(
        candidate_identity(), host.reference_contract()["contract"], []
    )
    document[section][field] = {}
    with pytest.raises(ValueError, match=field):
        host.read_summary(document)


def test_cross_profile_keeps_numeric_thresholds(tmp_path, monkeypatch):
    artifact(tmp_path / "a")
    artifact(tmp_path / "b", joint=1.01e-4)
    frozen(tmp_path / "b", monkeypatch)
    result = gpu_baseline.compare(
        tmp_path / "a", tmp_path / "b", required_tasks=["task1-2"], informational_cross_gpu=True
    )
    assert not result["passed"] and not result["acceptance_eligible"]
    assert result["tolerances"] == dict(
        qpos_rad=1e-4, block_position_m=1e-4, block_orientation_deg=0.05, image_psnr_db=35.0
    )


def test_scene_revision_preserves_host_and_both_data_identities(tmp_path, monkeypatch):
    from scripts.maintainer.scene_equivalence import expected_receipt

    for name in ("old", "new"):
        artifact(tmp_path / name)
    old = frozen(tmp_path / "old", monkeypatch)
    row = copy.deepcopy(old["trials"][0])
    row.pop("source_manifest_sha256")
    identity = dict(candidate_identity(), data_version="benchmark-2026.11.1")
    contract = dict(
        host.reference_contract()["contract"],
        benchmark="benchmark-2026.11.1",
        data_version="benchmark-2026.11.1",
        scene_equivalence=expected_receipt(),
    )
    document = host.candidate_summary(identity, contract, [row])
    for path in (tmp_path / "new/summary.json", tmp_path / "new/task1-2/summary.json"):
        write(path, document)
    result = gpu_baseline.compare(tmp_path / "old", tmp_path / "new", required_tasks=["task1-2"])
    assert result["passed"] and result["acceptance_eligible"]
    assert result["contract"]["data_version"] == "benchmark-2026.11.0"
    assert result["candidate_contract"] == contract
    assert host.read_summary(document)["contract"] == contract
    for field in ("scene_equivalence", "physics", "hardware_runtime_profile"):
        changed = copy.deepcopy(document)
        changed["contract"][field] = {}
        with pytest.raises(ValueError):
            host.read_summary(changed)


def test_kit_safety_is_scoped_and_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["worker"])
    root = tmp_path / "private-kit"
    for key in (
        "OMNI_CONFIG_PATH",
        "OMNI_CRASHREPORTER_ENABLED",
        "OMNI_CRASHREPORTER_SKIPOLDDUMPUPLOAD",
        "OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA",
    ):
        monkeypatch.delenv(key, raising=False)
    kit_safety.configure(root)
    assert "--/crashreporter/enabled=false" in sys.argv
    assert "--/crashreporter/devOnlyOverridePrivacyAndForceUpload=false" in sys.argv
    assert "--portable-root" in sys.argv
    assert os.environ["OMNI_CRASHREPORTER_ENABLED"] == "0"
    assert os.environ["OMNI_CONFIG_PATH"] == str(root / "config")
    assert "performance = false" in (root / "config/privacy.toml").read_text()
    values: dict[str, object] = dict(
        kit_safety.SETTINGS, **{"/crashreporter/dumpDir": str(root / "dumps")}
    )
    module = NS(get_settings=lambda: NS(get=values.get))
    monkeypatch.setitem(sys.modules, "carb.settings", module)
    kit_safety.verify(root)
    assert json.loads((root / "verified.json").read_text())["verified"]
    values["/crashreporter/enabled"] = True
    with pytest.raises(RuntimeError, match="suppression differs"):
        kit_safety.verify(root)
