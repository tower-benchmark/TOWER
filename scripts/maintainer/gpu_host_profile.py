"""Explicit desktop reference GPU provenance; the P0 and exact-tier contract stays intact."""

from __future__ import annotations

import copy
import importlib.metadata
import os
import platform
import subprocess
from typing import Any

from scripts.maintainer import gpu_identity as original

NAME = "desktop-4090-reference"
PROFILE = {
    "isaac_sim": "6.1.0.0",
    "python": "3.12.13",
    "platform": "Linux x86_64",
    "libc": "glibc 2.35",
    "gpu_model": "NVIDIA GeForce RTX 4090",
    "driver": "580.159.03",
    "numpy": "2.3.1",
    "scipy": "1.17.0",
    "h5py": "3.15.1",
}


def selected() -> str | None:
    value = os.environ.get("TOWER_GPU_REFERENCE_PROFILE")
    if value not in (None, NAME):
        raise ValueError("Unknown explicit GPU reference profile")
    return value


def contract_for(name: str | None) -> dict[str, Any]:
    document = original.reference_contract()
    if name is None:
        return document
    if name != NAME:
        raise ValueError("Unknown recorded GPU reference profile")
    for section in ("reference_identity", "contract"):
        document[section]["hardware_runtime_profile"] = dict(PROFILE)
    return document


def reference_contract() -> dict[str, Any]:
    return contract_for(selected())


def runtime_profile(gpu: int) -> dict[str, Any]:
    name, driver = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    return dict(
        isaac_sim=importlib.metadata.version("isaacsim"),
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}",
        libc=" ".join(platform.libc_ver()),
        gpu_model=name,
        driver=driver,
        **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "h5py")},
    )


def verify_runtime(gpu: int) -> dict[str, Any]:
    actual = runtime_profile(gpu)
    original.require_equal(
        reference_contract()["contract"]["hardware_runtime_profile"],
        actual,
        "hardware_runtime_profile",
    )
    return actual


def frozen_summary(document: dict[str, Any], gpu: int) -> dict[str, Any]:
    if selected() is None:
        return document
    return dict(document, host_profile=selected(), hardware_runtime_profile=verify_runtime(gpu))


def read_summary(document: dict[str, Any], *, repeat: bool = False) -> dict[str, Any]:
    name = document.get("host_profile")
    if name is None:
        return original.read_summary(document, repeat=repeat)
    pinned = contract_for(name)
    adapted = copy.deepcopy(document)
    if document.get("schema") == "tower-gpu-baseline-v1":
        original.require_equal(
            PROFILE, document.get("hardware_runtime_profile"), "hardware_runtime_profile"
        )
    elif document.get("schema") == "tower-gpu-baseline-v2":
        for section in ("contract", "reference_identity"):
            value = document.get(section)
            original.require_equal(
                pinned[section],
                original.normalize_contract(value)
                if section == "contract"
                else original.public_reference(value),
                section,
            )
            adapted[section] = original.reference_contract()[section]
        # Preserve the verified data revision while adapting only the host profile.
        adapted["contract"] = dict(
            document["contract"],
            hardware_runtime_profile=adapted["contract"]["hardware_runtime_profile"],
        )
    else:
        raise ValueError("Host profile requires a measured GPU run summary")
    # Reuse every existing source, expert, schema and candidate identity check.
    result = original.read_summary(adapted, repeat=repeat)
    for section in ("contract", "reference_identity"):
        result[section] = pinned[section]
    if document["schema"] == "tower-gpu-baseline-v2":
        result["contract"] = copy.deepcopy(document["contract"])
    if document["schema"] == "tower-gpu-baseline-v1":
        result["candidate_identity"]["profile_provenance"] = (
            "Measured desktop reference profile in summary and ENVIRONMENT.json"
        )
    return result


def candidate_summary(
    identity: dict[str, Any], contract: dict[str, Any], trials: list[dict[str, Any]]
) -> dict[str, Any]:
    if selected() is None:
        document = original.candidate_summary(identity, contract, trials)
        document["benchmark_version"] = original.benchmark_version(identity)
        return document
    document = dict(
        schema="tower-gpu-baseline-v2",
        host_profile=selected(),
        benchmark_version=original.benchmark_version(identity),
        candidate_identity=identity,
        contract=contract,
        trials=trials,
        reference_identity=reference_contract()["reference_identity"],
    )
    if "public" in identity:
        public = identity["public"]
        document.update(
            public=public,
            **{
                key: public[key]
                for key in ("release_manifest_sha256", "commit", "data_manifest_sha256")
            },
        )
    read_summary(document)
    return document


def comparison_identity(
    left: dict[str, Any], right: dict[str, Any], *, informational: bool
) -> None:
    for section in ("reference_identity", "contract"):
        a, b = copy.deepcopy(left[section]), copy.deepcopy(right[section])
        if informational:
            # Only this explicit, non-accepting comparison may cross hardware profiles.
            a.pop("hardware_runtime_profile")
            b.pop("hardware_runtime_profile")
        original.require_equal(
            original.normalize_contract(a) if section == "contract" else a,
            original.normalize_contract(b) if section == "contract" else b,
            section,
        )
