"""Versioned replay identities, independent of the installed simulator source tree.

The reference contract is transcribed from immutable P0 evidence. Legacy v1
never recorded commit/wheel identities; reading it must not invent those fields.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.maintainer.scene_equivalence import normalize_contract

CONTRACT_PATH = Path(__file__).with_name("gpu_reference_contract.json")
COMPONENTS = ("bootstrap", "command_mapping", "observation", "cameras", "scoring", "early_stop")


def reference_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_bytes())


def require_equal(expected: Any, actual: Any, path: str) -> None:
    """Strict JSON equality with a useful first differing field (including types)."""
    if type(expected) is not type(actual):
        raise ValueError(f"Replay contract differs: {path} (type)")
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f"Replay contract differs: {path} (keys)")
        for key, value in expected.items():
            require_equal(value, actual[key], f"{path}/{key}")
    elif expected != actual:
        raise ValueError(f"Replay contract differs: {path}")


def benchmark_version(identity):
    """Require a measured tower-sim revision; legacy receipts have no such field."""
    if identity.get("implementation") == "tower-sim":
        value = identity.get("benchmark_version")
        if value not in ("2026.11.0", "2026.11.1"):
            raise ValueError("Invalid candidate identity: benchmark_version is required")
        return value
    return identity.get("benchmark_version")


def validate_candidate(identity: dict[str, Any]) -> None:
    benchmark_version(identity)
    if identity.get("schema") != "tower-gpu-candidate-v2":
        raise ValueError("Invalid candidate identity: schema")
    if identity.get("implementation") != "tower-sim":
        raise ValueError("Invalid candidate identity: implementation")
    if not re.fullmatch(r"[0-9a-f]{40}", str(identity.get("commit", ""))):
        raise ValueError("Invalid candidate identity: commit")
    if identity.get("data_version") not in (
        reference_contract()["contract"]["data_version"],
        "benchmark-2026.11.1",
    ):
        raise ValueError("Invalid candidate identity: data_version")
    for field in ("wheel_sha256", "dependency_sha256"):
        values = identity.get(field)
        if (
            not isinstance(values, dict)
            or not values
            or not all(
                isinstance(name, str)
                and name
                and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest)
                for name, digest in values.items()
            )
        ):
            raise ValueError(f"Invalid candidate identity: {field}")
    substitutions = identity.get("frozen_components")
    if (
        not isinstance(substitutions, list)
        or any(name not in COMPONENTS for name in substitutions)
        or len(set(substitutions)) != len(substitutions)
    ):
        raise ValueError("Invalid candidate identity: frozen_components")


def public_trial(row):
    if "seed" not in row:
        return dict(row)
    from scripts.maintainer.archive_root import archive_bundle
    from scripts.maintainer.archive_scene import public_scene_id

    value = dict(row)
    value["scene_id"] = public_scene_id(
        Path(__file__).resolve().parents[2],
        value["task"],
        value.pop("seed"),
        v1_root=archive_bundle(),
    )
    return value


def public_reference(identity):
    from copy import deepcopy

    result = deepcopy(identity)
    result["experts"] = {
        task: {
            key: value for key, value in public_trial(dict(row, task=task)).items() if key != "task"
        }
        for task, row in result["experts"].items()
    }
    return result


def read_summary(document: dict[str, Any], *, repeat: bool = False) -> dict[str, Any]:
    """Normalize v1 runs, the committed P0 receipt, or a v2 candidate in memory.

    No reference data is read from the in-tree frozen runtime. Legacy hardware
    provenance belongs to P0's receipt, not to a newly measured v1 run.
    """
    schema = document.get("schema")
    pinned = reference_contract()
    if schema in ("tower-gpu-baseline-v1", "tower-p0-gpu-reference-v1"):
        trials = (
            document["trials"]
            if schema == "tower-gpu-baseline-v1"
            else document["repeat_runs" if repeat else "baseline_runs"]
        )
        if schema == "tower-p0-gpu-reference-v1":
            require_equal(
                pinned["reference_identity"]["benchmark"], document["reference"], "reference"
            )
            require_equal(
                pinned["contract"]["hardware_runtime_profile"],
                document["environment"],
                "environment",
            )
        for row in trials:
            if (
                row["source_manifest_sha256"]
                != pinned["reference_identity"]["source_manifest_sha256"]
            ):
                raise ValueError(f"Legacy v1 source_manifest_sha256 differs: {row['task']}")
        identity = {
            "schema": "tower-gpu-legacy-v1-identity",
            "implementation": "frozen",
            "source_manifest_sha256": pinned["reference_identity"]["source_manifest_sha256"],
            "commit": None,
            "wheel_sha256": None,
            "dependency_sha256": None,
            "data_version": pinned["contract"]["data_version"],
            "profile_provenance": "P0 archived profile; not measured in legacy per-task summaries",
        }
        result = dict(
            trials=trials,
            candidate_identity=identity,
            **{key: pinned[key] for key in ("reference_identity", "contract")},
        )
    elif schema == "tower-gpu-baseline-v2":
        for field in ("reference_identity", "contract"):
            require_equal(
                pinned[field],
                normalize_contract(document[field])
                if field == "contract"
                else public_reference(document[field]),
                field,
            )
        validate_candidate(document["candidate_identity"])
        require_equal(
            document["contract"]["data_version"],
            document["candidate_identity"]["data_version"],
            "candidate/data_version",
        )
        result = {
            key: document[key]
            for key in ("trials", "candidate_identity", "reference_identity", "contract")
        }
        if any("source_manifest_sha256" in row for row in result["trials"]):
            raise ValueError(
                "Candidate trials must not reuse the legacy source_manifest_sha256 field"
            )
    else:
        raise ValueError(f"Unsupported GPU summary schema: {schema!r}")
    result["trials"] = [public_trial(row) for row in result["trials"]]
    result["reference_identity"] = public_reference(result["reference_identity"])
    for row in result["trials"]:
        task = row["task"]
        expert = pinned["reference_identity"]["experts"].get(task)
        if expert is None:
            raise ValueError(f"Unknown reference task: {task}")
        for field, expected in (
            ("scene_id", expert["scene_id"]),
            ("expert_sha256", expert["sha256"]),
        ):
            require_equal(expected, row[field], f"{task}/{field}")
    return result


def candidate_summary(
    identity: dict[str, Any], contract: dict[str, Any], trials: list[dict[str, Any]]
) -> dict[str, Any]:
    document = dict(
        schema="tower-gpu-baseline-v2",
        candidate_identity=identity,
        trials=trials,
        reference_identity=reference_contract()["reference_identity"],
        contract=contract,
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
