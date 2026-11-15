"""Narrow cross-revision identity rule; replay formulas and tolerances are unchanged."""

import copy
import hashlib
import json
from pathlib import Path

from tower_sim.bench import data_release
from tower_sim.bench.scene_format import digest, safe_path

from scripts.maintainer.archive_scene import V1_PREFIX, V1Source, assert_runtime_equal
from scripts.maintainer.archive_scene import manifest as archive_manifest

ROOT = Path(__file__).resolve().parents[2]


def expected_receipt(root=ROOT):
    manifest = data_release.manifest()
    name = "benchmarks/2026.11.1/scene-equivalence.json"
    path = safe_path(root, name)
    payload = path.read_bytes()
    map_sha256 = hashlib.sha256(payload).hexdigest()
    if map_sha256 != manifest["files"][name]:
        raise ValueError("Changed scene equivalence map")
    table = json.loads(payload)
    pairs = {(row["task"], row["scene_id"]) for row in table["scenes"]}
    if len(table["scenes"]) != 600 or len(pairs) != 600:
        raise ValueError("Incomplete scene equivalence map")
    return dict(
        schema="tower-cross-revision-scenes-v1",
        old_data_version="benchmark-2026.11.0",
        new_data_version="benchmark-2026.11.1",
        map_sha256=map_sha256,
        payload_manifest_sha256=manifest["files"]["benchmarks/2026.11.1/data-manifest.json"],
        physical_map_sha256=digest(
            {f"{r['task']}/{r['scene_id']}": r["physical_sha256"] for r in table["scenes"]}
        ),
        scenes=600,
    )


def prove(root, assets, data_root=None, *, v1_root=None):
    """Read both actual releases and prove every physical scene against the pinned map."""
    receipt = expected_receipt(root)
    table = json.loads((Path(root) / "benchmarks/2026.11.1/scene-equivalence.json").read_bytes())
    old_manifest = archive_manifest()
    source = V1Source(root, v1_root)
    old_rows, new_rows = {}, {}
    for task in sorted({r["task"] for r in table["scenes"]}):
        name = V1_PREFIX + f"scenes/{task}.json"
        payload = source.read(name, old_manifest["files"][name])
        old_rows.update({digest(r): r for r in json.loads(payload)["scenes"]})
        new = data_release.read_payload(f"scenes/{task}.json", assets=assets, data_root=data_root)
        new_rows.update({(r["task"], r["scene_id"]): r for r in new["scenes"]})
    if len(old_rows) != 600 or len(new_rows) != 600:
        raise ValueError("Incomplete physical scene coverage")
    asset_manifest = json.loads((Path(assets) / "manifest.json").read_bytes())
    for row in table["scenes"]:
        key = (row["task"], row["scene_id"])
        old, new = old_rows[row["old_sha256"]], new_rows[key]
        physical = dict(episode=new["episode"], initial_state=new["initial_state"])
        assert_runtime_equal(old, new, assets, asset_manifest)
        if (
            digest(old) != row["old_sha256"]
            or digest(new) != row["new_sha256"]
            or digest(physical) != row["physical_sha256"]
        ):
            raise ValueError(f"Scene equivalence mismatch: {key}")
    return receipt


def normalize_contract(contract):
    """Return the original comparison identity ONLY for the pinned equal scene release.

    Keep both revisions and the equality receipt in original summaries and compare
    output. This normalization does not alter any physics/trace/expert field.
    """
    result = copy.deepcopy(contract)
    if result.get("data_version") == "benchmark-2026.11.1":
        if (
            result.get("benchmark") != "benchmark-2026.11.1"
            or result.pop("scene_equivalence", None) != expected_receipt()
        ):
            raise ValueError("Unproven cross-revision scene identity")
        result["benchmark"] = "benchmark-2026.11.0"
        result["data_version"] = "benchmark-2026.11.0"
    return result
