"""Maintainer-only archive comparison and identifier redaction.

The public wheels never import this module. These functions require
separately supplied immutable evidence; no executable frozen module is imported.
"""

import ast
import copy
import hashlib
import json
import re
from pathlib import Path

from scripts.maintainer.archive_root import ArchiveUnavailable as MissingV1Data

__all__ = ["MissingV1Data"]

from tower_sim.bench.scene_format import digest, safe_path, scene_number

V1_PREFIX = "simulator/benchmark-202611/"
HASH = re.compile(r"[0-9a-f]{64}\Z")


class V1Source:
    """Read hash-bound immutable v1 evidence from an explicit external archive."""

    def __init__(self, checkout, v1_root=None):
        from scripts.maintainer.archive_root import archive_bundle

        self.checkout = Path(checkout).resolve()
        self.bundle = archive_bundle(v1_root)

    def path(self, relative):
        if not relative.startswith(V1_PREFIX):
            raise ValueError("Expected a v1 bundle-relative resource")
        return safe_path(self.bundle, relative.removeprefix(V1_PREFIX))

    def read(self, relative, expected):
        if not relative.startswith(V1_PREFIX) or ".." in Path(relative).parts:
            raise ValueError("Expected a v1 bundle-relative resource")
        payload = self.path(relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Changed immutable v1 data: {relative}")
        return payload


def manifest_path():
    return Path(__file__).with_name("archive-data.json")


def manifest():
    return json.loads(manifest_path().read_bytes())


def v1_scene_files(document=None):
    document = document or manifest()
    return {
        name: value
        for name, value in document["files"].items()
        if name in (document["protocol"], document["scene_index"])
        or name.startswith(V1_PREFIX + "scenes/")
    }


def v1_scene(checkout, task, seed, *, v1_root=None):
    source = V1Source(checkout, v1_root)
    name = V1_PREFIX + f"scenes/{task}.json"
    pins = v1_scene_files()
    if name not in pins:
        raise ValueError(f"Unsupported task: {task}")
    body = json.loads(source.read(name, pins[name]))
    rows = [r for r in body["scenes"] if r["task"] == task and r["seed"] == seed]
    if len(rows) != 1:
        raise ValueError("Expected exactly one v1 scene task/seed")
    return rows[0]


def archived_scene(checkout, task, scene_id, *, v1_root=None):
    entries = [
        r for r in equivalence_map(checkout) if r["task"] == task and r["scene_id"] == scene_id
    ]
    if len(entries) != 1:
        raise ValueError("No unique archive-to-public scene binding")
    source = V1Source(checkout, v1_root)
    name = V1_PREFIX + f"scenes/{task}.json"
    rows = json.loads(source.read(name, v1_scene_files()[name]))["scenes"]
    matches = [row for row in rows if digest(row) == entries[0]["old_sha256"]]
    if len(matches) != 1:
        raise ValueError("No unique archived scene")
    return matches[0]


def public_scene_id(checkout, task, selector, *, v1_root=None):
    if isinstance(selector, str):
        scene_number(selector)
        return selector
    if type(selector) is int and 0 <= selector < 50:
        return f"scene-{selector:03d}"
    if v1_root is None:
        raise ValueError(
            "Use a released scene identifier; historical selectors require an explicit archive root"
        )
    original = v1_scene(checkout, task, selector, v1_root=v1_root)
    matches = [
        r["scene_id"]
        for r in equivalence_map(checkout)
        if r["task"] == task and r["old_sha256"] == digest(original)
    ]
    if len(matches) != 1:
        raise ValueError("No unique archive-to-public scene binding")
    return matches[0]


def equivalence_map(checkout):
    from tower_sim.bench.data_release import equivalence_map as read_map

    return read_map(checkout)


def protocol_path(checkout, *, v1_root=None):
    return V1Source(checkout, v1_root).path(manifest()["protocol"])


def relocate_v1(row, assets, manifest):
    """Complete historical reader: relocate only strings matching block assets."""
    by_hash = {item["sha256"]: item["path"] for item in manifest["files"]}
    mapping = {}
    for block in row["candidate"]["episode_spec"]["blocks"]:
        for field, binding in (
            ("asset_reference", "asset_sha256"),
            ("texture_atlas", "texture_sha256"),
        ):
            if block.get(field) is not None:
                if block[binding] not in by_hash:
                    raise ValueError(f"Release lacks required {field}: {block[binding]}")
                mapping[block[field]] = str(safe_path(assets, by_hash[block[binding]]))

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        return mapping.get(value, value) if isinstance(value, str) else value

    result = copy.deepcopy(row)
    result["candidate"] = relocate(result["candidate"])
    return result


def lineage(row):
    """Hash-only lineage: neither original paths nor code filenames leave evidence."""
    hashes = set()

    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str) and HASH.fullmatch(value):
            hashes.add(value)

    collect(row["candidate"]["provenance"])
    collect(row["initial_state"].get("source_bootstrap_sha256"))
    return dict(
        scene_id=f"{row['task']}/{row['seed']}",
        source_sha256=digest(row),
        content_sha256=sorted(hashes),
    )


def field_contract():
    return json.loads(Path(__file__).with_name("dropped-fields.json").read_bytes())


def remove_declared_fields(row):
    """Remove ONLY the committed JSON-pointer list; retain every other value."""
    result = copy.deepcopy(row)
    for pointer in field_contract()["dropped"]:
        parts = pointer.strip("/").split("/")
        parents = [result]
        for part in parts[:-1]:
            parents = [
                child
                for parent in parents
                for child in (
                    parent
                    if part == "*" and isinstance(parent, list)
                    else [parent[part]]
                    if isinstance(parent, dict) and part in parent
                    else []
                )
            ]
        for parent in parents:
            parent.pop(parts[-1], None)
    return result


def relabel(row, scene_id):
    """Mechanical wire/selector labels, separate from declared field removal."""
    result = copy.deepcopy(row)
    number = scene_number(scene_id)
    result.update(scene_id=scene_id, seed=number)
    candidate = result["candidate"]
    candidate.update(schema="tower-evaluation-state-v2", seed=number)
    candidate["episode_spec"].update(seed=number, logical_id=number)
    if "task7_scene" in candidate:
        candidate["task7_scene"]["seed"] = number
        result["initial_state"]["coordinate_frame"] = "encoder_with_single_J2_conversion"
    for group in (candidate["episode_spec"], candidate.get("task7_scene", {})):
        metadata = group.get("layout_metadata", {})
        if "task_id" in metadata:
            metadata["task_id"] = result["task"]
        for before, after in field_contract()["renamed"].items():
            if before in metadata:
                metadata[after] = metadata.pop(before)
    return result


def changed_pointers(before, after, pointer=""):
    """Exact JSON pointers changed by an adapter, independent of that adapter."""
    if type(before) is not type(after):
        return {pointer}
    if isinstance(before, dict):
        changed = set()
        for key in before.keys() | after.keys():
            path = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before or key not in after:
                changed.add(path)
            else:
                changed.update(changed_pointers(before[key], after[key], path))
        return changed
    if isinstance(before, list) and len(before) == len(after):
        return {
            path
            for index, (old, new) in enumerate(zip(before, after, strict=True))
            for path in changed_pointers(old, new, pointer + "/" + str(index))
        }
    return {pointer} if before != after else set()


def assert_runtime_labels(before, after):
    """Bind mechanical relabeling to the committed path allow-list."""
    contract = field_contract()
    allowed = set(contract["runtime_labels"])
    for group in ("episode_spec", "task7_scene"):
        for old, new in contract["renamed"].items():
            allowed.update(f"/candidate/{group}/layout_metadata/{key}" for key in (old, new))
    unexpected = changed_pointers(before, after) - allowed
    if unexpected:
        raise ValueError(f"Undeclared runtime label changes: {sorted(unexpected)}")


def public_scene(original, scene_id):
    """Archive conversion; never needed by the public reader."""
    from tower_sim.bench.scene_format import blocks, encode_scene

    row = relabel(remove_declared_fields(original), scene_id)
    for block in blocks(row["candidate"]):
        for field, binding in (
            ("asset_reference", "asset_sha256"),
            ("texture_atlas", "texture_sha256"),
        ):
            if block[field] is not None:
                if not HASH.fullmatch(block[binding]):
                    raise ValueError("Missing asset content hash")
                block[field] = "sha256:" + block[binding]
    return encode_scene(row)


def expected_runtime(original, body, assets, asset_manifest):
    """The FULL v1 runtime input minus exactly the declared removals and labels.

    This is deliberately never applied to the v2 result: an omitted physical
    field, including measured_asset_id, must produce an inequality.
    """
    expected = relabel(
        remove_declared_fields(relocate_v1(original, assets, asset_manifest)), body["scene_id"]
    )
    expected["candidate_sha256"] = digest(body["episode"])
    expected["initial_state_sha256"] = digest(body["initial_state"])
    return expected


def assert_runtime_equal(original, body, assets, asset_manifest):
    from tower_sim.bench.scene_format import resolve_physical

    expected = expected_runtime(original, body, assets, asset_manifest)
    assert_runtime_labels(
        remove_declared_fields(relocate_v1(original, assets, asset_manifest)), expected
    )
    resolved = resolve_physical(body, assets, asset_manifest)
    if expected != resolved:
        raise ValueError("Full runtime input differs after declared field removals")
    return resolved


def compatibility_view(body, original, assets, asset_manifest):
    assert_runtime_equal(original, body, assets, asset_manifest)
    return relocate_v1(original, assets, asset_manifest)


def dropped_key_reads(root):
    """Static reads of removed fields in runtime code (archive adapter excluded).

    Generic provenance names are matched with their container; specific field
    names are forbidden regardless of the local variable used by a consumer.
    """
    dropped = field_contract()["dropped"]
    keys = {p.rsplit("/", 1)[-1] for p in dropped}
    non_scene_reads = {
        ("packages/tower-sim/src/tower_sim/bench/evaluation.py", "request", "manifest"),
        ("tools/simulator/benchmark.py", "request", "manifest"),
        ("tools/simulator/benchmark_candidate.py", "request", "manifest"),
        ("packages/tower-sim/src/tower_sim/bench/policy_entry.py", "status", "diagnostic_only"),
        ("tools/simulator/benchmark_diagnostic.py", "status", "diagnostic_only"),
        ("tools/simulator/benchmark_diagnostic_score.py", "status", "diagnostic_only"),
    }
    findings = []
    roots = ("packages/tower-sim/src/tower_sim", "src/tower-eval/src/tower_eval", "tools/simulator")
    for directory in roots:
        for path in sorted((Path(root) / directory).rglob("*.py")):
            if "archive_compat" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                key, base = None, None
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                    key, base = node.slice, node.value
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"get", "pop", "setdefault"}
                    and node.args
                ):
                    key, base = node.args[0], node.func.value
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(node, ast.expr)
                    and base is not None
                    and key.value in keys
                    and (path.relative_to(root).as_posix(), ast.unparse(base), key.value)
                    not in non_scene_reads
                ):
                    findings.append(f"{path.relative_to(root)}:{node.lineno}: {key.value}")
    return findings


def verify_archive_data(checkout, *, v1_root=None):
    """Validate all archived scene pins without distributing the retired revision."""
    source = V1Source(checkout, v1_root)
    document = manifest()
    files = {}
    parsed = {}
    for name, expected in v1_scene_files(document).items():
        contents = source.read(name, expected)
        files[str(source.path(name))] = expected
        if name in (document["protocol"], document["scene_index"]):
            parsed[name] = json.loads(contents)
    index = parsed[document["scene_index"]]["scenes"]
    protocol = parsed[document["protocol"]]
    if len(index) != 600 or len({(r["task"], r["seed"]) for r in index}) != 600:
        raise ValueError("Expected exactly 600 unique archived scenes")
    if len(protocol["tasks"]) != 12:
        raise ValueError("Expected exactly 12 archived tasks")
    for task, settings in protocol["tasks"].items():
        if sum(row["task"] == task for row in index) != settings["scene_count"]:
            raise ValueError(f"Archived scene count differs for {task}")
    return files
