"""Public scene-v2 codec and content-addressed resolver, independent of archives."""

import copy
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "tower-scene-v2"
DEFAULT_BENCHMARK = "2026.11.1"
REVISIONS = ("2026.11.0", DEFAULT_BENCHMARK)
HASH = re.compile(r"[0-9a-f]{64}\Z")
PRIVATE_STRING = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\.(?:ecs\.)?internal\b", re.I)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe_path(root, name):
    root = Path(root).resolve()
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Unsafe scene resource path")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Scene resource escapes data root")
    return path


def revision(value):
    if value not in REVISIONS:
        raise ValueError(f"Unsupported benchmark revision: {value}")
    return value


def scene_number(scene_id):
    if not isinstance(scene_id, str) or not re.fullmatch(r"scene-[0-9]{3}", scene_id):
        raise ValueError("Expected a neutral scene identifier")
    number = int(scene_id.removeprefix("scene-"))
    if not 0 <= number < 50:
        raise ValueError("Scene identifier outside the released collection")
    return number


def validate_public(value):
    """Reject private strings; structural fields are governed by the schema allow-list."""
    if isinstance(value, dict):
        for key, item in value.items():
            validate_public(key)
            validate_public(item)
    elif isinstance(value, list):
        for item in value:
            validate_public(item)
    elif isinstance(value, str) and (
        value.startswith("/") or "://" in value or "\\" in value or PRIVATE_STRING.search(value)
    ):
        raise ValueError("Private path, URL, IP or host in public scene")


@lru_cache(maxsize=1)
def scene_schema():
    return json.loads(Path(__file__).with_name("scene-v2.schema.json").read_bytes())


def validate_schema(value, rule=None, *, path="scene"):
    """The standard JSON Schema subset used by our committed, closed schema.

    Independent Draft202012 validation is also run by the release gate. Unknown
    properties are never accepted; no archival redaction blacklist is needed.
    """
    document = scene_schema()
    rule = document if rule is None else rule
    if "$ref" in rule:
        rule = document["$defs"][rule["$ref"].removeprefix("#/$defs/")]
    types = rule["type"]
    types = [types] if isinstance(types, str) else types
    actual = (
        "null"
        if value is None
        else "boolean"
        if type(value) is bool
        else "integer"
        if type(value) is int
        else "number"
        if type(value) is float
        else "string"
        if isinstance(value, str)
        else "array"
        if isinstance(value, list)
        else "object"
        if isinstance(value, dict)
        else "unsupported"
    )
    if actual not in types and not (actual == "integer" and "number" in types):
        raise ValueError(f"Wrong scene field type: {path}")
    if "const" in rule and value != rule["const"]:
        raise ValueError(f"Wrong scene field value: {path}")
    if "enum" in rule and value not in rule["enum"]:
        raise ValueError(f"Unsupported scene field value: {path}")
    if isinstance(value, dict):
        properties = rule["properties"]
        if not set(rule.get("required", ())) <= value.keys() or set(value) - properties.keys():
            raise ValueError(f"Missing or unrecognized scene fields: {path}")
        for key, item in value.items():
            validate_schema(item, properties[key], path=path + "/" + key)
    elif isinstance(value, list):
        if not rule.get("minItems", 0) <= len(value) <= rule.get("maxItems", math.inf):
            raise ValueError(f"Wrong scene array length: {path}")
        for item in value:
            validate_schema(item, rule["items"], path=path + "/*")
    elif isinstance(value, str) and "pattern" in rule and not re.fullmatch(rule["pattern"], value):
        raise ValueError(f"Wrong scene field format: {path}")
    elif isinstance(value, (int, float)) and (
        not math.isfinite(value)
        or value < rule.get("minimum", -math.inf)
        or value > rule.get("maximum", math.inf)
    ):
        raise ValueError(f"Invalid scene number: {path}")


def blocks(episode):
    for group in (episode["episode_spec"], episode.get("task7_scene", {})):
        yield from group.get("blocks", ())


def decode_scene(body, asset_manifest=None) -> dict[str, Any]:
    validate_schema(body)
    validate_public(body)
    number = scene_number(body["scene_id"])
    expected = copy.deepcopy(body)
    hashes = (
        {entry["sha256"] for entry in asset_manifest.get("files", ())}
        if asset_manifest is not None
        else None
    )
    for block in blocks(expected["episode"]):
        for field, binding in (
            ("asset_reference", "asset_sha256"),
            ("texture_atlas", "texture_sha256"),
        ):
            value = block[binding]
            if value is None:
                if block[field] is not None:
                    raise ValueError("Unbound scene resource")
            else:
                if not HASH.fullmatch(value) or (hashes is not None and value not in hashes):
                    raise ValueError("Missing asset content hash")
                block[field] = "sha256:" + value
    if expected != body:
        raise ValueError("Noncanonical scene resources")
    # Only the verified manifest-bound resolver can authorize construction. Raw
    # DTO decoding is usable for schema round trips, but cannot authorize assets.
    episode = copy.deepcopy(body["episode"])
    episode["provenance"] = {"asset_closure_verified": asset_manifest is not None}
    episode["seed"] = number
    episode["episode_spec"].update(seed=number, logical_id=number)
    if "task7_scene" in episode:
        episode["task7_scene"]["seed"] = number
    return dict(
        task=body["task"],
        scene_id=body["scene_id"],
        seed=number,
        candidate=episode,
        initial_state=copy.deepcopy(body["initial_state"]),
    )


def encode_scene(row):
    """Round-trip a decoded public DTO, removing only adapter-injected fields."""
    episode = copy.deepcopy(row["candidate"])
    episode.pop("seed")
    episode.pop("provenance")
    episode["episode_spec"].pop("seed")
    episode["episode_spec"].pop("logical_id")
    if "task7_scene" in episode:
        episode["task7_scene"].pop("seed")
    body = dict(
        schema=SCHEMA,
        scene_id=row["scene_id"],
        task=row["task"],
        episode=episode,
        initial_state=copy.deepcopy(row["initial_state"]),
    )
    validate_schema(body)
    return body


def resolve_physical(body, assets, manifest, *, selector=None):
    row = decode_scene(body, manifest)
    if selector is not None:
        row["seed"] = selector
        row["candidate"]["seed"] = selector
        row["candidate"]["episode_spec"].update(seed=selector, logical_id=selector)
        if "task7_scene" in row["candidate"]:
            row["candidate"]["task7_scene"]["seed"] = selector
    by_hash = {entry["sha256"]: entry["path"] for entry in manifest["files"]}
    for block in blocks(row["candidate"]):
        for field, binding in (
            ("asset_reference", "asset_sha256"),
            ("texture_atlas", "texture_sha256"),
        ):
            if block[binding] is not None:
                block[field] = str(safe_path(assets, by_hash[block[binding]]))
    row["candidate_sha256"] = digest(body["episode"])
    row["initial_state_sha256"] = digest(body["initial_state"])
    return row
