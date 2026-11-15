"""Resolve all v1 scenes through the real reader from an export, tree, or tag."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from scripts.maintainer.archive_scene import V1_PREFIX, V1Source, manifest, relocate_v1
from scripts.maintainer.golden_support import ROOT, write
from tests.characterization.golden_compare import canonical


def scene_resolution_table(*, v1_root=None, checkout=ROOT) -> dict[str, str]:
    source = V1Source(checkout, v1_root)
    pins = manifest()["files"]

    def read(name):
        name = V1_PREFIX + name
        return json.loads(source.read(name, pins[name]))

    index = read("scene-index.json")["scenes"]
    files, scenes = {}, {}
    for task in read("protocol.json")["tasks"]:
        for row in read("scenes/" + task + ".json")["scenes"]:
            scenes[task, row["seed"]] = row
            for block in row["candidate"]["episode_spec"]["blocks"]:
                for key in ("asset_sha256", "texture_sha256"):
                    digest = block.get(key)
                    if digest:
                        files[digest] = dict(sha256=digest, path="objects/" + digest)
    with tempfile.TemporaryDirectory(prefix="tower-resolve-") as directory:
        root = Path(directory).resolve()
        assets = dict(schema="tower-benchmark-assets-v1", files=list(files.values()))
        write(root / "manifest.json", assets)

        def logical(value: Any) -> Any:
            if isinstance(value, str) and value.startswith(str(root) + "/"):
                return "@assets/" + value[len(str(root)) + 1 :]
            if isinstance(value, dict):
                return {key: logical(item) for key, item in value.items()}
            if isinstance(value, list):
                return [logical(item) for item in value]
            return value

        result = {}
        for row in index:
            resolved = relocate_v1(scenes[row["task"], row["seed"]], root, assets)
            result[
                f"{row['task']}/scene-{next(i for i, r in enumerate(json.loads(source.read(V1_PREFIX + 'scenes/' + row['task'] + '.json', pins[V1_PREFIX + 'scenes/' + row['task'] + '.json']))['scenes']) if r['seed'] == row['seed']):03d}"
            ] = hashlib.sha256(canonical(logical(resolved))).hexdigest()
    assert len(result) == 600
    return result
