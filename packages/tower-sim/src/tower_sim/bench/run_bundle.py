"""Durable, content-addressed inputs for relocatable benchmark attempts."""

import re
from pathlib import Path

from .resources import save_exclusive, sha
from .verification import read


def scene_ids(seeds):
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(n) is not int or not 0 <= n < 50 for n in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Invalid scene selection")
    return [f"scene-{n:03d}" for n in seeds]


def inside(root, path):
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes bundle")
    return result


def retain(request_path, attempt):
    """Copy bound input bytes before launch; never rewrite historical JSON."""
    request = read(request_path)
    bindings = {}
    for key in (
        "protocol",
        "runtime_manifest",
        "release_manifest",
        "dependency_manifest",
        "ik_dependency_manifest",
        "ik_bounds",
    ):
        bindings[request[key]] = request[key + "_sha256"]
    bindings.update(read(request["runtime_manifest"], request["runtime_manifest_sha256"])["files"])
    destination = attempt / "inputs"
    destination.mkdir()
    for source, digest in bindings.items():
        payload = Path(source).read_bytes()
        from .verification import digest as hash_bytes

        if hash_bytes(payload) != digest:
            raise ValueError("Changed input while retaining bundle")
        target = destination / digest
        if not target.exists():
            target.write_bytes(payload)
    (attempt / "request.json").write_bytes(Path(request_path).read_bytes())
    save_exclusive(
        attempt / "prepared.json",
        dict(
            schema="tower-benchmark-attempt-v1",
            request_sha256=sha(attempt / "request.json"),
            bindings=bindings,
        ),
    )


class Bundle:
    def __init__(self, attempt):
        self.attempt = attempt
        self.prepared = read(attempt / "prepared.json")
        if self.prepared.get("schema") != "tower-benchmark-attempt-v1":
            raise ValueError("Unsupported attempt schema")
        self.request = read(attempt / "request.json", self.prepared["request_sha256"])

    def path(self, original, expected):
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("Invalid binding digest")
        if self.prepared["bindings"].get(original) != expected:
            raise ValueError("Missing retained input binding")
        path = inside(self.attempt, "inputs/" + expected)
        if sha(path) != expected:
            raise ValueError("Changed retained input")
        return path

    def read(self, original, expected):
        return read(self.path(original, expected), expected)

    def field(self, key):
        return self.read(self.request[key], self.request[key + "_sha256"])
