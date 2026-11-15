"""Public evaluation data: content identity independent of its download locator."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from .download import download_file
from .scene_format import decode_scene, safe_path, scene_number


def manifest_path():
    return Path(__file__).with_name("public-data.json")


def manifest():
    return json.loads(manifest_path().read_bytes())


def payload_root(assets, data_root=None):
    return (
        Path(data_root).resolve()
        if data_root is not None
        else Path(assets).resolve() / "benchmark-2026.11.1"
    )


def read_payload(name, *, assets, data_root=None):
    expected = manifest()["payload_files"][name]
    payload = safe_path(payload_root(assets, data_root), name).read_bytes()
    if (
        len(payload) != expected["size"]
        or hashlib.sha256(payload).hexdigest() != expected["sha256"]
    ):
        raise ValueError(f"Changed benchmark payload: {name}")
    return json.loads(payload)


def verify_payloads(*, assets, data_root=None):
    root = payload_root(assets, data_root)
    verify_release(root)
    files = {}
    for name, expected in manifest()["payload_files"].items():
        read_payload(name, assets=assets, data_root=data_root)
        files[str(safe_path(root, name))] = expected["sha256"]
    return files


def equivalence_map(checkout):
    name = "benchmarks/2026.11.1/scene-equivalence.json"
    payload = safe_path(checkout, name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest()["files"][name]:
        raise ValueError("Changed scene equivalence map")
    return json.loads(payload)["scenes"]


def scene(task, scene_id, *, assets, data_root=None):
    scene_number(scene_id)
    body = read_payload(f"scenes/{task}.json", assets=assets, data_root=data_root)
    if body["schema"] != "tower-scenes-v2" or body["task"] != task:
        raise ValueError("Wrong scene collection")
    rows = [r for r in body["scenes"] if r["task"] == task and r["scene_id"] == scene_id]
    if len(rows) != 1:
        raise ValueError("Expected exactly one scene task/scene")
    decode_scene(rows[0])
    return rows[0]


def verify_locator(lock, *, document=None, require_pin=False):
    document = document or manifest()
    if lock.get("schema") != "tower-benchmark-data-lock-v1":
        raise ValueError("Unsupported data locator")
    if lock["manifest_sha256"] != document["manifest_sha256"]:
        raise ValueError("Data locator manifest_sha256 differs from content identity")
    if require_pin:
        if any("TODO(owner)" in str(lock[k]) for k in ("repository", "revision")):
            raise ValueError("data release not yet pinned; pass --data-root <dir>")
        if not re.fullmatch(r"[0-9a-f]{40}", lock["revision"]):
            raise ValueError("Data release must use an immutable HF commit")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", lock["repository"]):
            raise ValueError("Invalid data repository")
        safe_path(Path.cwd(), lock["directory"].strip("/") or ".")
    return lock


def verify_release(root, *, document=None):
    """Check the complete upload, including its independently pinned MANIFEST."""
    from .public_surface import check_lineage

    root = Path(root).resolve()
    document = document or manifest()
    payload = (root / "MANIFEST.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != document["manifest_sha256"]:
        raise ValueError("Changed data release MANIFEST")
    body = json.loads(payload)
    if body != dict(schema="tower-staged-release-v1", files=document["upload_files"]):
        raise ValueError("Data release file set differs")
    names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if names != set(body["files"]) | {"MANIFEST.json"}:
        raise ValueError("Unexpected data release files")
    for name, expected in body["files"].items():
        contents = safe_path(root, name).read_bytes()
        check_lineage(name, contents)
        if (
            len(contents) != expected["size"]
            or hashlib.sha256(contents).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"Changed data release file: {name}")
    return {str(root / name): row["sha256"] for name, row in body["files"].items()} | {
        str(root / "MANIFEST.json"): document["manifest_sha256"]
    }


def download(checkout, assets, *, data_root=None):
    """Download only an immutable owner-pinned release; verify before publication."""
    if data_root is not None:
        return verify_release(data_root)
    document = manifest()
    lock = verify_locator(
        json.loads(safe_path(checkout, "benchmarks/2026.11.1/assets.lock.json").read_bytes()),
        require_pin=True,
    )
    root = payload_root(assets)
    root.mkdir(parents=True, exist_ok=True)
    for temporary in root.rglob(".download-*"):
        if temporary.is_file():
            temporary.unlink()
    directory = lock["directory"].strip("/")
    prefix = "" if directory in ("", ".") else quote(directory) + "/"
    base = (
        f"https://huggingface.co/datasets/{lock['repository']}/resolve/{lock['revision']}/{prefix}"
    )

    def fetch(name, expected, size=None):
        download_file(
            base + quote(name), safe_path(root, name), expected, size=size, opener=urlopen
        )

    fetch("MANIFEST.json", document["manifest_sha256"])
    entries = json.loads((root / "MANIFEST.json").read_bytes())
    if entries != dict(schema="tower-staged-release-v1", files=document["upload_files"]):
        raise ValueError("Downloaded data file set differs")
    for name, row in entries["files"].items():
        fetch(name, row["sha256"], row["size"])
    return verify_release(root)
