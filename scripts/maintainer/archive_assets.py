"""Authenticated, archive-only asset fetch for frozen reproduction."""

import argparse
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

LOCK = Path(__file__).with_name("archive-assets.lock.json")


def download(destination):
    from tower_sim.bench.scene_format import safe_path

    hf_hub_download = importlib.import_module("huggingface_hub").hf_hub_download

    lock = json.loads(LOCK.read_bytes())
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)

    def fetch(name, expected, size=None):
        source = hf_hub_download(
            repo_id=lock["repository"],
            repo_type="dataset",
            revision=lock["revision"],
            filename=lock["directory"] + "/" + name,
            token=True,
        )
        payload = Path(source).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected or (
            size is not None and len(payload) != size
        ):
            raise ValueError(f"Changed archived asset: {name}")
        target = safe_path(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    manifest = fetch("manifest.json", lock["manifest_sha256"])
    for row in json.loads(manifest.read_bytes())["files"]:
        fetch(row["path"], row["sha256"], row["size"])
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    checkout = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=Path(__file__).parent, text=True
        ).strip()
    ).resolve()
    destination = args.destination.expanduser().resolve()
    if destination.is_relative_to(checkout):
        parser.error("Private archive assets must be downloaded outside the checkout")
    print(download(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
