"""Build a reviewable release manifest from three local wheels; never publishes."""

import sys

sys.dont_write_bytecode = True
import argparse
import io
import json
import subprocess
import zipfile
from email.parser import Parser
from pathlib import Path

from tower_sim.bench.public_surface import check_lineage
from tower_sim.bench.resources import save_exclusive
from tower_sim.bench.verification import (
    CHECKOUT_TREES,
    PACKAGES,
    checkout_files,
    checkout_source,
    digest,
    envelope,
    verify_checkout_layout,
    wheel_payload,
)

from scripts.maintainer.candidate_replay import wheel_source_path


def create(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    names = checkout_files(root)
    verify_checkout_layout(root)
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    packages = {}
    for name, version in PACKAGES.items():
        wheels = list(output.parent.glob(name.replace("-", "_") + "-*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"Exactly one release wheel required: {name}")
        wheel = wheels[0]
        payload = wheel.read_bytes()
        members = wheel_payload(payload, name, version)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.namelist():
                check_lineage(member, archive.read(member))
            metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("Invalid wheel metadata")
            message = Parser().parsestr(archive.read(metadata[0]).decode())
            if message["Name"] != name or message["Version"] != version:
                raise ValueError(f"Unexpected wheel release: {name}")
            for member in members:
                original = subprocess.check_output(
                    ["git", "-C", str(root), "show", f"HEAD:{wheel_source_path(member)}"]
                )
                if archive.read(member) != original:
                    raise ValueError(f"Wheel differs from committed source: {member}")
        packages[name] = dict(version=version, wheel=wheel.name, sha256=digest(payload))
    files = {name: digest(checkout_source(root, commit, name)) for name in names}
    body = dict(
        schema="tower-release-v2",
        public=envelope(),
        commit=commit,
        packages=packages,
        checkout_trees=list(CHECKOUT_TREES),
        checkout_files=files,
    )
    save_exclusive(output, body)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(Path(__file__).resolve().parents[2], args.output), indent=2))


if __name__ == "__main__":
    main()
