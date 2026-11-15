# SPDX-License-Identifier: Apache-2.0
"""Record installed simulator/IK identity and Python/configuration source hashes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path


def snapshot(kind):
    root = Path(sys.prefix).resolve()
    versions = {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()}
    if kind == "isaac":
        if versions.get("isaacsim") != "6.1.0.0" or sys.version_info[:2] != (3, 12):
            raise ValueError("Expected Isaac Sim 6.1.0.0 with Python 3.12")
    else:
        import pinocchio

        expected = {"casadi": "3.7.2", "pin-pink": "3.4.0", "numpy": "1.26.4"}
        if pinocchio.__version__ != "3.8.0" or any(
            versions.get(k) != v for k, v in expected.items()
        ):
            raise ValueError("Independent IK environment differs from the campaign versions")
        versions["pinocchio"] = pinocchio.__version__
    files = {}
    for path in root.rglob("*"):
        if path.is_file() and (
            path.suffix in {".py", ".kit", ".toml"} or path.name in {"METADATA", "RECORD"}
        ):
            files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(root=str(root), python=sys.version.split()[0], packages=versions, files=files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("isaac", "ik"))
    print(json.dumps(snapshot(parser.parse_args().kind)))
