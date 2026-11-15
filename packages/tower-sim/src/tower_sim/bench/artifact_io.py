"""JSON evidence I/O shared by public and maintainer entrypoints."""

import json
import lzma
from pathlib import Path


def read(path):
    path = Path(path)
    payload = path.read_bytes()
    if path.name.endswith(".json.xz"):
        payload = lzma.decompress(payload)
    return json.loads(payload)


def write(path, value):
    path = Path(path)
    if path.name.endswith(".json.xz"):
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        path.write_bytes(lzma.compress(payload.encode(), preset=9))
    else:
        path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
