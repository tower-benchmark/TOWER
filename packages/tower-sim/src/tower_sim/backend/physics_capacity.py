# SPDX-License-Identifier: Apache-2.0
"""Fixed GPU contact capacity and bounded detection of invalid PhysX runs."""

from pathlib import Path


def configure_capacity(context, capacity, found_lost_capacity=1048576):
    requested = {
        "gpu_total_aggregate_pairs_capacity": capacity,
        "gpu_found_lost_aggregate_pairs_capacity": found_lost_capacity,
    }
    report = {}
    for name, value in requested.items():
        getattr(context, "set_" + name)(value)
        actual = getattr(context, "get_" + name)()
        if actual != value:
            raise RuntimeError(f"PhysX {name} readback mismatch: {actual} != {value}")
        report[name] = actual
    return report


class PhysicsLogGuard:
    """Scan appended log bytes without retaining the potentially huge Kit log.

    PhysX errors invalidate the entire batch, including receipts written before
    detection. The caller persists that invalidation before stopping it.
    """

    def __init__(self, path, offset=0):
        self.path = Path(path)
        self.offset = offset
        self.tail = b""

    def check(self):
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            while chunk := stream.read(65536):
                data = self.tail + chunk
                marker = data.lower().find(b"physx error:")
                if marker >= 0:
                    excerpt = (
                        data[marker : marker + 1024]
                        .split(b"\n", 1)[0]
                        .decode("utf-8", errors="replace")
                    )
                    raise RuntimeError(excerpt)
                self.tail = data[-128:]
                self.offset = stream.tell()


def require_valid_run(directory):
    invalidation = Path(directory) / "INVALIDATED.json"
    if invalidation.exists():
        raise ValueError(
            f"campaign was invalidated; cannot resume, replay or export: {invalidation}"
        )
