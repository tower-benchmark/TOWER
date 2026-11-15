"""Public CPU fixtures use the same pinned asset release as the simulator."""

import argparse
import errno
import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path

from tower_sim.bench.download import RETRY_ERRORS

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "packages/tower-sim/src/tower_sim/data/benchmark/assets.lock.json"


class PublicAssetsUnavailable(RuntimeError):
    """A public fixture is unavailable; the test must report a marked skip."""


def verify_file_set(root):
    from tower_sim.bench.data_release import verify_release

    root = root.resolve()
    manifest = root / "manifest.json"
    expected = {"manifest.json"} | {
        row["path"] for row in json.loads(manifest.read_bytes())["files"]
    }
    release = verify_release(root / "benchmark-2026.11.1")
    expected.update(Path(path).relative_to(root).as_posix() for path in release)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError(f"Public asset file set differs: {sorted(actual ^ expected)}")


def public_asset_root(root=None):
    lock = json.loads(LOCK.read_bytes())
    if not isinstance(lock.get("revision"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", lock["revision"]
    ):
        raise ValueError("Public assets must use an immutable HF commit")
    explicit = root or os.environ.get("TOWER_ASSET_ROOT")
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    selected = Path(explicit) if explicit else cache / "tower/assets" / lock["manifest_sha256"]
    manifest = selected / "manifest.json"
    if not manifest.is_file():
        if explicit:
            raise ValueError("Incomplete explicit public asset root: manifest.json is missing")
        reason = "SKIP: public assets unavailable; set TOWER_ASSET_ROOT or --asset-root"
        if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TOWER_OFFLINE") == "1":
            raise PublicAssetsUnavailable(reason + " (offline)")
        sys.path.insert(0, str(ROOT / "tools/simulator"))
        download = importlib.import_module("benchmark_assets").download

        try:
            selected.parent.mkdir(parents=True, exist_ok=True)
            # Keep verified files across failures. Serialize users of this shared partial root.
            import fcntl

            completed = selected.parent / (".partial-" + lock["manifest_sha256"])
            with completed.with_suffix(".lock").open("a") as guard:
                fcntl.flock(guard, fcntl.LOCK_EX)
                if not manifest.is_file():
                    for temporary in completed.rglob(".download-*"):
                        temporary.unlink()
                    download(completed)
                    verify_file_set(completed)
                    # A concurrent publisher must still be verified below.
                    try:
                        completed.rename(selected)
                    except OSError as error:
                        if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                            raise
        except RETRY_ERRORS as error:
            raise PublicAssetsUnavailable(reason + f" ({type(error).__name__})") from error
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != lock["manifest_sha256"]:
        raise ValueError("Changed public asset manifest")
    from tower_sim.bench.data_release import verify_release

    verify_release(selected / "benchmark-2026.11.1")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    try:
        print(public_asset_root(args.asset_root))
    except PublicAssetsUnavailable as error:
        print(str(error))
        return 77
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
