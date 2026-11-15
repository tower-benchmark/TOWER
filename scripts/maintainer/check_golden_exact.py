"""Authoritative Linux replay of every archived golden (no simulator/GPU)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

sys.dont_write_bytecode = True
import tempfile
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
import scipy

from scripts.maintainer.golden_support import (
    GOLDEN,
    campaign_score,
    candidate_score,
    online_score,
    online_snapshot,
    online_stream_sha256,
    read,
    reference_bundle,
    sha,
    unpack,
    write,
)
from scripts.maintainer.gpu_identity import reference_contract
from tests.characterization.golden_compare import (
    assert_equivalent,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-root", type=Path, help="Public golden folder or staged asset root"
    )
    parser.add_argument("--root", type=Path, default=GOLDEN)
    parser.add_argument(
        "--reference-root",
        "--archive-root",
        type=Path,
        help="Frozen reference checkout from the private archive",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=None,
        help="Golden manifest and snapshots (independent of reference sources)",
    )
    parser.add_argument("--candidate-scorer", choices=("bench_2026_11",), default="bench_2026_11")
    parser.add_argument(
        "--receipt", type=Path, help="Write a deterministic, source-root-independent receipt"
    )
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="Allow missing archive evidence (still reported as skipped)",
    )
    args = parser.parse_args()
    from scripts.maintainer.archive_root import ArchiveUnavailable, archive_bundle, golden_root

    try:
        bundle = archive_bundle(args.reference_root)
    except ArchiveUnavailable as error:
        result = dict(passed=False, skipped=True, reason=str(error))
        if args.receipt:
            write(args.receipt, result)
        print(json.dumps(result))
        return 0 if args.allow_skip else 77
    os.environ["TOWER_ARCHIVE_ROOT"] = str(bundle)
    # Original corpus metadata is archive evidence; public metadata only renames IDs.
    args.fixture_root = args.fixture_root or golden_root(args.reference_root)
    if (
        sha(args.fixture_root / "MANIFEST.json")
        != reference_contract()["reference_identity"]["corpus_manifest_sha256"]
    ):
        raise RuntimeError("Golden corpus manifest identity differs")
    manifest = read(args.fixture_root / "MANIFEST.json")
    expected = manifest["exact_environment"]
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Exact tier requires the archived Linux x86-64 numerical environment")
    if platform.python_version() != "3.12.3" or platform.libc_ver() != ("glibc", "2.39"):
        raise RuntimeError("Exact tier requires Python 3.12.3 and glibc/libm 2.39")
    for name, actual in (
        ("numpy", np.__version__),
        ("scipy", scipy.__version__),
        ("h5py", h5py.__version__),
    ):
        if actual != expected[name]:
            raise RuntimeError(f"Exact tier requires {name}={expected[name]}, found {actual}")
    receipts = []
    for trial in manifest["trials"]:
        path = args.root / trial["archive"]
        if not path.is_file():
            path = args.fixture_root / trial["archive"]
        if path.stat().st_size != trial["size"] or sha(path) != trial["sha256"]:
            raise RuntimeError("Golden archive identity differs")
        original_archive_sha256 = sha(path)
        if args.golden_root is not None:
            from scripts.maintainer.fetch_golden import golden_archive

            path = golden_archive(trial["archive"], args.golden_root)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            unpack(path, run)
            assert_equivalent(
                read(run / "score.json"),
                campaign_score(run, trial["task"], reference_root=args.reference_root),
                portable=False,
            )
            assert_equivalent(
                read(run / "score.json"),
                candidate_score(run, trial["task"], implementation=args.candidate_scorer),
                portable=False,
            )
            snapshot = online_snapshot(trial["id"], fixture_root=args.fixture_root)
            if "archived_stream_sha256" in snapshot:
                archived_digest = snapshot.pop("archived_stream_sha256")
                if online_stream_sha256(snapshot["events"]) != archived_digest:
                    raise RuntimeError(f"Stored online stream digest differs: {trial['id']}")
            assert_equivalent(snapshot, online_score(run), portable=False)
            receipts.append(
                dict(
                    trial=trial["id"],
                    archive_sha256=sha(path),
                    original_archive_sha256=original_archive_sha256,
                    score_sha256=canonical_sha(read(run / "score.json")),
                    online_sha256=canonical_sha(snapshot),
                )
            )
        print(f"EXACT PASS {trial['id']} frozen+library+online", flush=True)
    if args.receipt:
        write(
            args.receipt,
            dict(
                schema="tower-golden-exact-v2",
                reference=manifest["reference"],
                source_manifest_sha256=sha(
                    reference_bundle(args.reference_root) / "SOURCE_MANIFEST.json"
                ),
                corpus_manifest_sha256=sha(args.fixture_root / "MANIFEST.json"),
                candidate_scorer=args.candidate_scorer,
                trials=receipts,
            ),
        )
    return 0


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
