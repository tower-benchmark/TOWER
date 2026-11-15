from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from scripts.maintainer.golden_support import GOLDEN, golden_archive, read, sha, unpack


def pytest_generate_tests(metafunc: Any) -> None:
    if "trial" in metafunc.fixturenames:
        rows = read(GOLDEN / "MANIFEST.json")["trials"]
        metafunc.parametrize("trial", rows, ids=[row["id"] for row in rows], scope="module")


@pytest.fixture(scope="module")
def recording(trial: dict[str, Any]) -> Iterator[Path]:
    path = golden_archive(trial["archive"])
    assert path.stat().st_size == trial["size"] and sha(path) == trial["sha256"]
    with tempfile.TemporaryDirectory(prefix="tower-golden-") as directory:
        unpack(path, Path(directory))
        yield Path(directory)


# Under pytest-xdist each worker keeps its own float-difference statistics. Workers
# send them to the controller, which reports the combined totals.
_WORKER_STATS: list[dict[str, float]] = []


def pytest_sessionfinish(session: Any) -> None:
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:
        from dataclasses import asdict

        from tests.characterization.golden_compare import STATS

        workeroutput["golden_float_stats"] = asdict(STATS)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: Any) -> None:
    stats = getattr(node, "workeroutput", {}).get("golden_float_stats")
    if stats is not None:
        _WORKER_STATS.append(stats)


def pytest_terminal_summary(terminalreporter: Any) -> None:
    from tests.characterization.golden_compare import STATS, summary

    for stats in _WORKER_STATS:
        STATS.differences += int(stats["differences"])
        STATS.max_abs_delta = max(STATS.max_abs_delta, stats["max_abs_delta"])
        STATS.max_rel_delta = max(STATS.max_rel_delta, stats["max_rel_delta"])
    terminalreporter.write_line(summary())
