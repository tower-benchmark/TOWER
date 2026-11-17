"""Explicit historical scene evidence for compatibility tests."""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--long-replay-first",
        action="store_true",
        help="Start the measured longest public campaign replay before shorter tests",
    )
    parser.addoption("--asset-root", type=Path, help="Public asset folder or staged upload root")
    parser.addoption("--golden-root", type=Path, help="Public golden folder or staged asset root")
    parser.addoption(
        "--v1-root", "--archive-root", type=Path, help="Explicit private archive export"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--long-replay-first"):
        # Profiled on 2026-11-17: this replay dominated the public suite. Revisit
        # the hint using CI's duration reports if the test or workload changes.
        # Start it immediately with a small queue behind it (xdist load keeps
        # at least two initial items per worker). Keep every item and assertion.
        longest = (
            "packages/tower-sim/tests/test_bench_scoring.py::"
            "test_full_campaign_streams[task1-3-openwam-public]"
        )
        items.sort(key=lambda item: item.nodeid != longest)


@pytest.fixture(scope="session")
def v1_source(pytestconfig):
    from scripts.maintainer.archive_scene import V1Source

    explicit = pytestconfig.getoption("--v1-root")
    if explicit is not None:
        return V1Source(Path(__file__).parent, explicit)
    return V1Source(Path(__file__).parent)


@pytest.fixture(scope="session")
def v1_export(v1_source, tmp_path_factory):
    from scripts.maintainer.archive_scene import V1_PREFIX, v1_scene_files

    root = tmp_path_factory.mktemp("v1-scene-export")
    for name, expected in v1_scene_files().items():
        path = root / name.removeprefix(V1_PREFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(v1_source.read(name, expected))
    return root


def pytest_configure(config):
    import os

    from scripts.maintainer.archive_root import archive_bundle

    assets = config.getoption("--asset-root")
    if assets is not None:
        os.environ["TOWER_ASSET_ROOT"] = str(assets)

    golden = config.getoption("--golden-root")
    if golden is not None:
        os.environ["TOWER_GOLDEN_ROOT"] = str(golden)
    explicit = config.getoption("--v1-root")
    if explicit is not None:
        archive_bundle(explicit)
        os.environ["TOWER_ARCHIVE_ROOT"] = str(explicit)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    import os

    from scripts.maintainer.archive_root import ArchiveUnavailable
    from scripts.maintainer.fetch_golden import GoldenUnavailable
    from scripts.maintainer.fetch_public_test_data import PublicAssetsUnavailable

    outcome = yield
    report = outcome.get_result()
    explicit = item.config.getoption("--v1-root") or os.environ.get("TOWER_ARCHIVE_ROOT")
    if (
        call.excinfo
        and isinstance(call.excinfo.value, ArchiveUnavailable)
        and item.get_closest_marker("archive")
        and not explicit
    ):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] + 1, str(call.excinfo.value))
    if (
        call.excinfo
        and isinstance(call.excinfo.value, PublicAssetsUnavailable)
        and item.get_closest_marker("public_assets")
        and os.environ.get("TOWER_REQUIRE_PUBLIC_DATA") != "1"
    ):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] + 1, str(call.excinfo.value))
    if (
        call.excinfo
        and isinstance(call.excinfo.value, GoldenUnavailable)
        and item.get_closest_marker("golden")
        and os.environ.get("TOWER_REQUIRE_PUBLIC_DATA") != "1"
    ):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] + 1, str(call.excinfo.value))
    if explicit and report.skipped and "SKIP: supply --archive-root" in str(report.longrepr):
        report.outcome = "failed"
        report.longrepr = "An explicit archive run must have zero archive skips"


def pytest_terminal_summary(terminalreporter):
    count = sum(
        "SKIP: supply --archive-root" in str(report.longrepr)
        for report in terminalreporter.stats.get("skipped", ())
    )
    terminalreporter.write_line(f"Archive evidence skips: {count}")

    public_count = sum(
        "SKIP: public " in str(report.longrepr)
        for report in terminalreporter.stats.get("skipped", ())
    )
    terminalreporter.write_line(f"Public data skips: {public_count}")
