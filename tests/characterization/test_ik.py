"""Appendix G: separate processes prevent native ABI/import-name collisions.

Every joint output has a 1e-9 rad absolute tolerance and zero relative tolerance.
The two FK implementations use 1e-9 for matrix elements. These tests run on
every platform with the required native dependencies.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.maintainer.golden_support import GOLDEN, ROOT, read

NATIVE = ("package", "standalone", "vendor", "posture-ipopt", "pink")


@pytest.mark.parametrize(
    "library", [pytest.param(False, marks=pytest.mark.archive), True], ids=["original", "library"]
)
@pytest.mark.parametrize("kind", ["numpy-dls", "sdk-mdh", *NATIVE])
def test_ik_golden(kind: str, library: bool, tmp_path: Path) -> None:
    if kind in NATIVE and os.environ.get("TOWER_REQUIRE_NATIVE_IK") != "1":
        pytest.skip(
            "Native Pinocchio/CasADi goldens run in native-eef; set TOWER_REQUIRE_NATIVE_IK=1"
        )
    if kind in NATIVE:
        assert importlib.util.find_spec("pinocchio") is not None, "native-eef requires Pinocchio"
    if not library:
        from scripts.maintainer.archive_root import archive_bundle

        archive_bundle()
    output = tmp_path / "actual.json"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("ik_scenarios.py")), kind, str(output)]
        + (["--library"] if library else []),
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    expected, actual = read(GOLDEN / ("ik-" + kind + ".json")), read(output)
    assert actual["implementation"] == expected["implementation"]
    assert len(actual["cases"]) == len(expected["cases"])
    for left, right in zip(expected["cases"], actual["cases"], strict=True):
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_allclose(
                right[key], left[key], atol=1e-9, rtol=0, err_msg=f"{kind}/{key}"
            )
