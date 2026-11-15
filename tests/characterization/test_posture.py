"""Exercise the frozen planner's rejection branch without loading station assets.

Only its unused calibrated IK/compensation imports are stubbed. Candidate
selection and its reverse-grasp filter are the actual frozen implementation.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.maintainer.golden_support import RUNTIME


@pytest.fixture
def planning(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in ("kin", "j2comp", "urdf_fk"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "shared_ik",
        SimpleNamespace(
            SharedIK=object,
            SPEC={},
            SPEC_SHA="fixture",
            LOWER=np.full(7, -1.0),
            UPPER=np.ones(7),
            KILL_LO=np.full(7, -2.0),
            KILL_HI=np.full(7, 2.0),
            headroom=lambda q: np.zeros(7),
        ),
    )
    spec = importlib.util.spec_from_file_location(
        "_p0_planning_guard", RUNTIME / "shared/planning.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.archive
def test_planner_rejects_reversed_grasp(planning: Any) -> None:
    """The frozen planner rejects on the candidate's symmetric flag; the ordinary
    fallback test below carries the geometric (no reversed target) guarantee."""
    with pytest.raises(planning.PlanningRejection, match="180-degree symmetric"):
        planning.tweak(
            np.zeros(3),
            np.diag([1.0, -1.0, -1.0]),
            0.0,
            0.0,
            (None, 0.0),
            {"_v5_candidate": {"symmetric": True}},
        )


@pytest.mark.archive
def test_failed_ordinary_candidates_never_fall_back_to_reverse(
    planning: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated: list[bool] = []
    solved: list[np.ndarray] = []

    def plan(rng: Any, parameters: Any) -> Any:
        return SimpleNamespace(keys={"back": [], "left": []})

    def original(
        trial: Any, arm: str, block: Any, target: Any, parameters: Any, rng: Any, info: Any
    ) -> None:
        generated.append(parameters["_v5_candidate"]["symmetric"])
        trial.keys[arm].append(dict(R=np.eye(3), p=np.zeros(3), label="grasp", grip=("close", 0.1)))

    def rejected(target: Any, seed: Any, **kwargs: Any) -> Any:
        solved.append(target.copy())
        return seed, float("inf"), float("inf")

    generator = SimpleNamespace(
        Plan=plan,
        M=SimpleNamespace(dimensions=lambda name: np.array([0.075, 0.025, 0.015])),
        base_T=lambda arm: np.eye(4),
        tcp_world=lambda point, rotation: np.block(
            [[rotation, point[:, None]], [np.array([[0.0, 0.0, 0.0, 1.0]])]]
        ),
    )
    planning.set_context(generator, {}, {}, {"back": ["block"], "left": []}, {"back": np.zeros(7)})
    monkeypatch.setattr(planning, "ik", rejected)
    with pytest.raises(planning.PlanningRejection, match="no permitted non-flipped grasp"):
        planning.choose_transfer(
            generator,
            original,
            plan(None, None),
            "back",
            None,
            None,
            {},
            np.random.default_rng(7),
            {},
            "block",
        )
    assert generated == [False] * 8
    assert len(solved) == 8
    for target in solved:
        np.testing.assert_array_equal(target[:3, :3], np.eye(3))
    audit = planning.AUDIT[-1]
    assert audit["selected"] is None and len(audit["candidates"]) == 16
    assert all(
        row["errors"][0]["reason"] == "user_rejected_180_degree_symmetric_grasp"
        for row in audit["candidates"][8:]
    )
