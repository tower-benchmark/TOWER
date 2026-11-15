"""Native SharedIK equivalence probe, executed only in an isolated interpreter."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from nero_kinematics.solvers.shared import SharedIK
from tower_sim.bench.ik_client import IKServiceClient, IKServiceOptions
from tower_sim.bench.ik_service import SCHEMA, solve_request


def compare(actual, expected):
    """Compare all diagnostics except wall-clock duration, in unchanged units."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (actual, expected)
        for key in expected:
            if key == "latency_s":
                assert np.isfinite(actual[key]) and actual[key] >= 0
                assert np.isfinite(expected[key]) and expected[key] >= 0
            else:
                compare(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            compare(a, b)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
    else:
        assert actual == expected, (actual, expected)


def main(*, reference=False):
    cases = json.loads(
        (Path.cwd() / "tests/golden/benchmark-2026.11/ik-posture-ipopt.json").read_text()
    )["cases"]
    solver = SharedIK()
    instances = [solver]
    if reference:
        from tests.characterization.ik_scenarios import RUNTIME, module_file

        # Only the unused physical-conversion dependency is stubbed: encoder IK
        # runs the complete frozen SharedIK and controlled_solvers algorithms.
        sys.modules["kin"] = module_file("kin", RUNTIME / "ik-capsule/tools/kin.py")
        sys.modules["j2comp"] = SimpleNamespace()
        module_file("controlled_solvers", RUNTIME / "shared/controlled_solvers.py")
        instances.insert(0, module_file("_p705_shared", RUNTIME / "shared/shared_ik.py").SharedIK())

    with IKServiceClient(IKServiceOptions(Path(sys.executable))) as client:
        for case in cases:
            # Exercise both arm-specific postures through the library and the
            # real service. The migration variant also compares frozen code.
            target, seed = np.array(case["target"]), np.array(case["seed"])
            expected = []
            for arm in (0, 1):
                actual = solver.encoder(target, seed, arm)
                assert actual["accepted"], actual
                assert {
                    "reason",
                    "step_rad",
                    "position_error_m",
                    "rotation_error_rad",
                } <= actual.keys()
                expected_result = actual
                if reference:
                    frozen = instances[0].encoder(target, seed, arm)
                    assert frozen["accepted"], frozen
                    compare(actual, frozen)
                    expected_result = frozen
                expected.append(expected_result)
            result = client.solve([[target, target]], [seed, seed])
            assert result["accepted"], result
            for arm, diagnostic in enumerate(result["diagnostics"]):
                compare(
                    diagnostic,
                    dict(row=0, arm=arm, variant="posture_ipopt_v5_shared", **expected[arm]),
                )
                compare(result["joints"][0][arm], expected[arm]["joints"])

        # Accept two full rows, then reject arm 1 in row 2. Arm 0 has already
        # solved that row; row 3 must not execute, and no partial joints escape.
        pair = np.array([cases[0]["target"], cases[1]["target"]])
        seed = np.array([cases[0]["seed"], cases[1]["seed"]])
        targets = np.tile(pair, (4, 1, 1, 1))
        targets[2, 1, :3, :3] = targets[2, 1, :3, :3] @ np.diag([1.0, -1.0, -1.0])
        req = dict(schema=SCHEMA, request_id="chunk", dt_s=0.1, poses=targets, seed=seed)
        outcomes, traces = [], []
        for instance in instances:
            calls = []

            def encoder(target, current, arm, calls=calls, instance=instance):
                calls.append((target.copy(), current.copy(), arm))
                return instance.encoder(target, current, arm)

            outcomes.append(
                solve_request(req, SimpleNamespace(encoder=encoder, identity=instance.identity))
            )
            traces.append(calls)
        outcomes.append(client.solve(targets, seed))
        baseline = outcomes[0]
        for result in outcomes:
            assert not result["accepted"] and result["joints"] is None
            assert len(result["diagnostics"]) == 6, result
            assert all(d["accepted"] for d in result["diagnostics"][:5])
            assert not result["diagnostics"][-1]["accepted"]
            compare(result["diagnostics"], baseline["diagnostics"])
        for calls, result in zip(traces, outcomes):
            assert len(calls) == 6
            for index, (target, current, arm) in enumerate(calls):
                row = index // 2
                np.testing.assert_array_equal(target, targets[row, arm])
                prior = (
                    seed[arm] if row == 0 else result["diagnostics"][(row - 1) * 2 + arm]["joints"]
                )
                np.testing.assert_array_equal(current, prior)
        assert not np.array_equal(traces[0][2][1], seed[0])

        outside = np.full((2, 7), 10.0)
        req.update(seed=outside)
        for result in [
            *(solve_request(req, instance) for instance in instances),
            client.solve(targets, outside),
        ]:
            assert not result["accepted"] and result["joints"] is None
            assert len(result["diagnostics"]) == 1
            assert result["diagnostics"][0]["reason"] == "seed_outside_kill_line"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true")
    main(reference=parser.parse_args().reference)
