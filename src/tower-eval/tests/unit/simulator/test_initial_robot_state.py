# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
from tower_eval.simulator import validate_initial_robot_state


def test_reset_positions_validate_without_changing_values():
    robot = SimpleNamespace(
        names=tuple(str(i) for i in range(16)), lower=np.full(16, -2.0), upper=np.full(16, 2.0)
    )
    saved = {"names": robot.names, "qpos": np.linspace(-1, 1, 16).tolist()}
    np.testing.assert_array_equal(validate_initial_robot_state(saved, robot), saved["qpos"])
    for fault in [
        saved | {"names": list(reversed(robot.names))},
        saved | {"qpos": [0] * 15},
        saved | {"qpos": [float("nan")] * 16},
        saved | {"qpos": [3] * 16},
    ]:
        with pytest.raises(ValueError):
            validate_initial_robot_state(fault, robot)
