# SPDX-License-Identifier: Apache-2.0
"""Deprecated configuration spellings must not change dispatched commands."""

import warnings

import numpy as np
import pytest
from tower_eval.config import EvaluationConfig, load_config
from tower_eval.controllers import Controller

from tests.characterization.controller_scenarios import ScriptedBackend, plain


@pytest.mark.parametrize("fraction", [0.01, 0.2, 1.0])
def test_legacy_aliases_warn_and_preserve_command_stream(fraction):
    legacy, canonical = ScriptedBackend(), ScriptedBackend()
    with pytest.warns(FutureWarning) as caught:
        # Positional callers of the legacy fraction remain supported too.
        old = Controller(legacy.robot, 0.8, fraction, gripper_control_profile="measured_fraction")
    assert len(caught) == 2
    assert any("gripper_step_fraction" in str(item.message) for item in caught)
    assert any("direct_absolute" in str(item.message) for item in caught)
    assert old.gripper_step_fraction == fraction
    assert old.gripper_control_profile == "measured_fraction"
    new = Controller(canonical.robot, gripper_control_profile="direct_absolute")
    old_events, new_events = [], []
    for width in (-0.01, 0.12, 0.04):
        target = np.full(16, 0.3)
        target[[7, 15]] = width
        old.execute_target(legacy, target, on_step=lambda **event: old_events.append(plain(event)))
        new.execute_target(
            canonical, target, on_step=lambda **event: new_events.append(plain(event))
        )
    assert old_events == new_events
    assert plain(legacy.state) == plain(canonical.state)


def test_omitted_aliases_preserve_defaults_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        controller = Controller(ScriptedBackend().robot)
        EvaluationConfig()
    assert controller.gripper_step_fraction == 0.2
    assert controller.gripper_control_profile == "measured_fraction"


@pytest.mark.parametrize("override", [False, True])
def test_unused_controller_config_is_rejected(tmp_path, override):
    path = tmp_path / "evaluation.toml"
    path.write_text("[controller]\ngripper_step_fraction = 0.2\n")
    overrides = ["controller.gripper_step_fraction=0.5"] if override else None
    with pytest.raises(
        ValueError, match=r"unused \[controller\] table.*controller.gripper_step_fraction"
    ):
        load_config(path, overrides)


@pytest.mark.parametrize("value", [-1, 0, 1.01])
def test_deprecated_fraction_still_validates(value):
    with pytest.warns(FutureWarning), pytest.raises(ValueError):
        Controller(ScriptedBackend().robot, gripper_step_fraction=value)
