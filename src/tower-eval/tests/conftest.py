# SPDX-License-Identifier: Apache-2.0
"""Shared deterministic simulator for controller and rollout tests."""

import pytest
from tower_eval_fakes import FakeSim


@pytest.fixture
def fake_sim():
    return FakeSim()
