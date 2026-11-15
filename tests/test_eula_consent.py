# SPDX-License-Identifier: Apache-2.0
"""Consent must be established before launcher side effects or worker startup."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
from tower_sim.replay.eula import (
    require_eula,
)


@pytest.mark.parametrize("value", [None, "", "NO", "false", "0", "true"])
def test_absent_or_negative_consent_is_rejected(value):
    environment = {} if value is None else {"OMNI_KIT_ACCEPT_EULA": value}
    before = environment.copy()
    with pytest.raises(
        RuntimeError,
        match="Omniverse License Agreement: https://docs.omniverse.nvidia.com/.*--accept-eula",
    ):
        require_eula(environment=environment)
    assert environment == before


@pytest.mark.parametrize("value", ["YES", "y", "1"])
def test_explicit_environment_consent_is_preserved(value):
    environment = {"OMNI_KIT_ACCEPT_EULA": value}
    require_eula(environment=environment)
    assert environment["OMNI_KIT_ACCEPT_EULA"] == value


def test_explicit_flag_can_grant_consent():
    environment = {"OMNI_KIT_ACCEPT_EULA": "NO"}
    require_eula(True, environment=environment)
    assert environment["OMNI_KIT_ACCEPT_EULA"] == "YES"
