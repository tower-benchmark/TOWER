# SPDX-License-Identifier: Apache-2.0
"""Require the caller's explicit consent before starting NVIDIA Isaac Sim."""

import os
from collections.abc import MutableMapping

EULA_URL = (
    "https://docs.omniverse.nvidia.com/platform/latest/common/"
    "NVIDIA_Omniverse_License_Agreement.html"
)


def require_eula(
    accept_eula: bool = False, *, environment: MutableMapping[str, str] | None = None
) -> None:
    """Only an explicit flag may write acceptance into the worker environment."""
    environment = os.environ if environment is None else environment
    if accept_eula:
        environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    if environment.get("OMNI_KIT_ACCEPT_EULA", "").strip().lower() not in {"yes", "y", "1"}:
        raise RuntimeError(
            f"Isaac Sim requires explicit acceptance of the NVIDIA Omniverse License Agreement: {EULA_URL}. "
            "After reviewing it, set OMNI_KIT_ACCEPT_EULA=YES or pass --accept-eula."
        )
