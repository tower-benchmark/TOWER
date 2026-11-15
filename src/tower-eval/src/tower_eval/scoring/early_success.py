# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for campaign row construction and online latches."""

from .detectors.early_success import FourPlaceHoldStop as FourPlaceHoldStop
from .detectors.early_success import Task2HoldStop as Task2HoldStop
from .detectors.early_success import TaskAwareHoldStop as TaskAwareHoldStop
from .detectors.early_success import score_recording as score_recording
from .recording import from_physical_rows as from_physical_rows
