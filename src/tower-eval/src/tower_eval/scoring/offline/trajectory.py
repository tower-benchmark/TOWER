# SPDX-License-Identifier: Apache-2.0
"""Historical paper-v1 API; implementation lives in the shared scoring tree."""

from ..detectors.common import first as first
from ..detectors.common import spans as spans
from ..detectors.common import sustained as sustained
from ..detectors.disturbance import disturbance_events as disturbance_events
from ..detectors.disturbance import renewed_contact as renewed_contact
from ..detectors.disturbance import target_intervals as target_intervals
from ..detectors.grasp import carry_mask as carry_mask
from ..detectors.grasp import hold_mask as hold_mask
from ..detectors.place import placement_geometry as placement_geometry
from ..detectors.place import stable_dwell as stable_dwell
from ..detectors.place import support_mask as support_mask
from ..engine import score_recording as score_recording
from ..geometry import angular_error as angular_error
from ..geometry import area as area
from ..geometry import boxes_touch as boxes_touch
from ..geometry import corners as corners
from ..geometry import fully_clear as fully_clear
from ..geometry import geometry as _block_geometry
from ..geometry import inside as inside
from ..geometry import intersection as intersection
from ..geometry import inverse as inverse
from ..geometry import polygon as polygon
from ..geometry import rotate as rotate
from ..recording import Recording as Recording
from ..recording import digest as digest
from ..recording import load_recording as load_recording
from ..rules import DEFAULT_RULES as DEFAULT_RULES
from ..rules import FOUR_PLACE_HOLD as FOUR_PLACE_HOLD
from ..rules import PROTOCOL as PROTOCOL
from ..rules import Rules as Rules
from ..rules import task_counts as task_counts


def score_run(run, task, rules=DEFAULT_RULES):
    task_counts(task)
    return score_recording(load_recording(run, rules), task, rules)


block_geometry = _block_geometry
