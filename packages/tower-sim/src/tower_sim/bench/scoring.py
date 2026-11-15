"""One canonical campaign scorer binding for online checks and final evidence.

The adapter uses the detector implementation module directly, not its legacy
compatibility exports. The caller supplies task/protocol resources explicitly.
"""

from typing import Literal

from tower_eval.scoring import api
from tower_eval.scoring.computation import REFERENCE
from tower_eval.scoring.detectors import early_success
from tower_eval.scoring.recording import from_campaign_rows
from tower_eval.scoring.rules import FOUR_PLACE_HOLD

from .fast_early_stop import CachedComputation
from .tasks import BenchTask


class BenchScorer:
    def __init__(
        self, task: BenchTask, *, mode: Literal["off", "fast", "audit"] = "off", computation=None
    ):
        if mode not in ("off", "fast", "audit"):
            raise ValueError("Early-stop mode must be off, fast or audit")
        self.task, self.mode = task, mode
        self.computation = (
            REFERENCE
            if mode == "off"
            else (computation if computation is not None else CachedComputation())
        )
        self.detector = self._detector(self.computation)
        self.reference_detector = self._detector(REFERENCE) if mode == "audit" else None
        self.audit_checks = 0

    def _detector(self, computation):
        if not self.task.early_success_stop:
            return None
        name = self.task.identity.legacy_id
        if self.task.success_rule == FOUR_PLACE_HOLD:
            return early_success.FourPlaceHoldStop(computation=computation)
        if name == "task2-2":
            return early_success.Task2HoldStop(computation=computation)
        return early_success.TaskAwareHoldStop(name, computation=computation)

    def recording(self, spec, rows, finger_geometry, *, failed=False):
        recording = from_campaign_rows(spec, rows, finger_geometry)
        # Preserve the finalizer's distinct status amendment (not the file loader).
        if failed and self.task.success_rule != FOUR_PLACE_HOLD:
            recording.status["status"] = "task_failure"
        return recording

    def check(self, recording, tick, *, failed=False):
        # The batch loop begins at 10 s even though Task5's detector accepts 2 s.
        # Protocol-disabled tasks cannot stop, even if a supplied detector latched.
        if not self.task.early_success_stop or failed or tick < 4000 or tick % 400:
            return None
        if self.detector is None:
            raise RuntimeError("Missing enabled campaign detector")
        result = self.detector.check(recording, tick, failed=failed)
        if self.reference_detector is not None:
            expected = self.reference_detector.check(recording, tick, failed=failed)
            self.audit_checks += 1
            if result != expected or self.detector.success != self.reference_detector.success:
                raise RuntimeError("fast_early_stop detector audit mismatch")
        return result

    def score(self, recording):
        # Final scoring always uses reference computations, including after a cache hit.
        return api.score_recording(
            recording,
            self.task.identity.id,
            ruleset="bench-2026.11",
            success_rule=self.task.success_rule,
        )
