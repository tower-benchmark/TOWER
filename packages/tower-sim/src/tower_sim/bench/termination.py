"""Record real terminal evidence after an unexecutable policy action.

An action rejected at t=0 still needs two actual physical samples for the public
trajectory scorer. Hold the last valid target for at most one 20 Hz interval;
never interpolate a fabricated state or issue the rejected action.
"""

from __future__ import annotations

import math


def abort_evidence(
    *, tick, state_ticks, camera_ticks, advance_held_tick, record_state, record_camera
):
    """Callbacks must execute/read the simulator, using integer 400 Hz ticks.

    The caller owns the last valid command and must keep it unchanged in
    advance_held_tick. All failures propagate as infrastructure errors. Returned
    failure and terminal times are distinct when a real evidence tail was needed.
    """
    if type(tick) is not int or not 0 <= tick <= 168000:
        raise ValueError("Invalid failure physics tick")
    for name, values in [("state", state_ticks), ("camera", camera_ticks)]:
        if (
            not values
            or values[0] != 0
            or any(type(v) is not int for v in values)
            or any(b <= a for a, b in zip(values, values[1:]))
            or values[-1] > tick
        ):
            raise ValueError(f"Invalid existing {name} evidence clock")
    failure_tick = tick
    # At any positive time the current state can be the second real sample.
    # Only a failure before the first physics tick requires a held-command tail.
    target = 20 if tick == 0 else tick
    camera_last = camera_ticks[-1]
    next_frame = math.floor(camera_last * 30 / 400) + 1
    for current in range(tick + 1, target + 1):
        advance_held_tick(current)
        if current >= math.ceil(next_frame * 400 / 30):
            record_camera(current)
            camera_last = current
            next_frame += 1
    if state_ticks[-1] < target:
        record_state(target)
    if camera_last < target:
        record_camera(target)
    return dict(
        failure_time_s=failure_tick / 400,
        terminal_time_s=target / 400,
        held_evidence_tail_s=(target - failure_tick) / 400,
        tail_command="last_valid_physical_target",
        terminal_tick=target,
    )


def terminal_evidence(
    task, rows, camera_times, command_times, report, *, tick, failure=None, success=None
):
    """Validate the frozen finalizer's clocks and terminal/scorer agreement.

    Returns only terminal evidence fields; persistence, source authorization,
    archives and command execution remain the runner's responsibility.
    """
    if type(tick) is not int or not 0 < tick <= task.duration_ticks:
        raise ValueError("Invalid terminal physics tick")
    stamp = tick / task.physics_hz
    if (
        len(rows) <= 1
        or len(camera_times) <= 1
        or rows[-1]["time"] != float(camera_times[-1])
        or rows[-1]["time"] != stamp
        or [r["time"] for r in rows] != list(command_times)
    ):
        raise ValueError("Terminal evidence clocks differ")
    if failure and success:
        raise ValueError("Terminal cannot be both failure and early success")
    if success:
        if success["protocol"] != task.success_rule or success["terminal_tick"] != tick:
            raise ValueError("Early success terminal differs")
    elif failure:
        failed = failure["failure_tick"]
        if (
            not 0 <= failed <= tick
            or failure["rejected_action_applied"] is not False
            or tick != (20 if failed == 0 else failed)
        ):
            raise ValueError("Abort evidence tail differs")
    elif tick != task.duration_ticks:
        raise ValueError("Unfailed episode did not reach its deadline")
    if report.get("valid") is not True:
        raise ValueError("Scorer rejected physical recording evidence")
    if success and (report["scores"]["s"] != 1 or report.get("success_time_s", stamp) != stamp):
        raise ValueError("Offline success disagrees with online detector")
    guard = bool(failure and failure.get("category") == "camera_near_plane_guard")
    if not success and task.early_success_stop and not guard and report["scores"]["s"] != 0:
        raise ValueError("Offline success disagrees with online detector")
    scored_success = bool(report["scores"]["s"])
    return dict(
        status="task_success" if success else "task_failure" if failure else "duration_reached",
        terminal_tick=tick,
        simulated_seconds=stamp,
        terminal_reason="task_success"
        if success
        else "camera_near_plane_guard"
        if guard
        else "policy_action_failure"
        if failure
        else "duration_reached",
        task_success=scored_success,
        scoring_revision=report.get("scoring_revision", report["protocol"]),
        scored_outcome="success" if scored_success else "failure",
    )
