"""Episode outcomes, separate from process completion and batch throughput."""


def outcome_fields(status, *, returncode=0):
    """Unknown errors/administrative stops are invalid, never policy failures.

    A task/safety failure must originate in a detector with recorded evidence;
    this mapping does not infer failure from a stall or a process exit.
    """
    if status == 'not_started':
        return dict(episode_result=None, valid_trial=False, task_success=False)
    outcome = {
        'task_success': 'Task Success',
        'task_failure': 'Task Failure',
        'duration_reached': 'Timeout Failure',
        'step_limit_reached': 'Timeout Failure',
        'safety_abort': 'Safety Abort',
    }.get(status, 'Invalid Trial') if returncode == 0 else 'Invalid Trial'
    return dict(episode_result=outcome, valid_trial=outcome != 'Invalid Trial',
                task_success=outcome == 'Task Success')
