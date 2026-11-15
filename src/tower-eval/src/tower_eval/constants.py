# SPDX-License-Identifier: Apache-2.0
"""Cadences and dimensions in SI units; no backend or environment discovery.

The policy runs at 10 Hz. Legacy controller/recording samples remain at 30 Hz;
that cadence is deliberately distinct from the 400 Hz benchmark physics clock.
"""

CONTROL_HZ = 10
PHYSICS_HZ = 400
ACTION_DT = 1 / CONTROL_HZ
POLICY_PERIOD_TICKS = PHYSICS_HZ // CONTROL_HZ
CONTROLLER_HZ = 30
CONTROLLER_DT = 1 / CONTROLLER_HZ
BLOCK_SIZE_M = (0.15, 0.05, 0.03)
GRIPPER_CHANNELS = (7, 15)

# Catalogue for the editable evaluator surface. Scene, TCP and scoring schemas
# remain defined in their respective modules.
_SCHEMAS = {
    "config": "tower-eval-v1",
    "run": "tower-eval-run-v1",
    "report": "tower-eval-report-v1",
    "result": "tower-result-v1",
    "action_chunk": "tower-action-chunk-v1",
    "dual_action": "tower-absolute-dual-v1",
}
