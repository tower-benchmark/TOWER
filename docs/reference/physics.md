# Physics and gripper coupling

Opposite-signed finger coordinates obey `finger_joint2 + finger_joint1 = 0`
when physical coupling is enabled
(simulator setting `coupled_gripper_fingers = true`, default `false`). Contact stops closure against an object;
the constraint removes sideways jaw translation without attaching the block.
Small settling and solver errors remain possible. Coupling is not a hardware
calibration claim. Preserve the released physics and controller settings for
comparable results; see [simulator setup](../isaac-sim.md).

See the [control safety contract](../guides/eef-control.md#control-safety-contract).
