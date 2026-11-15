# Observations and actions

Policies receive RGB images, encoder joints, TCP transforms, gripper apertures
and an instruction. Actions explicitly identify joint or EEF mode, units,
coordinate frame and timing. J2 conversion is performed once at the control
boundary. See the [complete interface](../reference/policy-interface.md) and
[control safety](../guides/eef-control.md).
