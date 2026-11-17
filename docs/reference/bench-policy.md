# Policy route limits

See the [public runtime](bench-public.md), [rendering](rendering.md), and
[policy contract](policy-interface.md).

See the [control safety contract](../guides/eef-control.md#control-safety-contract).

The runtime's exact-flip tripwire is narrower than the posture requirement: it
uses a 1e-6 dot-product tolerance (about 0.08 degrees), does not reject an entire
orientation hemisphere, and is bypassed by joint-absolute commands. A near-half
turn or a tilted half turn can still reach native IK. This guard alone therefore
does not establish task-specific contact or trajectory compliance. Unmodified
EEF targets are attempted once; rejection ends the action without a flipped retry.

Public benchmark runs do not require `--pilot-review`. Recorded TCP posture is
reported in `posture-flags.json`, independently of scores and action mode.
The automatic flag also covers joint commands and near-half-turn grasps; it
does not replace maintainer contact and approach/carry review. See
[posture flags](scoring.md#posture-flags-reported-not-scored).
