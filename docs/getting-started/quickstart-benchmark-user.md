# Benchmark user quickstart

1. Follow [installation](install.md) and run `tower smoke` in the evaluator environment.
2. Set up [Isaac](../isaac-sim.md) and the [asset download steps](../guides/evaluate-a-policy.md#download-and-verify-assets).
3. Serve your policy over the XPolicyLab WebSocket protocol
   ([Isaac runtime transport](../reference/policy-interface.md#isaac-benchmark-runtime)).
   Use one runtime process per policy server. The Isaac runtime uses atomic
   WebSocket inference and checks full-prediction evidence.
4. Follow [evaluate a policy](../guides/evaluate-a-policy.md) to validate the request and launch.
5. Keep the resolved configuration, scene and policy identities with results.
   Use [offline scoring](../guides/offline-scoring.md) for saved recordings.

CPU smoke output is not a task success result. Weight downloads remain unavailable
until the owner publishes them; you can supply your own compatible policy.
