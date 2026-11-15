# Benchmark user quickstart

1. Follow [installation](install.md) and run `tower smoke` in the evaluator environment.
2. Set up [Isaac](../isaac-sim.md) and the [asset download steps](../guides/evaluate-a-policy.md#download-and-verify-assets).
3. Serve your policy over the HTTP `/health` + `/predict` protocol
   ([HTTP compatibility](../reference/policy-interface.md#http-compatibility)).
   XPolicyLab WebSocket serving applies to `tower run` and in-process integrations;
   the Isaac benchmark runtime does not consume it yet.
4. Follow [evaluate a policy](../guides/evaluate-a-policy.md) to validate the request and launch.
5. Keep the resolved configuration, scene and policy identities with results.
   Use [offline scoring](../guides/offline-scoring.md) for saved recordings.

CPU smoke output is not a task success result. Weight downloads remain unavailable
until the owner publishes them; you can supply your own compatible policy.
