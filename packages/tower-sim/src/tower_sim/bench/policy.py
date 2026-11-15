"""Ordinary and MME policy input/output at the shared lifecycle's fixed ticks.

One WebSocket connection serves active environments sequentially.
Inference seed 42 remains server-owned.
"""

from __future__ import annotations

from concurrent.futures import Future

from tower_eval.policies.runtime_ws import (
    PolicyOutputError,
    PolicyTransportError,
    RuntimePolicyAdapter,
)

from . import control
from .resources import save


class OrdinaryPolicy:
    stage = "moving_policy_diagnostic"
    inference = "sequential per-environment WebSocket calls; physics waits for fixed-tick actions"

    def __init__(
        self, *, package, url, prompt, task, metrics, solver, adapter=None, client_root=None
    ):
        self.package = dict(package)
        self.prompt, self.task, self.metrics, self.solver = prompt, task, metrics, solver
        self.adapter = adapter or RuntimePolicyAdapter(
            url,
            expected_identity={"package_digest": self.package["package_digest"]},
            root=client_root,
        )
        self.rows = []

    def authorize(self):
        """Startup package authorization and first full-identity snapshot (C21)."""
        return self.adapter.health()

    def start(self, output):
        # Entry and worker share one connection and pin both full identity snapshots.
        health = self.adapter.health()
        save(output / "policy-health.json", health)
        return health

    def initialize(self, rows, camera_names, health):
        self.rows = rows
        trial_id = getattr(self.metrics.request_id, "trial", None)
        self.adapter.reset(
            seed=rows[0]["candidate"]["seed"],
            metadata={
                **({"trial_id": trial_id} if trial_id is not None else {}),
                "environment_seeds": [row["candidate"]["seed"] for row in rows],
                "task": self.task,
            },
        )

    def append_camera(self, index, archive, tick, *, rejected, terminal):
        pass

    def payload(self, index, observation, archive, tick):
        return dict(observation, images=archive.latest, prompt=self.prompt, env_idx=index)

    def prefetch(self, observations, archives, tick):
        # Materialize validated PNGs before sequential exchanges.
        # Physics waits for all actions, including per-environment rejections.
        payloads = {
            index: self.payload(index, observation, archives[index], tick)
            for index, observation in observations.items()
        }
        pending = {}
        for index in sorted(payloads):
            future = Future()
            try:
                future.set_result(
                    self.metrics.predict(
                        self.adapter,
                        payloads[index],
                        headers=self.metrics.headers(self.rows[index]["candidate"]["seed"], tick),
                    )
                )
            except (PolicyOutputError, PolicyTransportError, ValueError, OSError) as exc:
                future.set_exception(exc)
            pending[index] = future
        return pending

    def prediction(self, pending, index, observation, tick):
        output = self.rows[index]["output"]
        try:
            with self.metrics.span("policy.rpc_barrier_wait"):
                return pending[index].result()
        except (PolicyOutputError, PolicyTransportError) as exc:
            if exc.body is not None and getattr(exc, "body_is_json", True):
                save(output / "policy-error.json", exc.body)
            if isinstance(exc, PolicyOutputError):
                if exc.body == {
                    "code": "invalid_observation",
                    "error": "Negative gripper aperture",
                }:
                    save(
                        output / "policy-input-rejection.json",
                        dict(
                            observation_time_s=tick / 400,
                            observation=observation,
                            service_error=exc.body,
                            rejected_action_applied=False,
                            observation_clipped=False,
                            numerical_tolerance_changed=False,
                        ),
                    )
                    raise control.PolicyActionError(
                        "Frozen aperture observation gate rejected input: " + str(exc.body)
                    ) from exc
                raise control.PolicyActionError(str(exc.body)) from exc
            raise RuntimeError(f"Policy transport/input failure: {exc.body}") from exc

    def finalize(self, index):
        pass

    def close(self):
        self.adapter.close()


class MMEPolicy(OrdinaryPolicy):
    def authorize(self):
        raise ValueError(
            "MME-VLA needs an XPolicyLab WebSocket adapter; not available in this release"
        )


def from_package(**kwargs):
    cls = MMEPolicy if kwargs["package"].get("kind") == "MME-VLA" else OrdinaryPolicy
    return cls(**kwargs)
