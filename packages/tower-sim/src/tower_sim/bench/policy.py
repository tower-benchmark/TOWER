"""Ordinary and MME policy input/output at the shared lifecycle's fixed ticks.

The adapter is the only HTTP transport. No reset RPC, client RNG, simulator
imports or module-global patches: inference seed 42 remains server-owned.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from tower_eval.policies.legacy_predict import (
    LegacyPredictAdapter,
    PolicyOutputError,
    PolicyTransportError,
)

from . import control
from .mme_history import EpisodeHistory
from .resources import save


class OrdinaryPolicy:
    stage = "moving_policy_diagnostic"
    inference = "concurrent per-environment requests; physics waits for its fixed-tick actions"

    def __init__(self, *, package, url, prompt, task, metrics, solver, adapter=None):
        self.package = dict(package)
        self.prompt, self.task, self.metrics, self.solver = prompt, task, metrics, solver
        self.adapter = adapter or LegacyPredictAdapter(
            url, expected_identity={"package_digest": self.package["package_digest"]}
        )
        self.executor = None
        self.histories = []
        self.rows = []

    def authorize(self):
        """Startup package authorization and first full-identity snapshot (C21)."""
        return self.adapter.health()

    def start(self, output):
        # Frozen entry + worker each read health. Reuse ONE adapter so predictions
        # must match BOTH snapshots, even if same-package health drifts in between.
        health = self.adapter.health()
        save(output / "policy-health.json", health)
        return health

    def initialize(self, rows, camera_names, health):
        self.rows = rows
        self.executor = ThreadPoolExecutor(max_workers=min(16, len(rows)))

    def append_camera(self, index, archive, tick, *, rejected, terminal):
        pass

    def payload(self, index, observation, archive, tick):
        return dict(observation, images=archive.latest, prompt=self.prompt)

    def prefetch(self, observations, archives, tick):
        # Materialize validated PNG/history on the simulation thread before any
        # RPC. Workers only consume owned payloads; physics waits for all actions.
        payloads = {
            index: self.payload(index, observation, archives[index], tick)
            for index, observation in observations.items()
        }
        if self.executor is None:
            raise RuntimeError("Policy executor has not been initialized")
        return {
            index: self.executor.submit(
                self.metrics.predict,
                self.adapter,
                payload,
                headers=self.metrics.headers(self.rows[index]["candidate"]["seed"], tick),
            )
            for index, payload in payloads.items()
        }

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
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        self.adapter.close()


class MMEPolicy(OrdinaryPolicy):
    def initialize(self, rows, camera_names, health):
        self.histories = [
            EpisodeHistory(
                str(row["output"].resolve()), camera_names, health["identity"]["position_frames"]
            )
            for row in rows
        ]
        super().initialize(rows, camera_names, health)

    def append_camera(self, index, archive, tick, *, rejected, terminal):
        self.metrics.call(
            "history.append_validated",
            self.histories[index].append_validated,
            archive,
            tick,
            rejected=rejected,
            terminal=terminal,
        )

    def payload(self, index, observation, archive, tick):
        return dict(
            super().payload(index, observation, archive, tick),
            task=self.task,
            episode_id=self.histories[index].episode_id,
            observation_time_s=tick / 400,
            history=self.metrics.call("history.pack_request", self.histories[index].request, tick),
        )

    def finalize(self, index):
        self.histories[index].close()
        save(self.rows[index]["output"] / "mme-history.json", self.histories[index].receipt())

    def close(self):
        super().close()
        for history in self.histories:
            history.close()


def from_package(**kwargs):
    cls = MMEPolicy if kwargs["package"].get("kind") == "MME-VLA" else OrdinaryPolicy
    return cls(**kwargs)
