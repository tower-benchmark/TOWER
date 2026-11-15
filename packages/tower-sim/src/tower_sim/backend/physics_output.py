# SPDX-License-Identifier: Apache-2.0
"""Physics state-export policy, independent of controller and solver settings.

Only camera-free rollouts may skip USD pose/velocity writes. Tensor readback
stays enabled. Install after new_stage(), initialize physics normally, then
activate the policy after the existing reset/settle/FK initialization.
"""

USD_OUTPUT_KEYS = ("/physics/updateToUsd", "/physics/updateVelocitiesToUsd")
READBACK_KEY = "/physics/suppressReadback"


class PhysicsOutputPolicy:
    def __init__(self, settings, *, render: bool, mode: str = "auto"):
        if mode not in ("auto", "on"):
            raise ValueError("physics-usd-writeback must be auto or on")
        self.settings, self.render, self.mode = settings, render, mode
        self.ready = False
        self.expected_writeback = True
        self.report = {
            "schema": "jenga_physics_output_v1",
            "requested_mode": mode,
            "render_enabled": render,
            "initial_settings": self._read(),
            "initialization_writeback_enabled": True,
            "physics_or_controller_parameters_changed": False,
        }
        # These flags belong to the stage. Restore them AFTER new_stage() so
        # a previous headless environment cannot leave the renderer stale.
        self._set_writeback(True)
        # SimulationContext has not been constructed yet. It initializes
        # suppressReadback itself; do not assume that setting exists already.
        # All operational checks and finish_initialization remain strict.
        self._verify(dict.fromkeys(USD_OUTPUT_KEYS, True))
        self.report["initialization_settings"] = self._read()

    def _read(self) -> dict:
        return {key: self.settings.get(key) for key in (*USD_OUTPUT_KEYS, READBACK_KEY)}

    def _set_writeback(self, enabled: bool) -> None:
        for key in USD_OUTPUT_KEYS:
            if self.settings.get(key) != enabled:
                self.settings.set(key, enabled)
        self.expected_writeback = enabled

    def finish_initialization(self, *, camera_count: int) -> None:
        if self.ready:
            raise RuntimeError("Physics output policy is already active")
        if not self.render and camera_count:
            raise RuntimeError("Camera-free physics output policy cannot own cameras")
        self.verify()
        enabled = self.render or self.mode == "on"
        self._set_writeback(enabled)
        self.verify()
        self.ready = True
        self.report.update(
            active_settings=self._read(),
            headless_usd_export_optimized=not enabled,
            applied_after_reset_settle_and_fk=True,
        )

    def verify(self) -> None:
        expected = {**dict.fromkeys(USD_OUTPUT_KEYS, self.expected_writeback), READBACK_KEY: False}
        self._verify(expected)

    def _verify(self, expected: dict) -> None:
        actual = {key: self.settings.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"Physics output settings changed: expected {expected}, got {actual}"
            )

    def verify_capture(self) -> None:
        if not self.ready or not self.render or not self.expected_writeback:
            raise RuntimeError("Camera capture requires initialized, USD-updated rendered physics")
        self.verify()

    def verify_tensor_views(self, *views) -> None:
        if not self.ready or self.expected_writeback:
            return
        # RigidPrim can silently fall back to USD when its physics handle is
        # invalid. USD is intentionally stale here; never use that fallback.
        for view in views:
            try:
                valid = view.is_physics_handle_valid()
            except (AttributeError, RuntimeError) as error:
                raise RuntimeError(
                    "Optimized headless execution requires live physics tensor handles"
                ) from error
            if not valid:
                raise RuntimeError(
                    "Optimized headless execution requires live physics tensor handles"
                )
