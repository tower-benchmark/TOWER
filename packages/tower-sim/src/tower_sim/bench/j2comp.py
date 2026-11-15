"""Instance-owned calibrated J2 mapping; finite differences and six inverse iterations unchanged."""

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from nero_kinematics.fk import urdf as urdf_fk

from tower_sim.bench.asset_identity import station_identity

from .exact_j2_cache import CachedFK
from .options import DEFAULT_CONTROL_OPTIONS, ControlOptions

ARMS = ("left", "back")


class J2Compensation:
    """Use a verified scene configuration, never import-time asset or environment I/O.

    Production callers must use from_cache. Direct construction is UNVERIFIED
    and test-only: it checks numerical shape/rates but cannot bind a configuration
    to a physical scene. It establishes no calibration or pilot qualification.
    """

    def __init__(self, configuration, *, options: ControlOptions = DEFAULT_CONTROL_OPTIONS):
        """UNVERIFIED/test-only construction; use from_cache for runtime scenes."""
        self.CONFIG = copy.deepcopy(configuration)
        self.K2 = self.CONFIG["k2_rad_per_Nm_by_arm"]
        self.MASSES = self.CONFIG["mass_model"]
        if not np.isfinite([self.K2[a] for a in ARMS]).all() or self.CONFIG["command_hz"] != 200:
            raise ValueError("Invalid J2 calibration/rate")
        for row in self.MASSES.values():
            if (
                not np.isfinite(row["mass_kg"])
                or row["mass_kg"] < 0
                or np.shape(row["com_m"]) != (3,)
                or not np.isfinite(row["com_m"]).all()
            ):
                raise ValueError("Invalid mass/COM calibration")
        self.urdf_fk = urdf_fk
        self._fk: Callable[[Any, float], dict[str, Any]] = urdf_fk.fk
        self.cache_checks = 0
        if options.exact_j2_cache:
            cached = CachedFK(urdf_fk)
            rng = np.random.default_rng(730517)
            for q in [np.zeros(7), *rng.uniform(-1.5, 1.5, size=(8, 7))]:
                for width in [0.0, 0.05, 0.1]:
                    expected = self.potential(q, width)
                    transforms = cached(q, width)
                    actual = sum(
                        v["mass_kg"]
                        * 9.81
                        * (transforms[n][:3, :3] @ v["com_m"] + transforms[n][:3, 3])[2]
                        for n, v in self.MASSES.items()
                        if n in transforms
                    )
                    if np.asarray(expected).tobytes() != np.asarray(actual).tobytes():
                        raise ValueError("Cached FK changed J2 potential")
                    self.cache_checks += 1
            self._fk = cached

    @classmethod
    def from_cache(
        cls, cache: Path, scene: Path, *, options: ControlOptions = DEFAULT_CONTROL_OPTIONS
    ):
        data = json.loads(Path(cache).read_text())
        with Path(scene).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if station_identity(digest) != data["scene_sha256"]:
            raise ValueError("J2 scene cache does not match the calibrated station")
        return cls(data["configuration"], options=options)

    def potential(self, q, width):
        T = self._fk(q, width)
        return sum(
            v["mass_kg"] * 9.81 * (T[n][:3, :3] @ v["com_m"] + T[n][:3, 3])[2]
            for (n, v) in self.MASSES.items()
            if n in T
        )

    def tau2(self, q, width):
        q = np.asarray(q, float)
        d = np.zeros(7)
        d[1] = 1e-05
        return -(self.potential(q + d, width) - self.potential(q - d, width)) / 2e-05

    def delta2(self, q, width, arm="back"):
        return self.K2[arm] * self.tau2(q, width)

    def physical_from_encoder(self, q, width, arm="back"):
        p = np.array(q, float)
        p[1] += self.delta2(q, width, arm)
        return p

    def encoder_from_physical(self, q, width, arm="back", iterations=6):
        p = np.array(q, float)
        for _ in range(iterations):
            p[1] = q[1] - self.delta2(p, width, arm)
        return p

    def physical_from_encoder16(self, q16):
        p = np.array(q16, float)
        for a, arm in enumerate(ARMS):
            p[a * 8 + 1] += self.delta2(p[a * 8 : a * 8 + 7], p[a * 8 + 7], arm)
        return p

    def command_targets(self, source_times, commands, delays, end_tick, enabled=True):
        """200 Hz interpolation with fitted delays, sampled at 400 Hz physics."""
        times = np.arange(end_tick + 1) // 2 / self.CONFIG["command_hz"]
        enc = np.empty((len(times), 16))
        for col in range(16):
            delay = delays[col // 8, col % 8] if col % 8 < 7 else 0.0
            enc[:, col] = np.interp(times - delay, source_times, commands[:, col])
        phys = enc.copy()
        if enabled:
            for a, arm in enumerate(ARMS):
                s = slice(a * 8, a * 8 + 8)
                j = a * 8 + 1
                for i in range(len(enc)):
                    phys[i, j] = (
                        phys[i - 1, j]
                        if i and np.array_equal(enc[i, s], enc[i - 1, s])
                        else enc[i, j]
                        + self.delta2(enc[i, a * 8 : a * 8 + 7], enc[i, a * 8 + 7], arm)
                    )
        return (enc, phys)
