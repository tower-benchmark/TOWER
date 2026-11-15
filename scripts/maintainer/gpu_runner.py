"""CPU-importable replay interface; construction is supplied by the runtime.

Entry point: tower_sim.bench.gpu_baseline.create_runner() -> CandidateRunner.
See comparison.md for lifecycle, evidence, identity and substitution semantics.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast


@dataclass(frozen=True)
class ReplayRequest:
    task: str
    seed: int
    expert_sha256: str
    assets: Path
    output: Path
    gpu: int
    contract: dict[str, Any]
    frozen_components: tuple[str, ...] = ()
    reference_root: Path | None = None


@dataclass(frozen=True)
class CommandTrajectory:
    """Verified NPZ time_s (N,) and action_qpos (N,16), in encoder units."""

    time_s: Any
    action_qpos: Any


@dataclass(frozen=True)
class ReplayEvidence:
    """Finalized artifacts, persisted before closing Kit can exit the worker.

    trace.npz contains time (N,), qpos (N,16), blocks (N,B,7); cameras maps
    four names to first-frame RGB PNG paths relative to ReplayRequest.output.
    The terminal fields are the frozen replay's latch/status identities.
    """

    replay_completed: bool
    task_success: bool
    policy_action_failure: bool
    cameras: dict[str, str]


class PreparedReplay(Protocol):
    def replay(self, commands: CommandTrajectory) -> ReplayEvidence:
        """Validate the initial command against the scene, execute, finalize evidence."""
        ...

    def close(self) -> None:
        """Release Kit/resources after the harness has collected the summary."""
        ...


class CandidateRunner(Protocol):
    def identity(
        self,
        assets: Path,
        gpu: int,
        frozen_components: tuple[str, ...],
        reference_root: Path | None,
    ) -> dict[str, Any]:
        """Return candidate_identity and contract from verified inputs/environment.

        CPU-only preflight: hash actual wheels/dependencies; verify data contents,
        physics settings and hardware/runtime profile. Do not copy an expected
        contract without validation. No SimulationApp or GPU context here.
        Substitution must be explicit, supported and recorded, never a fallback.
        """
        ...

    def setup(self, request: ReplayRequest) -> PreparedReplay:
        """Create the task/seed scene and runtime; clean up if setup fails."""
        ...


def resolve_candidate() -> CandidateRunner:
    name = "tower_sim.bench.gpu_baseline"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name not in ("tower_sim", "tower_sim.bench", name):
            raise
        raise RuntimeError(
            "--implementation tower-sim is unavailable: install a runtime providing "
            "tower_sim.bench.gpu_baseline.create_runner(); no frozen fallback is allowed"
        ) from error
    factory = getattr(module, "create_runner", None)
    if not callable(factory):
        raise RuntimeError(f"Candidate runner must provide {name}.create_runner()")
    runner = factory()
    if not all(callable(getattr(runner, method, None)) for method in ("identity", "setup")):
        raise RuntimeError("Candidate runner does not implement identity() and setup()")
    return cast(CandidateRunner, runner)
