"""Probe installed eval/bench imports outside a checkout, without Isaac or generation.

Run a copied script with a private wheel-only venv's Python -I. This CPU probe
exercises public construction dependencies without starting a native app. The
complete-run imported-sources receipt remains a separate required GPU gate.
"""

import argparse
import base64
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import json
import pkgutil
import sys
import tempfile
from pathlib import Path

# Retired generation-only modules must remain outside the evaluator import closure.
GENERATION_MODULES = (
    "analysis_report",
    "batch_context",
    "benchmark",
    "block_randomization",
    "candidates",
    "cli",
    "contact_planning",
    "curobo_planner",
    "diagnose",
    "grasp_control",
    "inspect_gripper",
    "motion",
    "physics_gate",
    "pipeline",
    "pipeline_worker",
    "publication",
    "quality_gates",
    "quality_review",
    "randomize_cli",
    "randomized_replay",
    "randomized_replay_cli",
    "recording",
    "render_reference",
    "replay_validation",
    "smooth",
    "stack_preview",
    "worker",
)
FORBIDDEN = (
    "jenga_generator",
    "replay_runner_" + "isaac_sim",
    "task2_fixed",
    "tower_eval.simulator.isaac",
    "tower_sim.generation",
    "tower_sim.backend.model",
    "isaacsim",
    "omni",
    "pxr",
    "carb",
    "lula",
    "pinocchio",
    "casadi",
)


def forbidden(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN)


class ClosureGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise RuntimeError(f"Forbidden eval import: {fullname}")


def probe(checkout):
    checkout = Path(checkout).resolve()
    if not sys.flags.isolated:
        raise RuntimeError("The closure check requires Python -I isolation")
    if Path(__file__).resolve().is_relative_to(checkout):
        raise RuntimeError("Copy the closure script outside the checkout")
    if Path.cwd().is_relative_to(checkout):
        raise RuntimeError("Run the closure check outside the checkout")
    guard = ClosureGuard()
    sys.meta_path.insert(0, guard)
    try:
        import tower_sim.bench
        from tower_sim.bench import bootstrap, environment
        from tower_sim.replay.eula import require_eula

        for module in pkgutil.walk_packages(tower_sim.bench.__path__, "tower_sim.bench."):
            importlib.import_module(module.name)
        # Resolve the same lazy dependencies as the public construction factories.
        factories = bootstrap.ConstructionFactories()
        config_class = factories.config_class()
        with tempfile.TemporaryDirectory(prefix="tower-closure-stage-") as scratch:
            stage = Path(scratch) / "stage.usd"
            stage.write_text("CPU import probe; no native scene is constructed")
            values = {"stage": str(stage), "max_recoveries": 0}
            config = (
                config_class.from_v1(values)
                if hasattr(config_class, "from_v1")
                else config_class(**values)
            )
            assert config.gripper_finger_effort_limit_n == 22.0
            config.validate()
        factories.native_spec(
            {
                "episode_spec": {
                    "logical_id": 0,
                    "seed": 0,
                    "blocks": [],
                    "transfers": [],
                }
            }
        )
        factories.environment_class()
        environment.renderer_base()
        require_eula(environment={"OMNI_KIT_ACCEPT_EULA": "YES"})
        modules = {}
        records = {}
        for distribution in importlib.metadata.distributions():
            for member in distribution.files or ():
                root = str(member).split("/")[0]
                if root.startswith(("tower_", "nero_")) and member.hash is not None:
                    records.setdefault(
                        Path(str(distribution.locate_file(member))).resolve(), []
                    ).append((distribution.metadata["Name"], member))
        for name, module in list(sys.modules.items()):
            if forbidden(name):
                raise RuntimeError(f"Forbidden loaded module: {name}")
            source = getattr(module, "__file__", None)
            owned = name.split(".")[0].startswith(("tower_", "nero_"))
            if not source:
                if owned:
                    raise RuntimeError(f"Unrecorded installed module: {name}")
                continue
            path = Path(source).resolve()
            if path.is_relative_to(checkout):
                raise RuntimeError(f"Checkout import in wheel-only probe: {name}: {path}")
            if "/simulator/benchmark-202611/" in path.as_posix():
                raise RuntimeError(f"Frozen import: {name}: {path}")
            if owned:
                matches = records.get(path, [])
                if not path.is_file() or len(matches) != 1:
                    raise RuntimeError(f"Unrecorded installed module: {name}: {path}")
                distribution, member = matches[0]
                data = path.read_bytes()
                digest = (
                    base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                )
                if (
                    member.hash.mode != "sha256"
                    or member.hash.value != digest
                    or member.size != len(data)
                ):
                    raise RuntimeError(f"Installed RECORD mismatch: {name}: {path}")
                modules[name] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "distribution": distribution,
                    "record": str(member),
                }
        return {
            "schema": "tower-eval-import-closure-v1",
            "passed": True,
            "scope": "CPU wheel-only construction imports; native execution is a separate gate",
            "python": sys.executable,
            "cwd": str(Path.cwd()),
            "checkout_excluded": str(checkout),
            "distributions": {
                name: importlib.metadata.version(name)
                for name in (
                    "tower-eval",
                    "tower-sim",
                    "nero-kinematics",
                )
            },
            "modules": modules,
            "generation_only_removable": list(GENERATION_MODULES),
        }
    finally:
        sys.meta_path.remove(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(probe(args.checkout), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
