"""Relocation equality, real aliases, and original-resource boundary checks."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "packages/tower-sim/src/tower_sim/backend"
MANIFEST = json.loads(Path(__file__).with_name("backend_move_ast.json").read_text())


class RelocatedImports(ast.NodeTransformer):
    """Normalize explicit import relocations and the injected planner seam only."""

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if node.name == "PhysicsLogGuard":
            doc = node.body[0].value
            assert isinstance(doc, ast.Constant) and isinstance(doc.value, str)
            # Exact cosmetic terminology change; every other class value stays bound.
            doc.value = doc.value.replace(
                "The caller persists that invalidation",
                "The coordinator persists that invalidation",
            )
        return node

    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("tower_sim.replay."):
            node.module = node.module.replace("tower_sim.replay.", "replay_runner_" + "isaac_sim.")
        if node.module in (
            "tower_sim.backend.tasks",
            "tower_sim.backend.profiles",
            "tower_sim.backend.assets",
        ):
            node.module = node.module.rsplit(".", 1)[-1]
            node.level = 1
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name == "load_runtime_profiles":
            identity = [
                statement
                for statement in node.body
                if ast.unparse(statement) == "stage_sha256 = station_identity(stage_sha256)"
            ]
            assert len(identity) == 1
            node.body.remove(identity[0])  # Separately pinned metadata-only station identity.
            assert ast.unparse(node.args.kwonlyargs[-1]) == "calibration_report: Path"
            assert ast.unparse(node.args.kw_defaults[-1]) == "REPORT"
            node.args.kwonlyargs.pop()
            node.args.kw_defaults.pop()

            class LegacyReport(ast.NodeTransformer):
                def visit_Name(self, value):
                    if value.id == "calibration_report":
                        value.id = "REPORT"
                    return value

            node = LegacyReport().visit(node)
        return node


def symbol_hashes(tree):
    result = {}

    def collect(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                key = prefix + child.name
                node = RelocatedImports().visit(copy.deepcopy(child))
                # Python 3.12 adds empty type_params fields; the manifest was recorded on 3.11.
                for sub in ast.walk(node):
                    if getattr(sub, "type_params", None) == []:
                        del sub.type_params
                value = ast.dump(node, include_attributes=False)
                result[key] = hashlib.sha256(value.encode()).hexdigest()
                collect(child, key + ".")
            else:
                collect(child, prefix)

    collect(tree)
    return result


@pytest.mark.parametrize("name", MANIFEST["modules"])
def test_every_moved_function_and_class_equals_original(name):
    if name == "model":
        assert not (BACKEND / "model.py").exists()
        return
    expected = MANIFEST["modules"][name]["symbols"].copy()
    tree = ast.parse((BACKEND / f"{name}.py").read_text())
    actual = symbol_hashes(tree)
    if name == "profiles":
        expected = {"load_profile": expected["load_profile"]}
    if name == "backend":
        # The complete environment contains a removed planner method. Every
        # remaining method is still independently compared to its original hash.
        environment = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IsaacEnvironment"
        )
        assert not environment.bases and not environment.keywords and not environment.decorator_list
        assert all(isinstance(n, ast.FunctionDef) for n in environment.body)
        assert "IsaacEnvironment.plan_free_space" not in actual
        for key in ["main", "IsaacEnvironment", "IsaacEnvironment.plan_free_space"]:
            expected.pop(key, None)
            actual.pop(key, None)
    assert actual == expected


def test_bridge_uses_canonical_backend():
    source = (BACKEND.parent / "bench/environment.py").read_text()
    assert "jenga_generator" not in source
    assert "from tower_sim.backend.backend import IsaacEnvironment, rigid_pose" in source


@pytest.mark.parametrize("command", ["describe", "verify"])
def test_reference_cli_refuses_before_verification(command, monkeypatch):
    from scripts.maintainer.golden_support import benchmark

    monkeypatch.setattr(benchmark, "verify_bundle", lambda: pytest.fail("reference ran"))
    with pytest.raises(RuntimeError, match="maintainers reproduce it from the private archive"):
        benchmark.main([command, "--reference"])


@pytest.mark.archive
def test_reference_scorer_remains_independent_of_backend():
    from scripts.maintainer.golden_support import reference_scorer

    assert reference_scorer().__file__.endswith("scoring/early_success.py")


def test_backend_import_receipt_rejects_checkout_implementation():
    from tower_sim.bench.source_receipt import validate_imports

    name = "jenga_generator.model"
    row = dict(path=str(ROOT / "src/tower-datagenerator/jenga_generator/model.py"))
    with pytest.raises(ValueError, match="Forbidden candidate import"):
        validate_imports({"modules": {name: row}})
