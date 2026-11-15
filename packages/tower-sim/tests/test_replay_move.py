"""Every relocated symbol and both legacy import orders stay equivalent."""

import ast
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "packages/tower-sim/src/tower_sim"
MANIFEST = json.loads(Path(__file__).with_name("replay_move_ast.json").read_text())


class RelocatedImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "tower_sim.config" and node.names[0].name == "SimConfig":
            node.module = "tower_sim.backend.model"
            node.names = [ast.alias(name="Config")]
        elif node.module == "tower_eval.simulator" and node.names[0].name == "spec":
            node.module = "tower_sim.backend"
            node.names = [ast.alias(name="model")]
        elif node.module == "tower_eval.simulator.spec":
            node.module = "tower_sim.backend.model"
        if node.module == "tower_eval.simulator":
            node.module = "tower_eval.simulator.isaac"
        if node.module and node.module.startswith("tower_sim.replay."):
            node.module = node.module.replace("tower_sim.replay.", "replay_runner_" + "isaac_sim.")
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "load_runtime_profiles":
            for keyword in list(node.keywords):
                if keyword.arg == "calibration_report":
                    assert ast.unparse(keyword.value) == "resources.path('calibration_report')"
                    node.keywords.remove(keyword)
        return node


def symbol_hashes(tree, *, module=False):
    result = {}

    def collect(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                key = prefix + child.name
                node = RelocatedImports().visit(copy.deepcopy(child))
                # Keep the Python 3.11 manifest stable on Python 3.12.
                for sub in ast.walk(node):
                    if getattr(sub, "type_params", None) == []:
                        del sub.type_params
                value = ast.dump(node, include_attributes=False)
                result[key] = hashlib.sha256(value.encode()).hexdigest()
                collect(child, key + ".")
            else:
                collect(child, prefix)

    collect(tree)
    if module:

        class StripDefinitions(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                return None

            visit_AsyncFunctionDef = visit_FunctionDef
            visit_ClassDef = visit_FunctionDef

        body = StripDefinitions().visit(copy.deepcopy(tree))
        result["__module__"] = hashlib.sha256(
            ast.dump(body, include_attributes=False).encode()
        ).hexdigest()
    return result


def before_public_station_identity(source):
    return (
        source.replace("\nfrom tower_sim.bench.asset_identity import station_identity\n", "")
        .replace("    stage_identity_before = station_identity(stage_hash_before)\n", "")
        .replace(
            "runtime_stage_sha256=stage_identity_before if reference_report",
            "runtime_stage_sha256=stage_hash_before if reference_report",
        )
        .replace(
            "profile_source_stage_verified = provenance_hash == stage_identity_before",
            "profile_source_stage_verified = provenance_hash == stage_hash_before",
        )
    )


@pytest.mark.parametrize("name", MANIFEST["modules"])
def test_every_replay_symbol_matches_original(name):
    source = (PACKAGE / "replay" / f"{name}.py").read_text()
    if name == "dual_replay_worker":
        source = before_public_station_identity(source)
    if name == "replay_actions":
        neutral = "Single-arm replay is not part of the public benchmark."
        assert source.count(neutral) == 2
        source = source.replace(
            neutral, "Single-arm replay is retired; use tag " + "benchmark-2026.11.0" + "."
        )
        source = source.replace(
            'from tower_sim.backend.resources import default_path\n\nRUNNER_ROOT = default_path("replay_runner")',
            "RUNNER_ROOT = Path(__file__).resolve().parent",
        ).replace("import math\n", "import math\nimport re\n")
    actual = symbol_hashes(ast.parse(source), module=True)
    expected = MANIFEST["modules"][name]["symbols"].copy()
    if name == "replay_actions":
        actual.pop("default_output_dir")
        expected.pop("default_output_dir")
    assert actual == expected


@pytest.mark.archive
@pytest.mark.parametrize("name", ["dual_core", "dual_replay_worker", "replay_actions"])
def test_replay_bytes_match_git_original(name):
    from scripts.maintainer.archive_root import archived_file

    row = MANIFEST["modules"][name]
    path = archived_file(name + ".py", row["original_sha256"])
    original = path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == row["original_sha256"]
    if name != "replay_actions":
        current = (PACKAGE / "replay" / f"{name}.py").read_bytes()
        if name == "dual_replay_worker":
            current = before_public_station_identity(current.decode()).encode()
        assert current == original


def test_environment_relocation_preserves_every_symbol():
    actual = symbol_hashes(ast.parse((PACKAGE / "bench/environment.py").read_text()))
    assert actual == MANIFEST["environment"]["symbols"]
    assert not (PACKAGE / "bench/bridge.py").exists()


def test_native_wrapper_captures_renderer_then_model_after_app(monkeypatch, tmp_path):
    from tower_sim.bench import app

    events = []

    class NativeApp:
        def close(self):
            events.append("close")

    def factory(config):
        events.append("SimulationApp")
        return NativeApp()

    def renderer():
        events.append("renderer")
        return object()

    monkeypatch.setattr(app, "require_python312", lambda: None)
    with (
        app.native_render_initialization(factory, renderer) as native,
        app.grouped_runtime(
            native,
            (0,),
            tmp_path,
            sdk=tmp_path,
            startup="stock",
            model_setup=lambda session: events.append("model"),
        ) as launch,
    ):
        session = launch({"headless": True})
        events.append("environment")
        session.close()
    assert events == ["SimulationApp", "renderer", "model", "environment", "close"]


def test_pure_command_symbols_match_original():
    path = ROOT / "src/tower-eval/src/tower_eval/simulator/commands.py"
    assert symbol_hashes(ast.parse(path.read_text())) == MANIFEST["pure_commands"]


def test_eula_symbol_matches_original():
    assert (
        symbol_hashes(ast.parse((PACKAGE / "replay/eula.py").read_text()))
        == MANIFEST["eula"]["symbols"]
    )


def test_complete_run_closure_rejects_legacy_alias_even_with_canonical_identity():
    from tower_sim.bench.source_receipt import validate_imports

    row = {"path": str(PACKAGE / "backend/assets.py")}
    with pytest.raises(ValueError, match="Forbidden candidate import"):
        validate_imports(
            {
                "modules": {
                    "tower_sim.backend.assets": row,
                    "jenga_generator.assets": row,
                }
            }
        )


def test_renderer_factory_observes_patch_after_environment_import(tmp_path):
    code = """
from tower_sim.bench import environment
from tower_sim.replay import dual_replay_worker
marker = object()
dual_replay_worker._configure_calibration_renderer = marker
assert environment.renderer_base() is marker
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_three_way_joint_commands_identity():
    from tower_eval import simulator
    from tower_eval.simulator import commands

    assert simulator.joint_commands is commands.joint_commands
