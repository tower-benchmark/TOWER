"""Independent pre-split defaults/types, field coverage and generation boundaries."""

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, field, fields, make_dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "packages/tower-sim/src/tower_sim"
MANIFEST = json.loads(Path(__file__).with_name("config_split_ast.json").read_text())


def baseline_config():
    from tower_sim.config import DEFAULT_PROFILE, DEFAULT_RUNTIME_PROFILES, DEFAULT_STAGE

    namespace = dict(
        field=field,
        DEFAULT_STAGE=DEFAULT_STAGE,
        DEFAULT_PROFILE=DEFAULT_PROFILE,
        DEFAULT_RUNTIME_PROFILES=DEFAULT_RUNTIME_PROFILES,
    )
    return make_dataclass(
        "OriginalConfig",
        [
            (
                name,
                eval(MANIFEST["fields"][name]["annotation"], namespace),
                eval(MANIFEST["fields"][name]["default"], namespace),
            )
            for name in MANIFEST["field_order"]
        ],
    )


def test_all_defaults_and_types_equal_pre_split_config():
    from tower_sim.config import SimConfig

    baseline = baseline_config()
    expected = baseline()
    original_fields = {item.name: item for item in fields(baseline)}
    assert {item.name for item in fields(SimConfig)} == set(MANIFEST["sim_fields"])
    for item in fields(SimConfig):
        assert item.type == original_fields[item.name].type
        assert type(getattr(SimConfig(), item.name)) is type(getattr(expected, item.name))
        assert getattr(SimConfig(), item.name) == getattr(expected, item.name)
    a, b = SimConfig(), SimConfig()
    a.gpus.append(99)
    assert b.gpus == list(range(8))


def test_v1_validation_body_is_unchanged():
    node = next(
        n
        for n in ast.parse((PACKAGE / "_config_v1.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "validate"
    )
    for sub in ast.walk(node):
        if getattr(sub, "type_params", None) == []:
            del sub.type_params
    assert (
        hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        == MANIFEST["validation_ast_sha256"]
    )


def test_eval_read_field_inventory_is_complete():
    from tower_sim.config import SimConfig

    supported = {item.name for item in fields(SimConfig)}
    seen = set()
    sources = [
        *(
            p
            for p in sorted((PACKAGE / "backend").glob("*.py"))
            if p.name not in {"tasks.py", "profiles.py"}
        ),
        *sorted((PACKAGE / "replay").glob("*.py")),
        *sorted((PACKAGE / "bench").glob("*.py")),
    ]
    for path in sources:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "plan_free_space":
                # This injected generation API is not part of eval; verify it separately.
                node.body = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in MANIFEST["fields"]
                and ast.unparse(node.value)
                in {
                    "config",
                    "cfg",
                    "self.config",
                    "env.config",
                    "context.env.config",
                }
            ):
                seen.add(node.attr)
    assert seen <= supported, seen - supported
    assert {"gripper_finger_effort_limit_n", "reset_settle_seconds", "wrist_near_clip_mm"} <= seen


@pytest.mark.parametrize(
    "task,layers",
    [(f"task{task}-{layers}", layers) for task in (1, 2, 4, 5, 6, 7) for layers in (2, 3)],
)
def test_bench_v1_defaults_equal_original(task, layers, tmp_path):
    from types import SimpleNamespace

    from tower_sim.bench.bootstrap import adapt_config
    from tower_sim.config import SimConfig

    stage = tmp_path / "stage.usd"
    stage.write_text("synthetic CPU validation fixture")
    values = asdict(baseline_config()(task=task, layers_above_base=layers, stage=str(stage)))
    cfg, receipt = adapt_config(values, SimConfig, SimpleNamespace(name=task))
    cfg.validate()
    assert cfg.to_dict() == {name: values[name] for name in MANIFEST["sim_fields"]}
    assert receipt["relevant_settings_preserved"] is True
    assert not hasattr(cfg, "curobo_config") and not hasattr(cfg, "rollouts")
    # Retain the old refusal even for a generation key irrelevant to eval.
    values["max_recoveries"] = 1
    invalid, _ = adapt_config(values, SimConfig, SimpleNamespace(name=task))
    with pytest.raises(ValueError, match="retries are disabled"):
        invalid.validate()


def test_eval_and_v1_readers_never_import_generation(tmp_path):
    code = f"""
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("jenga_generator", "tower_sim.generation", "tower_sim.backend.model")):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from tower_eval.scenes import frozen_layout
from tower_sim.config import SimConfig
from tower_sim.bench import environment
from tower_sim.backend import tasks
cfg = SimConfig.from_v1({{"stage": {str(tmp_path / "stage.usd")!r}, "curobo_config": "legacy-unused"}})
cfg.validate()
environment.config_class()
environment.native_spec({{"episode_spec": {{"logical_id": 1, "seed": 2, "blocks": [], "transfers": []}}}})
environment.environment_class()
from tower_sim.replay.dual_core import DUAL_SCHEMAS, camera_datasets_for_schema
for schema in DUAL_SCHEMAS:
    assert len(camera_datasets_for_schema(schema)) == 4
"""
    (tmp_path / "stage.usd").write_text("CPU-only fixture")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_v1_data_defaults_and_relocated_constants_match_original():
    from tower_sim._config_v1 import defaults

    expected = asdict(baseline_config()())
    assert defaults() == expected
    for name, value in defaults().items():
        assert type(value) is type(expected[name]), name
    found = {}
    for path in (PACKAGE / "config.py", ROOT / "src/tower-eval/src/tower_eval/simulator/spec.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                found[node.targets[0].id] = hashlib.sha256(
                    ast.dump(node, include_attributes=False).encode()
                ).hexdigest()
    assert found == MANIFEST["constants"]


def test_v1_reader_rejects_unknown_fields():
    from tower_sim.config import SimConfig

    with pytest.raises(ValueError, match="Unsupported v1 config fields.*bogus"):
        SimConfig.from_v1({"bogus": 1})


@pytest.mark.parametrize(
    "name",
    [
        "tower_sim.backend.model",
        "tower_sim.generation",
        "tower_sim.generation.config",
        "tower_sim.generation.planted",
    ],
)
def test_complete_run_receipt_rejects_generation(name):
    from tower_sim.bench.source_receipt import validate_imports

    with pytest.raises(ValueError, match="Forbidden candidate import"):
        validate_imports({"modules": {name: {"path": str(PACKAGE / "backend/model.py")}}})


def test_retained_task_symbols_match_original_and_removed_symbols_are_absent():
    manifest = json.loads(Path(__file__).with_name("tasks_move_ast.json").read_text())
    tree = ast.parse((PACKAGE / "backend/tasks.py").read_text())
    actual = {}

    def collect(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                key = prefix + child.name
                normalized = copy.deepcopy(child)
                for sub in ast.walk(normalized):
                    if getattr(sub, "type_params", None) == []:
                        del sub.type_params
                actual[key] = hashlib.sha256(
                    ast.dump(normalized, include_attributes=False).encode()
                ).hexdigest()
                collect(child, key + ".")
            else:
                collect(child, prefix)

    collect(tree)
    retained = {
        "orientation_error",
        "TransferEvaluator",
        "TransferEvaluator.__init__",
        "TransferEvaluator.update",
    }
    assert set(actual) == retained
    assert actual == {key: value for key, value in manifest["symbols"].items() if key in retained}
    assert not (
        {"TowerTransfer", "build_layout", "create_task", "TaskLayoutConfig"}
        & {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    )
