"""AST parity with exact, reviewable seam deltas (no ignored code regions).

Each allowed old/new hunk is listed in construction_ast_allowlist.json. Imports,
annotations, constructor seams, unsupported camera branches and diagnostics are
explicit there, so a change even inside a seam requires review. Frozen sources
are read only and compared on every run; formatting/comments do not matter.
"""

import ast
import difflib
import json
from pathlib import Path

import pytest

from scripts.maintainer.archive_root import ArchivedToolPath
from tests.characterization.runtime_scenarios import ROOT, RUNTIME

BENCH = ROOT / "packages/tower-sim/src/tower_sim/bench"
CASES = [
    (
        ROOT / "packages/tower-sim/src/tower_sim/backend/backend.py",
        "IsaacEnvironment.__init__",
        "environment.py",
        "environment_class.MappedEnvironment.__init__",
    ),
    (
        RUNTIME / "support/task2_fixed/native61.py",
        "native_render_initialization.launch.configure",
        "app.py",
        "native_renderer.configure",
    ),
    (
        ArchivedToolPath("batch_reference_worker.py"),
        "bootstrap.prepare",
        "scene.py",
        "prepare_physics_scene",
    ),
    (ArchivedToolPath("task7_runtime.py"), "author_bonds", "task7.py", "author_bonds"),
    *[
        (ArchivedToolPath("batch_reference_worker.py"), name, "bootstrap.py", name)
        for name in ("initialize_homes", "verify_initial_readback")
    ],
    (RUNTIME / "support/task2_fixed/runtime.py", "gpu_readback", "app.py", "gpu_readback"),
    (
        ArchivedToolPath("batch_reference_worker.py"),
        "adapt_task12_config",
        "bootstrap.py",
        "adapt_config",
    ),
    (
        ArchivedToolPath("batch_reference_worker.py"),
        "bootstrap.MappedEnvironment._commands",
        "environment.py",
        "environment_class.MappedEnvironment._commands",
    ),
    *[
        (ArchivedToolPath("batch_backend.py"), name, "scene.py", name)
        for name in (
            "_digest",
            "_file_hash",
            "plans_for_candidates",
            "_author_block",
            "author_heterogeneous_blocks",
            "verify_usd_readback",
            "verify_physics_readback",
        )
    ],
]


def scopes(node):
    """Find named scopes through with/try blocks, without crossing another scope."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child
        else:
            yield from scopes(child)


def function_node(path, qualified_name):
    source = path.read_text().replace("from tower_sim.backend.", "from jenga_generator.")
    source = source.replace("replay_runner_" + "isaac_sim.", "tower_sim.replay.")
    node = ast.parse(source)
    for name in qualified_name.split("."):
        matches = [n for n in scopes(node) if n.name == name]
        assert len(matches) == 1, (path, qualified_name, name)
        node = matches[0]
    return node


def line_differences(old, new):
    return [
        {"old": old[i:j], "new": new[k:end]}
        for tag, i, j, k, end in difflib.SequenceMatcher(
            None, old, new, autojunk=False
        ).get_opcodes()
        if tag != "equal"
    ]


def differences(old_path, old_name, new_file, new_name):
    return line_differences(
        ast.unparse(function_node(old_path, old_name)).splitlines(),
        ast.unparse(function_node(BENCH / new_file, new_name)).splitlines(),
    )


def post_construction(path):
    """Keep every statement from the dt change through the initial-state check."""
    bootstrap = function_node(path, "bootstrap")
    matches = []
    for node in ast.walk(bootstrap):
        for _field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            starts = [
                i
                for i, stmt in enumerate(value)
                if isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and ast.unparse(stmt.value.func) == "env.context.set_simulation_dt"
            ]
            for start in starts:
                ends = [
                    i
                    for i, stmt in enumerate(value)
                    if isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and ast.unparse(stmt.value.func) == "verify_initial_readback"
                ]
                assert len(ends) == 1 and ends[0] > start
                matches.append(value[start : ends[0] + 1])
    assert len(matches) == 1
    return ast.unparse(ast.Module(body=matches[0], type_ignores=[])).splitlines()


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, marks=() if isinstance(case[0], Path) else pytest.mark.archive)
        for case in CASES
    ],
    ids=[f"{c[2]}:{c[3]}" for c in CASES],
)
def test_frozen_ast_parity(case):
    allowed = json.loads(Path(__file__).with_name("construction_ast_allowlist.json").read_text())
    assert differences(*case) == allowed[f"{case[2]}:{case[3]}"]


@pytest.mark.archive
def test_frozen_post_construction_ast_parity():
    allowed = json.loads(Path(__file__).with_name("construction_ast_allowlist.json").read_text())
    assert (
        line_differences(
            post_construction(ArchivedToolPath("batch_reference_worker.py")),
            post_construction(BENCH / "bootstrap.py"),
        )
        == allowed["bootstrap.py:post_construction"]
    )
