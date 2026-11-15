"""Regression coverage for public importer boundaries."""

import ast
import importlib
import importlib.util
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from tower_sim.backend import resources
from tower_sim.bench import source_receipt, verification

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINTS = [
    "tools/simulator/benchmark_environment.py",
    "tools/simulator/benchmark_diagnostic.py",
    "tools/simulator/benchmark.py",
    "tools/simulator/benchmark_candidate.py",
    "tools/simulator/benchmark_portable.py",
    "tools/simulator/benchmark_assets.py",
    "tools/simulator/benchmark_diagnostic_score.py",
    "scripts/maintainer/check_golden_exact.py",
    "scripts/maintainer/candidate_replay.py",
    "scripts/maintainer/gpu_baseline.py",
    "scripts/maintainer/candidate_release.py",
    "scripts/maintainer/candidate_policy.py",
    "scripts/maintainer/bench_scoring_replay.py",
]


@pytest.mark.parametrize("collector", ["startup", "complete"])
@pytest.mark.parametrize("kind", ["zip", "loader", "missing", "no-loader"])
def test_unbound_nonfile_imports_fail_closed(tmp_path, monkeypatch, collector, kind):
    name = "p712_nonfile_import"
    if kind == "zip":
        archive = tmp_path / "foreign.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr(name + ".py", "VALUE = 12\n")
        monkeypatch.syspath_prepend(str(archive))
        module = importlib.import_module(name)
        sys.modules.pop(name)
    else:
        module = NS(__spec__=NS(loader=None)) if kind == "no-loader" else NS(__loader__=NS())
        if kind == "missing":
            module.__file__ = str(tmp_path / "does-not-exist.py")
    monkeypatch.setitem(sys.modules, name, module)
    if collector == "startup":
        with pytest.raises(ValueError, match="Unbound non-file import"):
            verification.imported_sources()
    else:
        receipt = source_receipt.ImportReceipt()
        try:
            with pytest.raises(ValueError, match="Unbound non-file import"):
                receipt.collect()
        finally:
            receipt.close()


def test_only_explicit_sdk_archives_are_admitted(tmp_path, monkeypatch):
    import zipimport

    archive = tmp_path / "sdk.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("sdk_member.py", "VALUE = 12\n")
    module = NS(
        __file__=str(archive / "sdk_member.py"), __loader__=zipimport.zipimporter(str(archive))
    )
    monkeypatch.setattr(verification, "sdk_roots", lambda: [str(tmp_path)])
    assert source_receipt.module_source("sdk_member", module) == archive
    monkeypatch.setattr(verification, "sdk_roots", lambda: [])
    with pytest.raises(ValueError, match="Unbound non-file import"):
        source_receipt.module_source("sdk_member", module)


@pytest.mark.parametrize("entry", ENTRYPOINTS)
def test_entry_disables_bytecode_before_checkout_import(entry, tmp_path):
    # Execute the real entry preamble in a fresh interpreter, with bytecode
    # enabled initially, then import a real file and check no bytecode appears.
    source = (ROOT / entry).read_text()
    tree = ast.parse(source)
    stdlib = sys.stdlib_module_names
    stop = next(
        (
            i
            for i, node in enumerate(tree.body)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            or isinstance(node, (ast.Import, ast.ImportFrom))
            and (
                (node.module or "").split(".")[0]
                if isinstance(node, ast.ImportFrom)
                else node.names[0].name.split(".")[0]
            )
            not in stdlib | {"__future__"}
        ),
        len(tree.body),
    )
    prefix = ast.unparse(ast.Module(body=tree.body[:stop], type_ignores=[]))
    code = (
        "import sys; sys.dont_write_bytecode = False\n"
        + f"exec({prefix!r})\n"
        + "\nassert sys.dont_write_bytecode\n"
    )
    code += f"sys.path.insert(0, {str(tmp_path)!r})\nimport p712_bytecode_probe\n"
    (tmp_path / "p712_bytecode_probe.py").write_text("VALUE = 12\n")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX"}
    }
    result = subprocess.run(
        [sys.executable, "-c", f"__file__ = {str(ROOT / entry)!r}\n" + code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "__pycache__").exists()


def test_wheel_declares_runtime_dependencies():
    metadata = tomllib.loads((ROOT / "packages/tower-sim/pyproject.toml").read_text())
    assert "pyyaml>=6,<7" in metadata["project"]["dependencies"]
    assert "pillow>=10,<13" in metadata["project"]["dependencies"]
    assert "h5py>=3.11,<4" in metadata["project"]["dependencies"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    package = next(p for p in lock["package"] if p["name"] == "tower-sim")
    assert {"pyyaml", "h5py", "pillow"} <= {p["name"] for p in package["dependencies"]}


def test_runtime_profiles_uses_only_canonical_replay(monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith("replay_runner_" + "isaac_sim"), name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    path = ROOT / "packages/tower-sim/src/tower_sim/backend/runtime_profiles.py"
    spec = importlib.util.spec_from_file_location("tower_sim.backend._profile_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.load_profile.__module__ == "tower_sim.replay.dual_core"


def test_wheel_resource_defaults_never_follow_cwd(tmp_path, monkeypatch):
    package = tmp_path / "installed/tower_sim/backend/resources.py"
    package.parent.mkdir(parents=True)
    package.write_bytes(Path(resources.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location("p712_installed_resources", package)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("TOWER_SOURCE_ROOT", raising=False)
    evil = tmp_path / "foreign"
    evil.mkdir()
    monkeypatch.chdir(evil)
    for name, relative in resources.PATHS.items():
        planted = evil / relative
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted.write_text("foreign resource")
        result = module.default_path(name)
        assert result.is_absolute()
        assert result.is_relative_to(package.parents[1])
        assert not result.exists()
    monkeypatch.setenv("TOWER_SOURCE_ROOT", str(tmp_path / "bound"))
    assert module.default_path("stage") == module.DATA / resources.PATHS["stage"]


@pytest.mark.parametrize(
    "entry",
    [
        "benchmark_portable",
        "benchmark_diagnostic",
        "benchmark_diagnostic_score",
    ],
)
def test_reference_help_names_private_archive(entry):
    tree = ast.parse((ROOT / "tools/simulator" / (entry + ".py")).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and any(isinstance(a, ast.Constant) and a.value == "--reference" for a in n.args)
    ]
    assert calls
    for call in calls:
        help_value = next(k.value for k in call.keywords if k.arg == "help")
        assert isinstance(help_value, ast.Constant)
        assert (
            help_value.value
            == "Unavailable: the frozen reference runtime is maintained in the private archive"
        )


def test_source_free_native_submodule_requires_admitted_parent(tmp_path, monkeypatch):
    native = tmp_path / "sdk_native.so"
    native.write_bytes(b"native extension fixture")
    monkeypatch.setitem(sys.modules, "sdk_native", NS(__file__=str(native)))
    monkeypatch.setattr(verification, "sdk_roots", lambda: [str(tmp_path)])
    assert source_receipt.module_source("sdk_native.child", NS()) == native
    assert source_receipt.module_source("unbound", NS(__name__="sdk_native.child")) is None
    monkeypatch.setattr(verification, "sdk_roots", lambda: [])
    assert source_receipt.module_source("sdk_native.child", NS()) is None
