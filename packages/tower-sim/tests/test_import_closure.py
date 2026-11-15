"""Exercise probe() in isolated processes with installed RECORD test fixtures.

These fixtures test refusal paths; the release proof separately uses built wheels
in a private venv, with the copied checker outside the checkout.
"""

import base64
import csv
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def installed(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    for name, source in {
        "tower_sim": "packages/tower-sim/src/tower_sim",
        "tower_eval": "src/tower-eval/src/tower_eval",
        "nero_kinematics": "packages/nero-kinematics/src/nero_kinematics",
        "nero_eef_ik": "packages/nero-kinematics/src/nero_eef_ik",
    }.items():
        shutil.copytree(ROOT / source, site / name, ignore=shutil.ignore_patterns("__pycache__"))
    data = site / "tower_eval/_data"
    data.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "benchmarks/2026.11/tasks.json", data / "tasks-2026.11.json")
    # Copy the declared PyYAML dependency into this isolated fixture rather
    # than inheriting PYTHONPATH or modifying the gate venv.
    yaml = importlib.util.find_spec("yaml")
    assert yaml is not None and yaml.origin is not None
    shutil.copytree(Path(yaml.origin).parent, site / "yaml")
    script = tmp_path / "probe.py"
    shutil.copyfile(ROOT / "scripts/maintainer/check_eval_import_closure.py", script)
    return site, script


def write_records(site):
    for name in ("tower_sim", "tower_eval", "nero_kinematics"):
        info = site / f"{name}-1.0.dist-info"
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\nVersion: 1.0\n"
        )
        roots = [name] + (["nero_eef_ik"] if name == "nero_kinematics" else [])
        (info / "top_level.txt").write_text("\n".join(roots) + "\n")
        with (info / "RECORD").open("w", newline="") as stream:
            writer = csv.writer(stream)
            for root in roots:
                for path in sorted((site / root).rglob("*.py")):
                    data = path.read_bytes()
                    digest = (
                        base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                        .rstrip(b"=")
                        .decode()
                    )
                    writer.writerow(
                        [path.relative_to(site).as_posix(), "sha256=" + digest, len(data)]
                    )


def run_probe(installed, tmp_path, setup="", *, isolated=True, checkout=None):
    # This synthetic installation has its own checkout boundary. The gate venv
    # may live inside the real checkout; its third-party dependencies are not
    # fixture source. Explicit planted-path tests exercise checkout rejection.
    if checkout is None:
        checkout = tmp_path / "checkout"
        checkout.mkdir(exist_ok=True)
    site, script = installed
    code = f"""
import importlib.util, sys, types
sys.path.insert(0, {str(site)!r})
spec = importlib.util.spec_from_file_location('closure_probe', {str(script)!r})
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
{setup}
assert probe.probe({str(checkout)!r})['passed']
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, *(["-I"] if isolated else []), "-B", "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )


def refused(result, message):
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stderr, result.stderr


def test_installed_probe_success(installed, tmp_path):
    write_records(installed[0])
    result = run_probe(installed, tmp_path)
    assert result.returncode == 0, result.stderr


def test_probe_requires_isolation(installed, tmp_path):
    write_records(installed[0])
    refused(run_probe(installed, tmp_path, isolated=False), "requires Python -I isolation")


def test_probe_script_must_be_outside_checkout(installed, tmp_path):
    write_records(installed[0])
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    script = checkout / "probe.py"
    shutil.copyfile(installed[1], script)
    refused(
        run_probe((installed[0], script), tmp_path, checkout=checkout),
        "Copy the closure script outside",
    )


@pytest.mark.parametrize(
    "name", ["jenga_generator.motion", "replay_runner_" + "isaac_sim", "tower_eval.simulator.isaac"]
)
def test_probe_rejects_planted_import(installed, tmp_path, name):
    site, _ = installed
    # Make the planted import resolvable independently of the guard.
    if name == "jenga_generator.motion":
        (site / "jenga_generator").mkdir()
        (site / "jenga_generator/__init__.py").write_text("")
        (site / "jenga_generator/motion.py").write_text("PLANTED = True\n")
    elif name == "replay_runner_" + "isaac_sim":
        (site / ("replay_runner_" + "isaac_sim.py")).write_text("PLANTED = True\n")
    (site / "tower_sim/bench/_closure_plant.py").write_text(f"import {name}\n")
    write_records(site)
    refused(run_probe(installed, tmp_path), "Forbidden eval import:")


@pytest.mark.parametrize(
    "name", ["jenga_generator.motion", "replay_runner_" + "isaac_sim", "tower_eval.simulator.isaac"]
)
def test_probe_rejects_already_loaded_forbidden_module(installed, tmp_path, name):
    site, _ = installed
    write_records(site)
    setup = f"module = types.ModuleType({name!r})\nmodule.__file__ = {str(site / 'tower_eval/simulator/isaac.py')!r}\nsys.modules[{name!r}] = module"
    refused(run_probe(installed, tmp_path, setup), "Forbidden loaded module:")


@pytest.mark.parametrize("kind", ["checkout", "frozen"])
def test_probe_rejects_loaded_paths(installed, tmp_path, kind):
    write_records(installed[0])
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    path = (
        checkout / "planted.py"
        if kind == "checkout"
        else tmp_path / "simulator/benchmark-202611/planted.py"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("PLANTED = True\n")
    setup = f"module = types.ModuleType('planted_path')\nmodule.__file__ = {str(path)!r}\nsys.modules['planted_path'] = module"
    refused(
        run_probe(installed, tmp_path, setup, checkout=checkout),
        "Checkout import" if kind == "checkout" else "Frozen import",
    )


@pytest.mark.parametrize("kind", ["tampered", "unrecorded", "missing_file"])
def test_probe_requires_record_hashed_members(installed, tmp_path, kind):
    site, _ = installed
    write_records(site)
    setup = ""
    if kind == "tampered":
        path = site / "tower_sim/bench/environment.py"
        original = path.read_text()
        changed = original.replace("Explicit", "Xxplicit", 1)
        assert changed != original and len(changed.encode()) == len(original.encode())
        path.write_text(changed)
    else:
        path = site / "tower_planted.py"
        path.write_text("PLANTED = True\n")
        setup = "module = types.ModuleType('tower_planted')\nsys.modules['tower_planted'] = module"
        if kind != "missing_file":
            setup += f"\nmodule.__file__ = {str(path)!r}"
    refused(
        run_probe(installed, tmp_path, setup),
        "Installed RECORD mismatch" if kind == "tampered" else "Unrecorded installed module",
    )
