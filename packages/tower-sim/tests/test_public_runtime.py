"""Real CPU inventory and installed-package calibration construction regressions."""

import os
import shutil
import subprocess
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
from tower_sim.bench import source_receipt, verification

ROOT = Path(__file__).resolve().parents[3]
SOURCES = [
    ROOT / "packages/tower-sim/src",
    ROOT / "src/tower-eval/src",
    ROOT / "packages/nero-kinematics/src",
]


def isolated(code, cwd, paths):
    env = dict(os.environ)
    env.pop("TOWER_SOURCE_ROOT", None)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            f"import sys\nsys.path[:0] = {list(map(str, paths))!r}\n" + code,
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_isolated_bench_inventory():
    isolated(
        """
import importlib, pkgutil
import numpy, scipy, h5py, yaml
for optional in ("six", "torch"):
    if importlib.util.find_spec(optional) is not None:
        importlib.import_module("six.moves" if optional == "six" else optional)
import tower_sim.bench
for item in pkgutil.walk_packages(tower_sim.bench.__path__, 'tower_sim.bench.'):
    importlib.import_module(item.name)
from tower_sim.bench import source_receipt, verification
sys.modules['blocked_optional'] = None
startup = verification.imported_sources()
receipt = source_receipt.ImportReceipt()
try:
    complete = receipt.collect()
finally:
    receipt.close()
assert startup and complete['files']
assert 'blocked_optional' not in complete['modules']
for name in ('numpy', 'scipy', 'h5py', 'yaml', 'tower_sim.bench.environment'):
    assert name in complete['modules'], name
assert not {'isaacsim', 'omni', 'carb', 'pxr'} & set(sys.modules)
""",
        ROOT,
        SOURCES,
    )


def test_import_blocking_sentinel_has_no_payload():
    assert source_receipt.module_source("blocked_optional", None) is None


@pytest.mark.parametrize("namespace", ["omni", "six"])
def test_sdk_loader_provider_requires_matching_namespace_and_admitted_root(
    tmp_path, monkeypatch, namespace
):
    provider = tmp_path / "loader.py"
    provider.write_text("# admitted SDK loader fixture\n")
    provider_name = namespace + ".loader" if namespace == "omni" else "six"
    loader_type = type("MetaPathImporter", (), {"__module__": provider_name})
    loader = loader_type()
    name = namespace + ".moves"
    module = ModuleType(name)
    module.__loader__ = loader
    module.__spec__ = ModuleSpec(name, loader)
    monkeypatch.setitem(sys.modules, provider_name, NS(__file__=str(provider)))
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(verification, "sdk_roots", lambda: [str(tmp_path)])
    assert source_receipt.module_source(name, sys.modules[name]) == provider
    for foreign in ("carb.generated", "foreign.generated"):
        with pytest.raises(ValueError, match="Unbound non-file import"):
            source_receipt.module_source(foreign, module)
    monkeypatch.setattr(verification, "sdk_roots", lambda: [])
    with pytest.raises(ValueError, match="Unbound non-file import"):
        source_receipt.module_source(name, module)
    monkeypatch.setattr(verification, "sdk_roots", lambda: [str(tmp_path)])
    provider.unlink()
    with pytest.raises(ValueError, match="Unbound non-file import"):
        source_receipt.module_source(name, module)


@pytest.mark.parametrize(
    "name,kind,filename",
    [
        ("isaacsim.simulation_app", "module", None),
        ("isaacsim.simulation_app.SimulationApp", "class", None),
        ("isaacsim.simulation_app.AppFramework", "class", None),
        ("swig_runtime_data4", "module", None),
        ("torch._VF", "subclass", None),
        ("torch.ops", "subclass", "_ops.py"),
        ("torch.classes", "subclass", "_classes.py"),
    ],
)
def test_runtime_injected_objects_have_no_separate_payload(
    name, kind, filename, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verification, "sdk_roots", lambda: [])
    if kind == "class":
        module = type(name.rsplit(".", 1)[-1], (), {})
    elif kind == "subclass":
        module = type("RuntimeModule", (ModuleType,), {})(name)
    else:
        module = ModuleType(name)
    if filename:
        module.__file__ = filename
    monkeypatch.setitem(sys.modules, name, module)
    assert source_receipt.module_source(name, sys.modules[name]) is None
    # Existing payloads must still bind, including relative files.
    payload = tmp_path / "runtime.py"
    payload.write_text("VALUE = 1\n")
    module.__file__ = payload.name
    assert source_receipt.module_source(name, module) == payload


@pytest.mark.parametrize("metadata", ["loader", "spec", "spec-loader"])
def test_runtime_admission_never_bypasses_importer(metadata, tmp_path, monkeypatch):
    monkeypatch.setattr(verification, "sdk_roots", lambda: [])
    module = ModuleType("foreign.runtime")
    module.__file__ = str(tmp_path / "missing.py")
    if metadata == "loader":
        module.__loader__ = object()
    else:
        module.__spec__ = ModuleSpec(
            module.__name__, object() if metadata == "spec-loader" else None
        )
    with pytest.raises(ValueError, match="Unbound non-file import"):
        source_receipt.module_source(module.__name__, module)


@pytest.mark.parametrize("name", ["os.evil", "time.evil", "abc.evil"])
def test_unrelated_stdlib_parent_does_not_admit_payload(name):
    with pytest.raises(ValueError, match="Unbound non-file import"):
        source_receipt.module_source(name, NS(__spec__=NS(loader=None)))


@pytest.mark.parametrize(
    "reference", [False, pytest.param(True, marks=pytest.mark.archive)], ids=["public", "frozen"]
)
def test_outside_package_construction_uses_verified_calibration(tmp_path, reference):
    packages = tmp_path / "packages"
    packages.mkdir()
    for source in SOURCES:
        for name in ("tower_sim", "tower_eval", "nero_kinematics", "nero_eef_ik"):
            if (source / name).is_dir():
                shutil.copytree(
                    source / name, packages / name, ignore=shutil.ignore_patterns("__pycache__")
                )
    data = packages / "tower_eval/_data"
    data.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "benchmarks/2026.11/tasks.json", data / "tasks-2026.11.json")
    from scripts.maintainer.archive_root import archive_checkout

    original = (
        (
            archive_checkout() / "packages/tower-sim/src/tower_sim/backend/runtime_profiles.py"
        ).read_text()
        if reference
        else None
    )
    code = (
        f"ROOT = {str(ROOT)!r}\nORIGINAL = {original!r}\n"
        + """
import json, types
from pathlib import Path
import numpy as np
from tower_sim.backend import assets, runtime_profiles
from tower_sim.bench import environment
from tower_sim.bench.resources import Resources
assert assets.REPORT.is_file(), assets.REPORT
resources = Resources.from_roots(checkout=ROOT, assets=ROOT)
report_path = resources.path('calibration_report')
report_bytes = report_path.read_bytes()
report = resources.read('calibration_report')
profiles = resources.path('runtime_profiles')
# Only native stage I/O is stubbed: retain the real calibration bytes, loaders,
# manifest checks, arithmetic and all returned arrays. No USD is opened on CPU.
stage = Path(ROOT) / '__cpu_native_stage_stub__'
checksum = assets.checksum
stage_hash = report['stage']['source_sha256_after']
assets.checksum = lambda path: stage_hash if Path(path) == stage else checksum(path)
if ORIGINAL is None:
    expected = runtime_profiles.load_runtime_profiles(profiles, stage_hash)
else:
    reference = types.ModuleType('tower_sim.backend._legacy_profile_reference')
    reference.__package__ = 'tower_sim.backend'
    ORIGINAL = ORIGINAL.replace('from .model import ARM_ORDER, DEFAULT_RUNTIME_PROFILES, DT, JOINT_NAMES, digest', 'from tower_eval.simulator.spec import ARM_ORDER, DT, JOINT_NAMES, digest\\nfrom tower_sim.config import DEFAULT_RUNTIME_PROFILES')
    exec(compile(ORIGINAL, '<legacy runtime profile>', 'exec'), reference.__dict__)
    reference.REPORT = report_path
    expected = reference.load_runtime_profiles(profiles, stage_hash)
assert runtime_profiles.load_runtime_profiles.__kwdefaults__['calibration_report'] == assets.REPORT
# Poison only the global/default path: construction must honor supplied Resources.
runtime_profiles.REPORT = Path(ROOT) / '__invalid_default_calibration__'
runtime_profiles.load_runtime_profiles.__kwdefaults__['calibration_report'] = runtime_profiles.REPORT
# Stub SDK import names only, stopping immediately before the model is built.
def module(name, **attrs):
    value = types.ModuleType(name)
    value.__dict__.update(attrs)
    sys.modules[name] = value
    if '.' in name:
        parent, key = name.rsplit('.', 1)
        if parent not in sys.modules:
            module(parent)
        setattr(sys.modules[parent], key, value)
    return value
for name in ('omni.physx', 'omni.timeline', 'omni.usd'):
    module(name)
module('isaacsim.core.api', SimulationContext=object)
module('isaacsim.core.cloner', Cloner=object)
module('isaacsim.core.prims', Articulation=object, RigidPrim=object)
module('pxr', **{name: object for name in ('Gf', 'PhysxSchema', 'Usd', 'UsdGeom', 'UsdPhysics', 'UsdShade')})
module('carb', settings=types.SimpleNamespace(get_settings=lambda: types.SimpleNamespace(set=lambda *args: None)))
config = environment.config_class()()
config.camera_batch_mode = 'individual'
config.stage = str(stage)
config.runtime_profile_dir = str(profiles)
cls = environment.environment_class()
obj = cls.__new__(cls)
class ReachedModel(Exception): pass
def model_factory(*args):
    actual = obj.runtime_profile
    actual['provenance'].pop('generator_gripper_effort_application')
    encode = lambda value: json.dumps(value, sort_keys=True, default=lambda item: item.tolist())
    assert encode(actual) == encode(expected)
    assert obj.report == report
    assert actual['provenance']['reference_report'] == str(report_path)
    assert actual['provenance']['reference_report_sha256'] == checksum(report_path)
    assert report_path.read_bytes() == report_bytes
    raise ReachedModel
try:
    obj.__init__(None, config, [object()], np.zeros(16), Path(ROOT), model_factory=model_factory,
                 renderer=None, prepare_scene=None, apply_drives=None, clone_hook=None,
                 resources=resources, robot_config={})
except ReachedModel:
    pass
else:
    raise AssertionError('did not reach the model construction boundary')
"""
    )
    isolated(code, ROOT, [packages])
