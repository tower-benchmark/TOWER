"""Frozen code is loaded by path in tests only; numerical configuration is unchanged."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scripts.maintainer.archive_root import ArchivedPath, ArchivedToolPath

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ArchivedPath("runtime")


def load_frozen(name, relative):
    from scripts.maintainer.archive_root import archive_bundle

    RUNTIME = archive_bundle() / "runtime"
    spec = importlib.util.spec_from_file_location(name, RUNTIME / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def frozen_loader():
    return load_frozen


@pytest.fixture
def frozen_control(monkeypatch):
    monkeypatch.syspath_prepend(str(ArchivedToolPath("")))
    return load_frozen("_p705_control", ArchivedToolPath("control.py"))


@pytest.fixture
def compensation(monkeypatch, tmp_path):
    """Exercise the real import verifier with a synthetic station, not a claimed pilot.

    Only the scene byte/hash binding is replaced at the I/O boundary. Every mass,
    COM, compliance coefficient and algorithm comes from the immutable sources.
    The production from_cache verifier is tested separately with mismatches too.
    """
    from tower_sim.backend.resources import DATA

    cache_path = DATA / "benchmark/runtime/ik-capsule/runtime/scene-cache.json"
    cache = json.loads(cache_path.read_text())
    scene = tmp_path / "station/station-dual-j2-wrist-shared-v3.usdz"
    scene.parent.mkdir()
    scene.write_bytes(b"explicit CPU-only synthetic station")
    cache["scene_sha256"] = hashlib.sha256(scene.read_bytes()).hexdigest()
    cache_file = tmp_path / "scene-cache.json"
    cache_file.write_text(json.dumps(cache))
    read_text = Path.read_text
    original_cache = Path(RUNTIME / "ik-capsule/runtime/scene-cache.json")

    def read(path, *args, **kwargs):
        if path in (cache_path, original_cache):
            return json.dumps(cache)
        return read_text(path, *args, **kwargs)

    fk = load_frozen("_p705_urdf_fk", "ik-capsule/runtime/urdf_fk.py")
    monkeypatch.setitem(sys.modules, "urdf_fk", fk)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setenv("TOWER_ASSET_ROOT", str(tmp_path))
    original = load_frozen("_p705_j2comp", "ik-capsule/runtime/j2comp.py")
    return original, cache["configuration"], cache_file, scene


@pytest.fixture
def frozen_import_dependencies(monkeypatch):
    """Supply relocated shared dependencies to the archived CPU test oracle only.

    The original bootstrap and camera sources stay byte-identical in the archive.
    Their removed import names resolve to the same shared modules as before purge;
    Config uses the independent pre-split field/default manifest and v1 validator.
    No generator implementation, layout or planner is supplied.
    """
    import importlib
    from types import ModuleType

    from test_config_split import baseline_config
    from tower_eval.simulator import commands, spec
    from tower_sim._config_v1 import validate

    from scripts.maintainer.archive_root import archive_bundle

    archive_bundle()
    prefix = "jenga_" + "generator"
    package = ModuleType(prefix)
    package.__path__ = []
    model = ModuleType(prefix + ".model")
    model.__dict__.update({k: v for k, v in vars(spec).items() if not k.startswith("__")})
    model.Config = baseline_config()
    model.Config.validate = validate
    monkeypatch.setitem(sys.modules, prefix, package)
    monkeypatch.setitem(sys.modules, prefix + ".model", model)
    package.model = model
    for name in ("backend", "physics_scene", "profiles"):
        module = importlib.import_module("tower_sim.backend." + name)
        monkeypatch.setitem(sys.modules, prefix + "." + name, module)
        setattr(package, name, module)
    monkeypatch.setitem(sys.modules, "tower_eval.simulator." + "isaac", commands)
    replay_prefix = "replay_runner_" + "isaac_sim"
    replay_package = ModuleType(replay_prefix)
    replay_package.__path__ = []
    worker = importlib.import_module("tower_sim.replay.dual_replay_worker")
    replay_package.dual_replay_worker = worker
    monkeypatch.setitem(sys.modules, replay_prefix, replay_package)
    monkeypatch.setitem(sys.modules, replay_prefix + ".dual_replay_worker", worker)
