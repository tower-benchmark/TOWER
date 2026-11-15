"""Build a CPU-only public release fixture; only Kit construction is replaced."""

PROBE = r"""
import hashlib, importlib, json, os, shutil, subprocess, sys, types, zipfile
from pathlib import Path
source, scratch = map(Path, sys.argv[1:3])
prepare_rows = sys.argv[3] == "prepare"
checkout = scratch / "checkout"
installed = scratch / "installed"
portable = scratch / "sdk"
generated = portable / "cache/ogn_generated"
checkout.mkdir(); installed.mkdir(); generated.mkdir(parents=True)
for name in ("tools", "scripts", "benchmarks"):
    shutil.copytree(source / name, checkout / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
modules = {
    "tower_sim": "packages/tower-sim/src/tower_sim",
    "tower_eval": "src/tower-eval/src/tower_eval",
    "nero_kinematics": "packages/nero-kinematics/src/nero_kinematics",
    "nero_eef_ik": "packages/nero-kinematics/src/nero_eef_ik",
}
for name, relative in modules.items():
    shutil.copytree(source / relative, installed / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
# Match the evaluator wheel's force-included public resources.
for source_name, destination in (
    ("benchmarks/2026.11/tasks.json", "_data/tasks-2026.11.json"),
    ("src/tower-eval/examples/random_policy/README.md", "examples/random_policy/README.md"),
    ("src/tower-eval/examples/random_policy/model.py", "examples/random_policy/model.py"),
):
    path = installed / "tower_eval" / destination; path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / source_name, path)
# A real file-backed SDK namespace supplies location metadata; no SDK function runs.
(generated / "isaacsim").mkdir()
(generated / "isaacsim/__init__.py").write_text("")
sys.path[:0] = [str(checkout), str(checkout / "tools/simulator"), str(installed), str(generated)]
sys.argv = ["public-worker-fixture", "--portable-root", str(portable)]
os.environ.pop("TOWER_ARCHIVE_ROOT", None)
from scripts.maintainer import archive_root

def forbidden(*args, **kwargs):
    raise AssertionError("Public worker read the private archive")
archive_root.archive_bundle = forbidden
from tower_sim.bench import bootstrap, verification as v
from tower_sim.bench.resources import Resources, sha
import benchmark_assets

# Synthetic payload identity lives only in the isolated fixture package.
data = scratch / "data"; data.mkdir()
# Minimal synthetic scene, validated by the real scene codec and resolver below.
from tower_sim.bench.scene_format import scene_schema
schema = scene_schema()
asset_sha = hashlib.sha256(b"{}").hexdigest()
def sample(rule):
    if "$ref" in rule: return sample(schema["$defs"][rule["$ref"].split("/")[-1]])
    if "const" in rule: return rule["const"]
    if "enum" in rule: return rule["enum"][0]
    kind = rule["type"]
    if isinstance(kind, list): kind = "null" if "null" in kind else kind[0]
    if kind == "null": return None
    if kind == "object": return {key:sample(rule["properties"][key]) for key in rule.get("required", [])}
    if kind == "array": return [sample(rule["items"]) for _ in range(rule.get("minItems", 0))]
    if kind == "boolean": return False
    if kind in ("integer", "number"): return rule.get("minimum", 0)
    if "sha256:" in rule.get("pattern", ""): return "sha256:" + asset_sha
    if rule.get("pattern") == "[0-9a-f]{64}": return asset_sha
    return "fixture"
scene = sample(schema); scene["scene_id"] = "scene-000"
scene["episode"]["episode_spec"]["layout_metadata"].update(catalog_task=1, evaluation_task="task1-2", movable_layers=2)
payload = data / "scenes/task1-2.json"; payload.parent.mkdir()
payload.write_text(json.dumps(dict(schema="tower-scenes-v2", task="task1-2", scenes=[scene])))
payload_files = {"scenes/task1-2.json":dict(size=payload.stat().st_size, sha256=sha(payload))}
(data / "MANIFEST.json").write_text(json.dumps(dict(schema="tower-staged-release-v1", files=payload_files)))
manifest_path = installed / "tower_sim/bench/public-data.json"
document = json.loads(manifest_path.read_text())
document.update(payload_files=payload_files, upload_files=payload_files, manifest_sha256=sha(data / "MANIFEST.json"))
for name, pin in document["files"].items():
    if isinstance(pin, dict):
        path = checkout / name
        lock = json.loads(path.read_text()); lock["manifest_sha256"] = document["manifest_sha256"]
        path.write_text(json.dumps(lock)); pin["manifest_sha256"] = document["manifest_sha256"]
manifest_path.write_text(json.dumps(document))
assets = scratch / "assets"; assets.mkdir()
resources = Resources.from_roots(checkout=checkout, assets=assets)
for name, path in resources.paths.items():
    if path.is_relative_to(assets):
        if name == "kinematics":
            path.mkdir(parents=True, exist_ok=True)
            for filename in ("nero.urdf", "nero.yaml", "model.json"):
                (path / filename).write_text("synthetic CPU fixture")
        else:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
files = [dict(path=p.relative_to(assets).as_posix(), size=p.stat().st_size, sha256=sha(p))
         for p in assets.rglob("*") if p.is_file()]
(assets / "manifest.json").write_text(json.dumps(dict(
    schema="tower-benchmark-assets-v1", release="fixture", files=files,
    scene_index_sha256=benchmark_assets.SCENE_INDEX_SHA256)))
# Change synthetic content pins only in this isolated fixture installation.
for name, key, value in (
    ("benchmark/assets.lock.json", "manifest_sha256", sha(assets / "manifest.json")),
    ("benchmark/runtime/ik-capsule/runtime/scene-cache.json", "scene_sha256", sha(resources.path("scene"))),
):
    path = installed / "tower_sim/data" / name
    body = json.loads(path.read_text()); body[key] = value; path.write_text(json.dumps(body))
pins_path = installed / "tower_sim/data/pins.json"
pins = json.loads(pins_path.read_text())
for name in pins: pins[name] = sha(installed / "tower_sim" / name)
pins_path.write_text(json.dumps(pins))
(checkout / ".gitignore").write_text("dist/\n")
for command in (["git", "init", "-q", str(checkout)], ["git", "-C", str(checkout), "add", "."],
                ["git", "-C", str(checkout), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "fixture"]):
    subprocess.run(command, check=True)
commit = v.git(checkout, "rev-parse", "HEAD").decode().strip()
dist = checkout / "dist"; dist.mkdir()
packages = {}
for name, version in v.PACKAGES.items():
    wheel = dist / (name + ".whl")
    with zipfile.ZipFile(wheel, "w") as archive:
        for module in v.PACKAGE_MODULES[name]:
            for path in (installed / module).rglob("*"):
                if path.is_file(): archive.write(path, path.relative_to(installed).as_posix())
    packages[name] = dict(version=version, wheel=wheel.name, sha256=sha(wheel))
release = dict(schema="tower-release-v2", public=v.envelope(), commit=commit, packages=packages,
               checkout_trees=list(v.CHECKOUT_TREES),
               checkout_files={name:sha(checkout / name) for name in v.checkout_files(checkout)})
(dist / "release-manifest.json").write_text(json.dumps(release))

calls = []
def construct(**kwargs):
    calls.append(kwargs)
    output = kwargs["output"]; output.mkdir(parents=True)
    shutil.copytree(kwargs["resources"].path("kinematics"), output / "kinematics")
    return types.SimpleNamespace(output=output, close=lambda: None)
bootstrap.bootstrap = construct
public = importlib.import_module("benchmark_candidate")
runner = public.create_runner(benchmark_version="2026.11.1", data_root=data, v1_root=None)
request = types.SimpleNamespace(task="task1-2", seed=0, assets=assets, output=scratch / "trial", gpu=0, frozen_components=())
prepared = runner.setup(request, rows=None if prepare_rows else [])
assert len(prepared.rows) == int(prepare_rows)
if prepare_rows:
    assert prepared.rows[0]["portable_initial_state_only"]
    assert prepared.rows[0]["candidate"]["provenance"]["asset_closure_verified"]
try:
    assert len(calls) == 1
    bound = prepared.collect_sources()
    assert bound[str((assets / "manifest.json").resolve())] == sha(assets / "manifest.json")
    receipt = json.loads((request.output / "imported-sources.json").read_text())
    assert receipt["public_closure"]["passed"] and receipt["validation_policy"]["forbidden_roots"] == []
    # Both filename and module-alias guards must survive the absence of archive paths.
    for module_name, filename in (("guard_probe", "control.py"), ("guard_probe.control", "neutral.py")):
        path = generated / filename; path.write_text("value = 1\n")
        module = types.ModuleType(module_name); module.__file__ = str(path)
        sys.modules[module_name] = module
        try:
            prepared.collect_sources()
        except ValueError as error:
            assert "Forbidden candidate import" in str(error), str(error)
        else:
            raise AssertionError("Pinned archive stem guard did not reject the loaded module")
        finally:
            del sys.modules[module_name]
    prepared.collect_sources()
    manifest = assets / "manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    try:
        prepared.collect_sources()
    except ValueError as error:
        assert "runtime_resources" in str(error), str(error)
    else:
        raise AssertionError("Changed public asset manifest was not rejected")
finally:
    prepared.close()
print(json.dumps(dict(passed=True, archive_environment_present="TOWER_ARCHIVE_ROOT" in os.environ,
                     bootstrap_calls=len(calls), prepared_rows=len(prepared.rows), public_closure=True, filename_and_alias_rejected=True)))
"""
