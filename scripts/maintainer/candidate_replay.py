"""Maintainer-only binding of verified release data/builds to tower-sim replay."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import subprocess
import sys
import sysconfig

sys.dont_write_bytecode = True
import zipfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import cast

import numpy as np
from tower_sim.bench.resources import Resources, save, sha

from scripts.maintainer.gpu_host_profile import reference_contract, runtime_profile
from scripts.maintainer.gpu_identity import require_equal
from scripts.maintainer.gpu_runner import ReplayEvidence

ROOT = Path(__file__).resolve().parents[2]


def aggregate(mapping):
    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def archive_data_identity(root, assets, *, files=None, v1_root=None):
    from tower_sim.bench.asset_identity import asset_manifest_identity

    def hashed(path):
        value = sha(path)
        if files is not None:
            files[str(path.resolve())] = value
        return value

    from scripts.maintainer.archive_root import archive_bundle

    hashed(assets / "manifest.json")
    bundle = archive_bundle(v1_root)
    archive = bundle.parents[1]
    groups = {
        "scenes": [p for p in (bundle / "scenes").rglob("*") if p.is_file()],
        "physics_resources": [
            p
            for p in (bundle / "runtime").rglob("*")
            if p.suffix in (".json", ".urdf", ".yaml", ".yml")
        ]
        + list((archive / "src/tower-datagenerator/jenga_generator/data").glob("*.json")),
    }
    data = {
        name + "_sha256": aggregate({p.relative_to(archive).as_posix(): hashed(p) for p in paths})
        for name, paths in groups.items()
    }
    data.update(
        assets_manifest_sha256=asset_manifest_identity(hashed(assets / "manifest.json")),
        corpus_manifest_sha256=hashed(archive / "tests/golden/benchmark-2026.11/MANIFEST.json"),
        protocol_sha256=hashed(bundle / "protocol.json"),
        scene_index_sha256=hashed(bundle / "scene-index.json"),
    )
    require_equal(reference_contract()["contract"]["immutable_data"], data, "immutable_data")
    return data


def asset_identity(assets, *, files=None):
    """Bind the public asset manifest and every member, with no archive lookup."""

    def hashed(path):
        value = sha(path)
        if files is not None:
            files[str(path.resolve())] = value
        return value

    identity = {"assets_manifest_sha256": hashed(assets / "manifest.json")}
    manifest = json.loads((assets / "manifest.json").read_text())
    seen = set()
    for row in manifest["files"]:
        path = (assets / row["path"]).resolve()
        if path in seen or not path.is_relative_to(assets.resolve()) or not path.is_file():
            raise ValueError("Unsafe or duplicate asset member")
        seen.add(path)
        if path.stat().st_size != row["size"] or hashed(path) != row["sha256"]:
            raise ValueError(f"Changed asset: {row['path']}")
    return identity


def data_identity(root, assets, *, files=None, v1_root=None):
    """Keep the full archived comparison contract for maintainer identities."""
    data = archive_data_identity(root, assets, files=files, v1_root=v1_root)
    asset_identity(assets, files=files)
    return data


EVAL_RESOURCES = {
    "tower_eval/_data/tasks-2026.11.json": "benchmarks/2026.11/tasks.json",
    "tower_eval/examples/random_policy/README.md": "src/tower-eval/examples/random_policy/README.md",
    "tower_eval/examples/random_policy/model.py": "src/tower-eval/examples/random_policy/model.py",
}


def wheel_source_path(member):
    if member in EVAL_RESOURCES:
        return EVAL_RESOURCES[member]
    name = member.split("/", 1)[0]
    roots = {
        "tower_sim": "packages/tower-sim/src",
        "tower_eval": "src/tower-eval/src",
        "nero_kinematics": "packages/nero-kinematics/src",
        "nero_eef_ik": "packages/nero-kinematics/src",
    }
    if ".." in Path(member).parts or name not in roots:
        raise ValueError(f"Unexpected wheel member: {member}")
    return f"{roots[name]}/{member}"


def frozen_module_stems(root, *, v1_root=None):
    from scripts.maintainer.archive_root import archive_bundle

    manifest = json.loads((archive_bundle(v1_root) / "SOURCE_MANIFEST.json").read_text())
    return {
        Path(name).stem
        for name in manifest["files"]
        if name.startswith("runtime/") and name.endswith(".py") and Path(name).stem != "__init__"
    }


def expected_import_roots(root):
    # Manifest stems such as model/tasks/control also occur in these legitimate
    # packages. A matching namespace alone is insufficient: the path must belong
    # to that loaded package. Bare names and external reference or storage copies fail.
    packages = (
        "tower_sim",
        "tower_eval",
        "nero_kinematics",
        "nero_eef_ik",
        "asyncio",
        "coverage",
        "omni",
        "torchgen",
        "torch",
        "isaacsim",
        "botocore",
        "s3transfer",
        "boto3",
        "jinja2",
    )
    roots: dict[str, list[str]] = {
        name: list(getattr(sys.modules.get(name), "__path__", ())) for name in packages
    }
    # Kit's extension importer does not expose all extension namespaces through
    # omni.__path__. Enumerate only namespace directories inside this loaded SDK;
    # arbitrary loaded-module paths and external/symlinked copies remain forbidden.
    sdk_file = getattr(sys.modules.get("isaacsim"), "__file__", None)
    if sdk_file:
        roots["omni"].extend(sdk_omni_roots(Path(sdk_file).resolve().parent))
    # Conda builds pyexpat as an extension rather than a builtin. Its inert
    # `model` submodule shares an archive stem; bind only the interpreter's
    # exact native payload, never a namespace-wide standard-library directory.
    expat_file = getattr(sys.modules.get("pyexpat"), "__file__", None)
    if expat_file:
        expat = Path(expat_file).resolve()
        native_stdlib = Path(sysconfig.get_path("stdlib")).resolve() / "lib-dynload"
        if (
            expat.parent == native_stdlib
            and expat.name in {"pyexpat" + suffix for suffix in EXTENSION_SUFFIXES}
            and expat.is_file()
        ):
            roots["pyexpat"] = roots["xml.parsers.expat"] = [str(expat)]
    return roots


def sdk_omni_roots(sdk):
    paths = []
    for category in ("exts", "extscache", "extsDeprecated"):
        for path in sorted((sdk / category).glob("*/omni")):
            resolved = path.resolve()
            if resolved.is_relative_to(sdk) and resolved.is_dir():
                paths.append(str(resolved))
    return paths


def expected_entrypoints(root, release=None):
    entries = {"__main__": []}
    from tower_sim.bench.verification import ENTRYPOINTS

    for relative in ENTRYPOINTS:
        stem = Path(relative).stem
        if release is not None:
            expected = release["checkout_files"][relative]
        else:
            committed = subprocess.check_output(
                ["git", "-C", str(root), "show", f"HEAD:{relative}"]
            )
            expected = hashlib.sha256(committed).hexdigest()
        binding = dict(path=str((root / relative).resolve()), sha256=expected)
        entries["__main__"].append(binding)
        entries[f"scripts.maintainer.{stem}"] = [binding]
        entries[stem] = [binding]
    return entries


class RuntimeProvenance:
    """Snapshot actual resource roots before construction and recheck at collection."""

    def __init__(
        self,
        root,
        assets,
        resources,
        *,
        benchmark_version="2026.11.1",
        data_root=None,
        v1_root=None,
    ):
        import nero_kinematics

        self.root, self.assets, self.resources = Path(root), Path(assets), resources
        self.benchmark_version, self.data_root = benchmark_version, data_root
        self.v1_root = v1_root
        self.files = {}
        self.files.update(self.scene_files())
        self.bind_data(self.files)
        package = Path(nero_kinematics.__file__).resolve().parent
        self.roots = [
            *(resources.path(name) for name in resources.paths),
            package / "profiles",
            package / "models",
        ]
        self.files.update(self.snapshot())

    def bind_data(self, files):
        if self.v1_root is not None or self.benchmark_version == "2026.11.0":
            data_identity(self.root, self.assets, files=files, v1_root=self.v1_root)
        else:
            asset_identity(self.assets, files=files)

    def scene_files(self) -> dict[str, str]:
        from tower_sim.bench.verification import verify_data, verify_package_data

        if self.benchmark_version == "2026.11.0":
            from scripts.maintainer.archive_scene import verify_archive_data

            return verify_archive_data(self.root, v1_root=self.v1_root) | verify_package_data()
        return cast(
            dict[str, str],
            verify_data(
                self.root,
                benchmark=self.benchmark_version,
                assets=self.assets,
                data_root=self.data_root,
                v1_root=self.v1_root,
            )["files"]
            | verify_package_data(),
        )

    def snapshot(self):
        files = {}
        for root in self.roots:
            paths = root.rglob("*") if root.is_dir() else (root,)
            for path in paths:
                if path.is_file() and "__pycache__" not in path.parts:
                    files[str(path.resolve())] = sha(path)
        return files

    def bind_runtime(self, directory):
        # Bootstrap reads these copies. Bind them to the pre-construction originals,
        # never to a fresh unchecked digest after they have already been consumed.
        for name in ("nero.urdf", "nero.yaml", "model.json"):
            source = self.resources.path("kinematics") / name
            path = directory / name
            expected = self.files[str(source.resolve())]
            require_equal(expected, sha(path), f"runtime_resource/{path}")
            self.files[str(path.resolve())] = expected
        self.roots.append(directory)

    def verify(self):
        current = self.scene_files()
        self.bind_data(current)
        current.update(self.snapshot())
        require_equal(self.files, current, "runtime_resources")
        return current


def build_identity(root):
    """Bind clean committed sources, wheel payloads, loaded paths and dependency bytes."""
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if status.strip():
        raise ValueError(
            "Candidate identity requires a clean committed checkout (including untracked files)"
        )
    wheels, imports = {}, {}
    for name in ("tower_sim", "tower_eval", "nero_kinematics"):
        paths = list((root / "dist").glob(f"{name}-*.whl"))
        if len(paths) != 1:
            raise ValueError(f"Exactly one built {name} wheel required in dist")
        wheel = paths[0]
        module = importlib.import_module(name)
        assert module.__file__ is not None
        location = Path(module.__file__).resolve().parent
        imports[name] = str(Path(module.__file__).resolve())
        with zipfile.ZipFile(wheel) as archive:
            members = [
                n for n in archive.namelist() if n.startswith(name + "/") and not n.endswith("/")
            ]
            for member in members:
                committed_path = wheel_source_path(member)
                committed = subprocess.check_output(
                    ["git", "-C", str(root), "show", f"HEAD:{committed_path}"]
                )
                if archive.read(member) != committed:
                    raise ValueError(f"Wheel differs from committed source: {member}")
                installed = location / member.removeprefix(name + "/")
                # These three force-included evaluator files live outside its package root.
                if not installed.is_file() and member in EVAL_RESOURCES:
                    installed = root / EVAL_RESOURCES[member]
                if not installed.is_file() or installed.read_bytes() != archive.read(member):
                    raise ValueError(f"Loaded source differs from wheel: {member}")
        wheels[wheel.name] = sha(wheel)
    dependencies = {}
    for name in ("numpy", "scipy", "h5py", "Pillow", "PyYAML", "isaacsim"):
        distribution = importlib.metadata.distribution(name)
        files = {
            str(f): sha(distribution.locate_file(f))
            for f in distribution.files or []
            if not str(f).endswith(".pyc") and Path(str(distribution.locate_file(f))).is_file()
        }
        if not files:
            raise ValueError(f"No installed dependency bytes: {name}")
        dependencies[f"{name}-{distribution.version}-installed-content"] = aggregate(files)
    return commit, wheels, dependencies, imports


def execution_contract(root, data, profile, *, v1_root=None):
    """Report the fixed replay recipe and loaded protocol, independently of expectations.

    Bootstrap, drives and loop contracts test these effective rates/modes against
    the original executable behavior. There are no user-selectable physics knobs.
    backend.SimulationContext starts at 1/120; bootstrap forces individual cameras,
    sets 1/400 and applies v4; execute steps at 1/400 and records every 20 ticks.
    These describe that fixed configuration, not an additional live measurement.
    """
    from scripts.maintainer.archive_root import archive_bundle

    protocol = json.loads((archive_bundle(v1_root) / "protocol.json").read_text())
    return dict(
        benchmark="benchmark-2026.11.0",
        data_version="benchmark-2026.11.0",
        hardware_runtime_profile=profile,
        immutable_data=data,
        physics=dict(
            bootstrap_hz=120,
            camera_batch_mode="individual",
            command_mode="command_replay",
            drive_profile="v4",
            record_hz=20,
            simulation_hz=protocol["simulation"]["physics_hz"],
        ),
        task_order=list(protocol["tasks"]),
    )


class JsonSink:
    def __init__(self, path):
        self.stream = Path(path).open("x")  # noqa: SIM115 -- owned until replay finally

    def emit(self, record):
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()


class CandidateRunner:
    def __init__(
        self,
        *,
        root=ROOT,
        public=False,
        benchmark_version="2026.11.1",
        data_root=None,
        v1_root=None,
    ):
        self.root = Path(root).resolve()
        self.public = public
        self.benchmark_version = benchmark_version
        self.data_root = data_root
        self.v1_root = v1_root
        self.release = None
        self.shader_cache = None

    def identity(self, assets, gpu, frozen_components, reference_root):
        # Substitutions have a separate explicit diagnostic factory, never fallback.
        if frozen_components:
            from scripts.maintainer.candidate_substitutions import validate

            validate(reference_root, frozen_components)
        contract = reference_contract()["contract"]
        data = data_identity(self.root, Path(assets), v1_root=self.v1_root)
        profile = runtime_profile(gpu)
        require_equal(contract["hardware_runtime_profile"], profile, "hardware_runtime_profile")
        actual_contract = execution_contract(self.root, data, profile, v1_root=self.v1_root)
        require_equal(contract, actual_contract, "contract")
        if self.public:
            from tower_sim.bench.verification import (
                dependency_identity,
                verify_data,
                verify_release,
                verify_startup,
            )

            verify_data(
                self.root,
                benchmark=self.benchmark_version,
                assets=assets,
                data_root=self.data_root,
                v1_root=self.v1_root,
            )
            release = verify_release(self.root / "dist/release-manifest.json", checkout=self.root)
            verify_startup(release)
            self.release = release
            commit = release["commit"]
            wheels = {p["wheel"]: p["sha256"] for p in release["packages"].values()}
            dependencies = {
                name: aggregate(row) for name, row in dependency_identity()["packages"].items()
            }
            imports = {
                name.replace("-", "_"): row["path"]
                for name, row in release["installed_packages"].items()
            }
        else:
            commit, wheels, dependencies, imports = build_identity(self.root)
        from tower_sim.bench.verification import envelope

        if self.benchmark_version == "2026.11.1":
            from scripts.maintainer.scene_equivalence import prove

            actual_contract = dict(
                actual_contract,
                benchmark="benchmark-2026.11.1",
                data_version="benchmark-2026.11.1",
                scene_equivalence=prove(self.root, assets, self.data_root, v1_root=self.v1_root),
            )
        return dict(
            candidate_identity=dict(
                schema="tower-gpu-candidate-v2",
                implementation="tower-sim",
                benchmark_version=self.benchmark_version,
                commit=commit,
                wheel_sha256=wheels,
                package_import_paths=imports,
                dependency_sha256=dependencies,
                data_version=actual_contract["data_version"],
                frozen_components=list(frozen_components),
                **(
                    {"public": envelope(self.release, benchmark=self.benchmark_version)}
                    if self.public
                    else {}
                ),
            ),
            contract=actual_contract,
        )

    def setup(self, request, *, rows=None, task=None):
        from tower_sim.bench import app, bootstrap
        from tower_sim.bench.source_receipt import ImportReceipt
        from tower_sim.bench.tasks import BenchTask, validate_spec

        if self.public:
            from tower_sim.bench.verification import (
                verify_data,
                verify_release,
            )

            verify_data(
                self.root,
                benchmark=self.benchmark_version,
                assets=request.assets,
                data_root=self.data_root,
                v1_root=self.v1_root,
            )
            self.release = verify_release(
                self.root / "dist/release-manifest.json", checkout=self.root
            )
        assets = importlib.import_module("benchmark_assets")
        resources = Resources.from_roots(checkout=self.root, assets=request.assets)
        provenance = RuntimeProvenance(
            self.root,
            request.assets,
            resources,
            benchmark_version=self.benchmark_version,
            data_root=self.data_root,
            v1_root=self.v1_root,
        )
        if rows is None:
            scenes = assets.prepare_scenes(
                request.task,
                [request.seed],
                request.assets,
                request.output / "scenes",
                reference=False,
                benchmark_version=self.benchmark_version,
                data_root=self.data_root,
                v1_root=self.v1_root,
            )
            rows = assets.load_rows(scenes)
        task = task or BenchTask.from_resources(request.task, resources)
        receipt = ImportReceipt()
        if self.public:
            from tower_sim.bench.verification import verify_startup

            verify_startup(self.release)
        factories = bootstrap.ConstructionFactories(app=receipt.app_factory(app.simulation_app))
        sdk_spec = importlib.util.find_spec("isaacsim")
        if sdk_spec is None or sdk_spec.origin is None:
            raise RuntimeError("Isaac SDK package location is unavailable")
        sdk = Path(sdk_spec.origin).resolve().parent
        context = None
        try:
            if "bootstrap" in request.frozen_components:
                from scripts.maintainer.candidate_substitutions import bootstrap_context

                context = bootstrap_context(request, resources, rows, task)
                receipt.phase = "after_simulation_app"
            else:
                context = bootstrap.bootstrap(
                    resources=resources,
                    rows=rows,
                    task=bootstrap.ConstructionTask(
                        task.identity.legacy_id, task.identity.layers, validate_spec
                    ),
                    output=request.output / "simulation",
                    gpu=request.gpu,
                    sdk=sdk,
                    fit_receipt_sha256=sha(resources.path("tcp_fit_receipt")),
                    factories=factories,
                    shader_cache=self.shader_cache,
                )
            provenance.bind_runtime(context.output / "kinematics")
            if self.release is not None:
                from tower_sim.bench.verification import envelope

                public = envelope(self.release, benchmark=self.benchmark_version)
                context.public_provenance = dict(
                    public=public,
                    **{
                        k: public[k]
                        for k in ("release_manifest_sha256", "commit", "data_manifest_sha256")
                    },
                )
            return PreparedReplay(
                request, context, rows, task, receipt, provenance=provenance, release=self.release
            )
        except BaseException:
            receipt.close()
            if context is not None:
                context.close()
            raise


class PreparedReplay:
    def __init__(self, request, context, rows, task, receipt, *, provenance=None, release=None):
        self.request, self.context, self.rows, self.task, self.receipt = (
            request,
            context,
            rows,
            task,
            receipt,
        )
        self.sink = None
        self.provenance = provenance
        self.release = release

    def collect_sources(self):
        from tower_sim.bench.source_receipt import validate_imports

        from scripts.maintainer.archive_root import archive_bundle

        if self.provenance is None:
            raise RuntimeError("Runtime resource provenance must be bound before replay")
        resources = self.provenance.verify()
        receipt = self.receipt.collect()
        receipt["resources"] = resources
        receipt["files"].update(resources)
        if (
            self.provenance.v1_root is not None
            or getattr(self.provenance, "benchmark_version", "2026.11.1") == "2026.11.0"
        ):
            forbidden_roots = [str(archive_bundle(self.provenance.v1_root) / "runtime")]
            forbidden_stems = frozen_module_stems(
                self.provenance.root, v1_root=self.provenance.v1_root
            )
        else:
            from tower_sim.bench.import_guard import matched_archive_stems

            forbidden_roots = []
            forbidden_stems = matched_archive_stems(receipt)
        validation_policy = dict(
            forbidden_roots=forbidden_roots,
            forbidden_stems=sorted(forbidden_stems),
            allowed_packages=expected_import_roots(self.provenance.root),
            allowed_modules=expected_entrypoints(self.provenance.root, self.release),
        )
        if self.release is not None:
            from tower_sim.bench.verification import sdk_roots, verify_release, verify_sources

            # Recheck the same release at completion, including HEAD and untracked code.
            release = verify_release(
                self.provenance.root / "dist/release-manifest.json",
                checkout=self.provenance.root,
                expected=self.release["manifest_sha256"],
            )
            roots = sdk_roots()
            verify_sources(
                {row["path"]: row["sha256"] for row in receipt["modules"].values()},
                release,
                allowed_roots=roots,
            )
            receipt["public_closure"] = dict(
                passed=True,
                allowed_sdk_roots=roots,
                release_manifest_sha256=release["manifest_sha256"],
                commit=release["commit"],
                imported_files=len(receipt["modules"]),
            )
        receipt["validation_policy"] = validation_policy
        save(self.request.output / "imported-sources.json", receipt)
        if not self.request.frozen_components:
            validate_imports(receipt, **validation_policy)
        return receipt["files"]

    def replay(self, commands):
        from tower_sim.bench import replay
        from tower_sim.bench.fast_observed_joints import observed_joint_validator
        from tower_sim.bench.j2comp import J2Compensation
        from tower_sim.bench.metrics import Metrics, TrialRequestID
        from tower_sim.bench.options import ControlOptions

        row = self.rows[0]
        np.testing.assert_allclose(
            commands.action_qpos[0], row["initial"]["qpos"], atol=2e-6, rtol=0
        )
        # Preserve frozen bootstrap prevalidation's dense tick extent (20 for initial-only rows).
        row.update(times=commands.time_s, actions=commands.action_qpos)
        # Cache is enabled only after the original bootstrap, as in the reference replay.
        self.context.compensation = J2Compensation.from_cache(
            self.context.resources.path("scene_cache"), self.context.resources.path("scene")
        )
        validator = observed_joint_validator(
            self.context.compensation, options=ControlOptions(observed_joints="fast")
        )
        factories = replay.ReplayFactories()
        if self.request.frozen_components:
            from scripts.maintainer.candidate_substitutions import components

            factories, validator = components(self.request, self.context, factories, validator)
        self.sink = JsonSink(self.request.output / "metrics.jsonl")
        metrics = Metrics(self.sink, TrialRequestID(self.request.task))

        try:
            code = replay.execute(
                self.context,
                self.rows,
                self.task,
                metrics=metrics,
                source_collector=self.collect_sources,
                observed_validator=validator,
                expert_sha256=self.request.expert_sha256,
                factories=factories,
            )
            if code != 0:
                raise RuntimeError(f"Candidate command replay failed: {code}")
            return collect(self.request.output, row, self.request)
        finally:
            save(
                self.request.output / "fast-observed-joints.json",
                dict(
                    certificate=getattr(validator, "certificate", None),
                    stats=getattr(validator, "stats", None),
                ),
            )
            metrics.close()
            self.sink.close()
            self.sink = None

    def close(self):
        self.receipt.close()
        if self.sink is not None:
            self.sink.close()
        self.context.close()


def collect(output, row, request=None):
    """All GPU-derived artifacts are durable before the harness closes Kit."""
    import h5py
    from PIL import Image

    trial = row["output"]
    status = json.loads((trial / "status.json").read_text())
    result = json.loads((output / "simulation/batch-policy-results.json").read_text())
    if (
        not status["evidence_complete"]
        or not result["completed"]
        or request is None
        or status["seed"] != request.seed
        or status["checkpoint_package_digest"] != request.expert_sha256
    ):
        raise ValueError("Cannot collect incomplete or differently bound candidate evidence")
    with h5py.File(trial / "episode/states.hdf5") as stream:
        np.savez_compressed(
            output / "trace.npz",
            **{key: cast(h5py.Dataset, stream[key])[:] for key in ("time", "qpos", "blocks")},
        )
    cameras = {}
    with h5py.File(trial / "camera-frames.hdf5") as stream:
        for name in stream:
            if name == "time":
                continue
            filename = f"first-{name}.png"
            Image.open(io.BytesIO(bytes(cast(h5py.Dataset, stream[name])[0]))).convert("RGB").save(
                output / filename
            )
            cameras[name] = filename
    if len(cameras) != 4:
        raise ValueError("Expected four candidate camera views")
    return ReplayEvidence(True, status["task_success"], status["policy_action_failure"], cameras)
