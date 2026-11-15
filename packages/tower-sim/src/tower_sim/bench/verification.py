"""Release-bound executable closure, installed dependencies and benchmark data."""

import base64
import csv
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import platform
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .data_release import verify_locator, verify_payloads
from .resources import sha
from .scene_format import DEFAULT_BENCHMARK, revision
from .source_receipt import module_source

CONTRACT = "tower-public-benchmark-v2"
DATA_VERSION = "benchmark-2026.11.1"
PACKAGES = {"tower-sim": "0.1.0", "tower-eval": "0.1.0", "nero-kinematics": "0.2.0"}
PACKAGE_MODULES = {
    "tower-sim": ("tower_sim",),
    "tower-eval": ("tower_eval",),
    "nero-kinematics": ("nero_kinematics", "nero_eef_ik"),
}
DEPENDENCIES = ("numpy", "scipy", "h5py", "Pillow", "PyYAML", "isaacsim")
IK_PINS = {"pin": "3.8.0", "casadi": "3.7.2", "pin-pink": "3.4.0", "numpy": "1.26.4"}
CONDA_IK_PACKAGES = {
    "pin": ("pinocchio", "pinocchio-python", "libpinocchio"),
    "casadi": ("casadi",),
    "numpy": ("numpy",),
    "scipy": ("scipy",),
}
CHECKOUT_TREES = ("tools/simulator", "scripts")
BOUND_TREES = (*CHECKOUT_TREES, "packages", "src")
ENTRYPOINTS = (
    "scripts/maintainer/gpu_baseline.py",
    "scripts/maintainer/candidate_policy.py",
    "tools/simulator/benchmark_candidate.py",
    "tools/simulator/benchmark_portable.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path, expected=None):
    """Hash and parse the same bytes, including request-bound manifests."""
    payload = Path(path).read_bytes()
    if expected is not None and digest(payload) != expected:
        raise ValueError(f"Changed request binding: {path}")
    return json.loads(payload)


def envelope(release=None, *, benchmark=DEFAULT_BENCHMARK):
    revision(benchmark)
    release = release or {}
    return dict(
        schema="tower-public-envelope-v1",
        benchmark=benchmark,
        data_version="benchmark-" + benchmark,
        request_schema=CONTRACT,
        packages=dict(PACKAGES),
        release_manifest_sha256=release.get("manifest_sha256"),
        commit=release.get("commit"),
        data_manifest_sha256=digest(data_manifest_path(benchmark).read_bytes()),
    )


def package_identity():
    result = {}
    for name, expected in PACKAGES.items():
        actual = metadata.version(name)
        if actual != expected:
            raise ValueError(f"Unsupported package version: {name} {actual}; expected {expected}")
        module = importlib.import_module(name.replace("-", "_"))
        if module.__file__ is None:
            raise ValueError(f"Installed package has no file identity: {name}")
        result[name] = dict(version=actual, path=str(Path(module.__file__).resolve()))
    return result


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes its bound root: {relative}")
    return path


def data_manifest_path(benchmark=DEFAULT_BENCHMARK):
    revision(benchmark)
    if benchmark != DEFAULT_BENCHMARK:
        raise ValueError("Only the public benchmark revision is distributed")
    return Path(__file__).with_name("public-data.json")


def verify_data(
    checkout, *, benchmark=DEFAULT_BENCHMARK, assets=None, data_root=None, v1_root=None
) -> dict[str, Any]:
    root = Path(checkout).resolve()
    payload = data_manifest_path(benchmark).read_bytes()
    manifest = json.loads(payload)
    if manifest["data_version"] != "benchmark-" + benchmark:
        raise ValueError("Unsupported benchmark data version")
    parsed = {}
    pins = manifest["files"]
    files = {}
    for relative, expected in pins.items():
        path = contained(root, relative)
        contents = path.read_bytes()
        if isinstance(expected, dict):
            verify_locator(json.loads(contents), document=manifest)
            continue
        if digest(contents) != expected:
            raise ValueError(f"Changed benchmark data: {relative}")
        files[str(path)] = expected
        if relative in (manifest["protocol"], manifest["scene_index"]):
            parsed[relative] = json.loads(contents)
    protocol = parsed[manifest["protocol"]]
    index = parsed[manifest["scene_index"]]["scenes"]
    selector = "seed" if benchmark == "2026.11.0" else "scene_id"
    if len(index) != 600 or len({(r["task"], r[selector]) for r in index}) != 600:
        raise ValueError("Expected exactly 600 unique released scenes")
    if len(protocol["tasks"]) != 12:
        raise ValueError("Expected exactly 12 benchmark tasks")
    for task, settings in protocol["tasks"].items():
        if sum(row["task"] == task for row in index) != settings["scene_count"]:
            raise ValueError(f"Scene count differs for {task}")
    payload_files = {}
    if benchmark == DEFAULT_BENCHMARK and assets is not None:
        payload_files = verify_payloads(assets=assets, data_root=data_root)
    return dict(
        **envelope(benchmark=benchmark),
        verified_payload_files=len(payload_files),
        verified_data_files=len(pins),
        scenes=600,
        tasks=12,
        files={**files, **payload_files},
    )


def verify_package_data():
    """Bind the unchanged installed calibration, profiles and IK resources."""
    root = Path(__file__).resolve().parents[1]
    pins = read(root / "data/pins.json")
    files = {}
    for name, expected in pins.items():
        path = contained(root, name)
        if sha(path) != expected:
            raise ValueError(f"Changed evaluation package data: {name}")
        files[str(path)] = expected
    return files


def conda_ik_identity(dependency):
    """Verify archived native Conda packages when no pip RECORD exists."""
    root = Path(sys.base_prefix).resolve()
    module = importlib.import_module("pinocchio" if dependency == "pin" else dependency)
    expected_version = IK_PINS.get(dependency, module.__version__)
    if module.__version__ != expected_version:
        raise ValueError(f"Unsupported IK dependency: {dependency}")
    manifests, files = {}, {}
    for name in CONDA_IK_PACKAGES[dependency]:
        candidates = []
        for path in (root / "conda-meta").glob(name + "-*.json"):
            payload = path.read_bytes()
            body = json.loads(payload)
            if body.get("name") == name:
                candidates.append((path, payload, body))
        if len(candidates) != 1:
            raise ValueError(f"Expected one Conda IK manifest: {name}")
        path, payload, body = candidates[0]
        if body["version"] != expected_version:
            raise ValueError(f"Unsupported IK dependency: {name}")
        manifests[str(path)] = digest(payload)
        rows = body["paths_data"]["paths"]
        if {row["_path"] for row in rows} != set(body["files"]):
            raise ValueError(f"Conda IK file set differs: {name}")
        for row in rows:
            loaded = contained(root, row["_path"])
            if loaded.suffix == ".pyc":
                continue
            expected = row.get("sha256_in_prefix") or row.get("sha256")
            actual = sha(loaded)
            if expected is None or actual != expected:
                raise ValueError(f"Installed dependency differs from Conda manifest: {loaded}")
            files[str(loaded)] = actual
    if module.__file__ is None:
        raise ValueError(f"Imported {dependency} has no file identity")
    if not files or str(Path(module.__file__).resolve()) not in files:
        raise ValueError(f"Imported {dependency} is outside its verified Conda manifests")
    return dict(version=module.__version__, conda_manifests=manifests, files=files)


def verify_conda_record_entry(identity, path, payload, record, contents):
    """Conda can rewrite wheel scripts/INSTALLER while retaining the original RECORD.

    Both that exact RECORD and the transformed file must be owned and hash-verified
    by the installed Conda manifest; an unexplained RECORD mismatch stays fatal.
    """
    if identity["files"].get(str(record.resolve())) != digest(contents) or identity["files"].get(
        str(path.resolve())
    ) != digest(payload):
        raise ValueError(f"RECORD or installed bytes differ from Conda manifest: {path}")


def dependency_identity(names=DEPENDENCIES) -> dict[str, Any]:
    """Verify RECORD hashes against installed bytes; also bind unhashed .py files."""
    packages = {}
    for name in names:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            if name not in CONDA_IK_PACKAGES or names == DEPENDENCIES:
                raise
            packages[name] = conda_ik_identity(name)
            continue
        if name in IK_PINS and names != DEPENDENCIES and distribution.version != IK_PINS[name]:
            raise ValueError(f"Unsupported IK dependency: {name} {distribution.version}")
        records = [f for f in distribution.files or () if str(f).endswith(".dist-info/RECORD")]
        if not records and name in CONDA_IK_PACKAGES and names != DEPENDENCIES:
            packages[name] = conda_ik_identity(name)
            continue
        if len(records) != 1:
            raise ValueError(f"Expected installed dependency RECORD: {name}")
        record = Path(str(distribution.locate_file(records[0])))
        contents = record.read_bytes()
        files = {}
        conda = None
        for relative, encoded, size in csv.reader(io.StringIO(contents.decode())):
            path = Path(str(distribution.locate_file(relative)))
            if (
                path.suffix == ".pyc"
                and "__pycache__" in path.parts
                and Path(importlib.util.source_from_cache(str(path))).is_file()
            ):
                continue
            payload = path.read_bytes()
            if encoded:
                algorithm, expected = encoded.split("=", 1)
                actual = (
                    base64.urlsafe_b64encode(hashlib.new(algorithm, payload).digest())
                    .rstrip(b"=")
                    .decode()
                )
                if actual != expected or (size and len(payload) != int(size)):
                    if name not in CONDA_IK_PACKAGES or names == DEPENDENCIES:
                        raise ValueError(
                            f"Installed dependency differs from RECORD: {name}: {relative}"
                        )
                    conda = conda or conda_ik_identity(name)
                    verify_conda_record_entry(conda, path, payload, record, contents)
            files[relative] = digest(payload)
        if not files:
            raise ValueError(f"No installed dependency bytes: {name}")
        packages[name] = dict(
            version=distribution.version, record_sha256=digest(contents), files=files
        )
        if conda is not None:
            packages[name]["conda"] = conda
    return dict(
        schema="tower-dependencies-v2",
        python=platform.python_version(),
        executable=str(Path(sys.executable).absolute()),
        packages=packages,
    )


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def checkout_files(root):
    """Exact tracked set, including non-Python runtime assets and subprocess entries."""
    return git(root, "ls-files", "-z", "--", *CHECKOUT_TREES).decode().strip("\0").split("\0")


def verify_checkout(root, body):
    if git(root, "rev-parse", "HEAD").decode().strip() != body["commit"]:
        raise ValueError("Checkout HEAD differs from release commit")
    verify_checkout_layout(root)
    from .public_surface import check_public

    for name in git(root, "ls-files", "-z").decode().split("\0"):
        path = root / name
        if name and path.is_file():
            check_public(name, path.read_bytes())
    files = body["checkout_files"]
    if body.get("checkout_trees") != list(CHECKOUT_TREES) or set(files) != set(
        checkout_files(root)
    ):
        raise ValueError("Release checkout file set differs from tracked runtime trees")
    for relative, expected in files.items():
        path = contained(root, relative)
        if sha(path) != expected:
            raise ValueError(f"Changed release checkout file: {relative}")
    return {str(contained(root, name)): value for name, value in files.items()}


def verify_checkout_layout(root):
    """Keep tracked bytes clean; permit private environments outside bound trees."""
    if git(root, "status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("Release requires a clean checkout of all tracked files")
    for ignored in ((), ("--ignored",)):
        extra = git(
            root, "ls-files", "--others", *ignored, "--exclude-standard", "-z", "--", *BOUND_TREES
        )
        if extra:
            raise ValueError(
                f"Untracked runtime file in bound tree: {extra.decode()!r}; remove it. "
                "For bytecode caches, set PYTHONDONTWRITEBYTECODE=1 (or PYTHONPYCACHEPREFIX "
                "outside the checkout) and delete existing __pycache__ directories"
            )


def wheel_payload(payload, name, version):
    from .public_surface import check_lineage, check_public

    package = name.replace("-", "_")
    info = f"{package}-{version}.dist-info/"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.namelist():
            check_lineage(member, archive.read(member))
            check_public(member, archive.read(member))
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate release wheel member")
        result = {}
        for member in names:
            parts = PurePosixPath(member).parts
            if member.startswith("/") or ".." in parts or "\\" in member:
                raise ValueError(f"Unexpected release wheel member: {member}")
            if member.endswith("/"):
                continue
            if parts[0] in PACKAGE_MODULES[name] and not member.endswith((".pyc", ".pth")):
                result[member] = archive.read(member)
            elif not (
                member
                in {
                    info + p
                    for p in ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt")
                }
                or member.startswith(info + "licenses/")
            ):
                raise ValueError(f"Unexpected release wheel member: {member}")
        if not result:
            raise ValueError(f"Release wheel has no package payload: {name}")
        return result


def verify_release(path, *, checkout=None, expected=None):
    path = Path(path).resolve()
    payload = path.read_bytes()
    if expected is not None and digest(payload) != expected:
        raise ValueError("Changed request binding: release_manifest")
    body = json.loads(payload)
    if body.get("schema") != "tower-release-v2" or body.get("public") != envelope():
        raise ValueError("Unsupported public release manifest")
    if set(body["packages"]) != set(PACKAGES):
        raise ValueError("Release must include all three workspace packages")
    root = Path(checkout).resolve() if checkout is not None else path.parent.parent
    checkout_bindings = verify_checkout(root, body)
    installed = package_identity()
    wheel_files = {}
    for name, entry in body["packages"].items():
        wheel = contained(path.parent, entry["wheel"])
        contents = wheel.read_bytes()
        if digest(contents) != entry["sha256"]:
            raise ValueError(f"Changed release wheel: {name}")
        if entry["version"] != installed[name]["version"]:
            raise ValueError(f"Release package version differs: {name}")
        location = Path(installed[name]["path"]).parent
        members = wheel_payload(contents, name, entry["version"])
        loaded_files = {
            module + "/" + p.relative_to(location.parent / module).as_posix()
            for module in PACKAGE_MODULES[name]
            for p in (location.parent / module).rglob("*")
            if p.is_file() and not (p.suffix == ".pyc" and "__pycache__" in p.parts)
        }
        if loaded_files != set(members):
            raise ValueError(f"Installed package file set differs from release wheel: {name}")
        for member, original in members.items():
            module, relative = member.split("/", 1)
            loaded = contained(location.parent / module, relative)
            if loaded.read_bytes() != original:
                raise ValueError(f"Installed package differs from release wheel: {member}")
            wheel_files[str(loaded)] = digest(original)
    return dict(
        body,
        manifest_sha256=digest(payload),
        installed_packages=installed,
        wheel_files=wheel_files,
        checkout_bindings=checkout_bindings,
        checkout=str(root),
    )


def portable_root(argv):
    """Read Kit's post-parser option without abbreviations or missing-value indexing."""
    found = None
    for index, argument in enumerate(argv):
        if argument == "--portable-root":
            value = argv[index + 1] if index + 1 < len(argv) else ""
        elif argument.startswith("--portable-root="):
            value = argument.split("=", 1)[1]
        else:
            continue
        if not value or value.startswith("--"):
            raise ValueError("--portable-root requires a path")
        if found is not None:
            raise ValueError("Duplicate --portable-root")
        found = value
    return found


def sdk_roots(*, generated=None):
    """Interpreter/SDK locations, never inferred from arbitrary imported modules."""
    roots = {
        str(Path(sysconfig.get_path(name)).resolve())
        for name in ("stdlib", "platstdlib", "purelib", "platlib")
    }
    # Debian/Ubuntu ship this stdlib entry as a symlink into /etc/pythonX.Y.
    # Admit its exact target, never its parent or a module-selected location.
    customization = Path(sysconfig.get_path("stdlib")) / "sitecustomize.py"
    if customization.is_symlink() and customization.is_file():
        roots.add(str(customization.resolve()))
    if generated is None:
        generated = portable_root(sys.argv[1:])
    if generated is not None:
        roots.add(str(Path(generated).resolve() / "cache/ogn_generated"))
    return sorted(roots)


def validate_sdk_roots(checkout, allowed_roots):
    """Interpreter locations cannot authorize any checkout runtime/source tree."""
    trees = [(Path(checkout) / tree).resolve() for tree in BOUND_TREES]
    roots = [Path(root).resolve() for root in allowed_roots]
    for root in roots:
        if any(root.is_relative_to(tree) or tree.is_relative_to(root) for tree in trees):
            raise ValueError(f"SDK root overlaps a bound tree: {root}")
    return roots


def imported_sources():
    """Snapshot the file closure already loaded, before creating any GPU context."""
    files = {}
    for name, module in list(sys.modules.items()):
        source = module_source(name, module)
        if source is not None:
            path = str(source)
            if path not in files:
                files[path] = sha(path)
    return files


def verify_startup(release, *, generated=None):
    """Fail before GPU work on an unsafe layout or an already imported unbound file."""
    verify_checkout_layout(Path(release["checkout"]))
    roots = sdk_roots(generated=generated)
    verify_sources(imported_sources(), release, allowed_roots=roots)


def verify_sources(sources, release, *, allowed_roots=(), resources=None):
    if not sources:
        raise ValueError("Empty public source closure")
    bound = dict(release["checkout_bindings"], **release["wheel_files"])
    bound.update(resources or {})
    root = Path(release["checkout"]).resolve()
    roots = validate_sdk_roots(root, allowed_roots)
    packages = [
        Path(row["path"]).parent.parent / module
        for name, row in release["installed_packages"].items()
        for module in PACKAGE_MODULES[name]
    ]
    guarded = [*(root / tree for tree in BOUND_TREES), *packages]
    for filename, expected in sources.items():
        path = Path(filename).resolve()
        if sha(path) != expected:
            raise ValueError(f"Changed imported file: {filename}")
        if str(path) in bound:
            if bound[str(path)] != expected:
                raise ValueError(f"Imported file differs from release: {filename}")
        elif any(path.is_relative_to(p.resolve()) for p in guarded):
            raise ValueError(f"Unbound public import in bound tree: {filename}")
        elif any(path.is_relative_to(p) for p in roots):
            continue
        else:
            raise ValueError(f"Unbound public import: {filename}")
    return sources


class ReleaseVerification:
    """Verified public authorization. A duck-typed verify method is insufficient."""

    def __init__(self, *, checkout, request, verify_assets):
        self.checkout = Path(checkout)
        self.request = request
        self.verify_assets = verify_assets
        self.identity = self.verify()

    @property
    def provenance(self):
        public = self.identity["public"]
        return dict(
            public=public,
            **{k: public[k] for k in ("release_manifest_sha256", "commit", "data_manifest_sha256")},
        )

    def verify(self):
        request = self.request
        release = verify_release(
            request["release_manifest"],
            checkout=self.checkout,
            expected=request["release_manifest_sha256"],
        )
        verify_startup(release)
        benchmark = request.get("public", {}).get("benchmark", DEFAULT_BENCHMARK)
        if request.get("public") != envelope(release, benchmark=benchmark):
            raise ValueError("Request envelope differs from verified release")
        expected = read(request["dependency_manifest"], request["dependency_manifest_sha256"])
        actual = dependency_identity()
        if actual != expected:
            raise ValueError("Installed dependencies differ from the dependency manifest")
        expected_ik = read(
            request["ik_dependency_manifest"], request["ik_dependency_manifest_sha256"]
        )
        probe = subprocess.run(
            [
                request["ik_python"],
                "-m",
                "tower_sim.bench.verification",
                "--ik",
                "--release",
                request["release_manifest"],
                "--release-sha256",
                request["release_manifest_sha256"],
                "--checkout",
                str(self.checkout),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        ik = json.loads(probe.stdout)
        if ik != expected_ik:
            raise ValueError("IK dependencies differ from the dependency manifest")
        data = verify_data(
            self.checkout,
            benchmark=benchmark,
            assets=request["assets"],
            data_root=request.get("data_root"),
            v1_root=request.get("v1_root"),
        )
        data["files"].update(verify_package_data())
        assets = self.verify_assets(request["assets"])
        return dict(
            public=envelope(release, benchmark=benchmark),
            release=release,
            dependencies=actual,
            ik_dependencies=ik,
            data=data,
            assets=assets,
        )

    def verify_sources(self, sources):
        from .resources import Resources

        resources = dict(self.identity["data"]["files"])
        assets = Path(self.request["assets"]).resolve()
        paths = Resources.from_roots(checkout=self.checkout, assets=assets)
        lock = paths.path("asset_lock")
        asset_hash = read(lock, resources[str(lock.resolve())])["manifest_sha256"]
        manifest = read(assets / "manifest.json", asset_hash)
        resources[str(assets / "manifest.json")] = asset_hash
        resources.update(
            {str(contained(assets, row["path"])): row["sha256"] for row in manifest["files"]}
        )
        for name in ("nero.urdf", "nero.yaml", "model.json"):
            source = paths.path("kinematics") / name
            target = Path(self.request["output"]) / "simulation/kinematics" / name
            resources[str(target.resolve())] = resources[str(source.resolve())]
        return verify_sources(
            sources,
            self.identity["release"],
            allowed_roots=sdk_roots(
                generated=Path(self.request["output"]).with_name(
                    Path(self.request["output"]).name + "-kit"
                )
            ),
            resources=resources,
        )


def ik_python_sources():
    """Bind native-stack Python code, including files omitted from RECORD."""
    result = {}
    for name in ("pinocchio", "casadi", "pink", "numpy", "scipy"):
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError(f"Missing IK module location: {name}")
        root = Path(module.__file__).resolve().parent
        result[name] = {
            str(path.relative_to(root)): sha(path) for path in sorted(root.rglob("*.py"))
        }
        if not result[name]:
            raise ValueError(f"Missing IK Python source identity: {name}")
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--ik", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--release-sha256")
    parser.add_argument("--checkout", type=Path)
    args = parser.parse_args(argv)
    if args.ik:
        if args.release is None or args.release_sha256 is None or args.checkout is None:
            parser.error("IK verification requires a bound release and checkout")
        verify_release(args.release, checkout=args.checkout, expected=args.release_sha256)
    names = tuple(IK_PINS) + ("scipy",) if args.ik else DEPENDENCIES
    identity = dependency_identity(names)
    if args.ik:
        identity["python_sources"] = ik_python_sources()
    print(json.dumps(identity, indent=2))  # noqa: T201 -- metadata probe


if __name__ == "__main__":
    main()
