"""Imported-source provenance and first-import timing, without module patching."""

import re
import sys
from importlib.machinery import EXTENSION_SUFFIXES, BuiltinImporter, FrozenImporter, NamespaceLoader
from pathlib import Path
from zipimport import zipimporter

from .resources import sha


def module_source(name, module):
    """Bind a loaded module or refuse an importer with no verifiable source.

    Builtin/frozen modules and code-free namespace packages have no payload.
    SDK zip archives and meta-path modules bind their archive/loader bytes;
    an arbitrary PYTHONPATH importer is never an admission root.
    The __file__-first rule trusts module metadata only because a custom loader
    must first be installed by admitted code; binding still requires verification.
    Runtime-created objects with neither loader nor spec likewise can only be
    injected by already-running, bound code. Without an existing file they carry
    no separate payload; importer-created entries still require a bound source.
    """
    if module is None:
        # The import-blocking sentinel carries no executable payload.
        return None
    source = getattr(module, "__file__", None)
    if source and Path(source).is_file():
        return Path(source).resolve()
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None) or getattr(module, "__loader__", None)
    if not source and (
        loader is BuiltinImporter or loader is FrozenImporter or type(loader) is NamespaceLoader
    ):
        return None
    if (
        not source
        and loader is None
        and spec is None
        and (
            name in {"typing.io", "typing.re", "cython_runtime"}
            or re.fullmatch(r"_cython_\d+_\d+_\d+", name)
        )
    ):
        # Cython creates these inert native-runtime bookkeeping modules while
        # loading SDK numerical extensions; they have no importer or payload.
        return None
    from .verification import sdk_roots

    roots = [Path(root) for root in sdk_roots()]
    if not source and loader is None and spec is None:
        # Pybind/Cython extension submodules may have no separate file or loader.
        # Bind their actual native parent; never infer an SDK root from it.
        qualified_names = {name}
        actual_name = getattr(module, "__name__", name)
        stdlib_aliases = {
            "xml.parsers.expat.errors": "pyexpat.errors",
            "xml.parsers.expat.model": "pyexpat.model",
        }
        if (
            actual_name.split(".")[0] == name.split(".")[0]
            or stdlib_aliases.get(name) == actual_name
        ):
            qualified_names.add(actual_name)
        for qualified in qualified_names:
            parent = qualified.rpartition(".")[0]
            while parent:
                parent_module = sys.modules.get(parent)
                parent_spec = getattr(parent_module, "__spec__", None)
                if qualified in stdlib_aliases.values() and getattr(
                    parent_spec, "loader", None
                ) in (BuiltinImporter, FrozenImporter):
                    return None
                native = getattr(parent_module, "__file__", None)
                if native and str(native).endswith(tuple(EXTENSION_SUFFIXES)):
                    path = Path(native).resolve()
                    if path.is_file() and any(path.is_relative_to(root) for root in roots):
                        return path
                parent = parent.rpartition(".")[0]
    if loader is None and spec is None:
        # Existing files were bound above; preserve native-parent binding first.
        # Relative/nonexistent __file__ values on runtime objects are not payloads.
        return None
    if type(loader) is zipimporter:
        archive = Path(loader.archive).resolve()
        if archive.is_file() and any(archive.is_relative_to(root) for root in roots):
            return archive
    provider = sys.modules.get(type(loader).__module__)
    provider_file = getattr(provider, "__file__", None)
    if type(loader).__module__.split(".")[0] == name.split(".")[0] and provider_file:
        path = Path(provider_file).resolve()
        if path.is_file() and any(path.is_relative_to(root) for root in roots):
            return path
    raise ValueError(f"Unbound non-file import: {name}: {source!r} ({type(loader).__name__})")


class ImportReceipt:
    """An inert-after-close audit observer; audit hooks cannot be unregistered."""

    def __init__(self):
        self.phase = "before_simulation_app"
        self.active = True
        self.events = []
        self.preexisting = {
            name: self.module(name) for name in ("yaml", "yaml._yaml") if name in sys.modules
        }
        sys.addaudithook(self.observe)

    def observe(self, event, args):
        if self.active and event == "import" and args[0] in ("yaml", "yaml._yaml"):
            self.events.append(dict(module=args[0], phase=self.phase, requested_file=args[1]))

    @staticmethod
    def module(name):
        module = sys.modules.get(name)
        source = getattr(module, "__file__", None)
        if source is None:
            return None
        path = Path(source).resolve()
        return dict(path=str(path), sha256=sha(path), version=getattr(module, "__version__", None))

    def app_factory(self, factory):
        def launch(*args, **kwargs):
            self.phase = "during_simulation_app"
            app = factory(*args, **kwargs)
            self.phase = "after_simulation_app"
            return app

        return launch

    def collect(self):
        modules = {}
        files = {}
        for name, module in list(sys.modules.items()):
            source = module_source(name, module)
            if source is not None:
                path = str(source)
                if path not in files:
                    files[path] = sha(path)
                modules[name] = dict(path=path, sha256=files[path])
                if hasattr(module, "__tower_entry_sha256__"):
                    modules[name]["startup_sha256"] = module.__tower_entry_sha256__
        return dict(
            schema="tower-bench-imported-sources-v1",
            files=files,
            modules=modules,
            yaml=dict(
                preexisting=self.preexisting,
                first_import_events=self.events,
                loaded={name: self.module(name) for name in ("yaml", "yaml._yaml")},
            ),
        )

    def close(self):
        self.active = False


def validate_imports(
    receipt, *, forbidden_roots=(), forbidden_stems=(), allowed_packages=None, allowed_modules=None
):
    """Reject forbidden live code, not merely an eager-import inventory."""
    generation = {
        "jenga_generator.motion",
        "jenga_generator.curobo_planner",
        "jenga_generator.worker",
        "jenga_generator.benchmark",
    }
    package_roots = {
        name: tuple(Path(root).resolve() for root in roots)
        for name, roots in (allowed_packages or {}).items()
    }
    forbidden = tuple(Path(root).resolve() for root in forbidden_roots)
    for name, row in receipt["modules"].items():
        path = Path(row["path"]).resolve()
        # Canonical tower_sim ports may retain a frozen stem. Every other module
        # with a manifest stem is forbidden regardless of its checkout or storage root.
        frozen_stem = path.stem in forbidden_stems or name.rsplit(".", 1)[-1] in forbidden_stems
        known_package = any(
            name.startswith(package + ".") and any(path.is_relative_to(root) for root in roots)
            for package, roots in package_roots.items()
        )
        # A maintainer entry can share a frozen filename while running as
        # __main__. Bind that exception to its exact name, path AND committed
        # bytes; never allow a bare stem or a copied entry in another root.
        expected = next(
            (
                binding
                for binding in (allowed_modules or {}).get(name, ())
                if path == Path(binding["path"]).resolve()
            ),
            None,
        )
        known_entry = bool(
            expected
            and row.get("sha256") == expected["sha256"] == sha(path)
            and row.get("startup_sha256") == expected["sha256"]
        )
        if (
            name in generation
            or name.split(".")[0]
            in {"jenga_generator", "replay_runner_" + "isaac_sim", "task2_fixed"}
            or name
            in {"tower_eval.simulator.isaac", "tower_sim.backend.model", "tower_sim.generation"}
            or name.startswith("tower_sim.generation.")
            or (frozen_stem and not (known_package or known_entry))
            or any(path.is_relative_to(root) for root in forbidden)
        ):
            raise ValueError(f"Forbidden candidate import: {name}: {path}")
