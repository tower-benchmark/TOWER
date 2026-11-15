"""Ordered SimulationApp factories and scoped process startup state.

No SDK imports or environment changes occur at import time. EULA consent is
supplied by the launcher; this factory never invents consent.
"""

import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .experience import local_stock_experience, omit_viewport_overlays
from .resources import save, sha

RENDER_COMPLETION_ARGS = (
    "--/app/updateOrder/checkForHydraRenderComplete=1000",
    "--/app/renderer/waitIdle=true",
    "--/app/hydraEngine/waitIdle=true",
)


def require_python312():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Isaac 6.1 construction requires Python 3.12")


def simulation_app(config):
    require_python312()
    import isaacsim  # pyright: ignore[reportMissingImports]

    # Deliver requested completion settings through the SDK argv boundary too.
    original = list(sys.argv)
    try:
        sys.argv.extend(
            flag
            for flag in RENDER_COMPLETION_ARGS
            if flag in config.get("extra_args", []) and flag not in sys.argv
        )
        return isaacsim.SimulationApp(config)
    finally:
        sys.argv[:] = original


def native_renderer(resources, base):
    def configure(stage, *args, **kwargs):
        base(stage, *args, **kwargs)
        import carb  # pyright: ignore[reportMissingImports]
        from pxr import Usd  # pyright: ignore[reportMissingImports]

        saved = Usd.Stage.Open(str(resources.path("appearance_scene")))
        path = "/__DualReplayCalibrationRender/DefaultPreviewDome"
        source, target = saved.GetPrimAtPath(path), stage.GetPrimAtPath(path)
        if not source or not target:
            raise ValueError("Missing V2.2 calibration dome")
        for attr in source.GetAttributes():
            if attr.GetName().startswith("inputs:") and attr.HasAuthoredValueOpinion():
                target.CreateAttribute(
                    attr.GetName(), attr.GetTypeName(), custom=attr.IsCustom()
                ).Set(attr.Get())
        for key, value in resources.read("renderer_controls").items():
            carb.settings.get_settings().set(key, value)

    return configure


class AppSession:
    """Own app/model lifetimes without replacing app.close or any module global."""

    def __init__(self, app, renderer):
        self.app, self.renderer = app, renderer
        self.models = []
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.app, name)

    def close(self, *args, **kwargs):
        if not self.closed:
            self.closed = True
            try:
                for model in self.models:
                    close = getattr(model, "close", None)
                    if close is not None:
                        close()
            finally:
                self.app.close(*args, **kwargs)


@contextmanager
def native_render_initialization(factory, renderer_factory):
    def launch(*args, **kwargs):
        app = factory(*args, **kwargs)
        try:
            # The renderer dependency is captured only after the native app exists.
            return AppSession(app, renderer_factory())
        except BaseException:
            app.close()
            raise

    yield launch


@contextmanager
def grouped_runtime(
    factory,
    group,
    output,
    *,
    sdk,
    startup="headless",
    zero_delay=False,
    experience=None,
    model_setup=lambda session: None,
    clock=time.perf_counter,
):
    if not group or min(group) < 0 or len(set(group)) != len(group):
        raise ValueError("GPU groups require distinct nonnegative device indices")
    if startup not in ("stock", "headless"):
        raise ValueError("startup must be stock or headless")
    experience = experience or omit_viewport_overlays(local_stock_experience)

    def launch(config):
        require_python312()
        configured = {
            **config,
            # skipOldDumpUpload covers previous crashes only. Disable the SDK
            # reporter plugin itself so a new failure cannot upload a dump.
            "enable_crashreporter": False,
            "multi_gpu": len(group) > 1,
            "max_gpu_count": len(group),
            "active_gpu": group[0],
            "disable_viewport_updates": bool(config.get("headless", True)),
            "physics_gpu": group[0],
            "extra_args": [
                *config.get("extra_args", []),
                "--/renderer/multiGpu/activeGpus=" + ",".join(map(str, group)),
            ],
        }
        if zero_delay:
            configured["extra_args"].extend(RENDER_COMPLETION_ARGS)
        if startup == "headless" and configured.get("headless", True):
            if not configured.get("experience"):
                import isaacsim  # pyright: ignore[reportMissingImports]

                if isaacsim.__file__ is None:
                    raise RuntimeError("Isaac SDK package location is unavailable")
                installed_sdk = Path(isaacsim.__file__).parent
                if Path(sdk).resolve() != installed_sdk.resolve():
                    raise ValueError("SDK root differs from the imported isaacsim package")
                configured["experience"] = str(experience(installed_sdk, output))
            configured["extra_args"].append("--/crashreporter/skipOldDumpUpload=1")
        save(Path(output) / "launch.json", configured)
        start = clock()
        session = factory(configured)
        try:
            model_setup(session)
            save(
                Path(output) / "startup.json",
                dict(
                    SimulationApp_seconds=clock() - start,
                    mode=startup,
                    experience=configured.get("experience", "stock"),
                ),
            )
        except BaseException:
            session.close()
            raise
        return session

    yield launch


def gpu_readback(group):
    import carb  # pyright: ignore[reportMissingImports]

    controls = carb.settings.get_settings()
    keys = (
        "/renderer/multiGpu/enabled",
        "/renderer/multiGpu/activeGpus",
        "/renderer/multiGpu/maxGpuCount",
        "/renderer/activeGpu",
        "/physics/cudaDevice",
    )
    actual = {key: controls.get(key) for key in keys}
    if (
        actual[keys[0]] != (len(group) > 1)
        or str(actual[keys[1]]) != ",".join(map(str, group))
        or actual[keys[2]] != len(group)
        or actual[keys[3]] != group[0]
        or actual[keys[4]] != group[0]
    ):
        raise RuntimeError(
            "Isaac did not activate the requested renderer GPU group: " + str(actual)
        )
    return actual


def digest_tree(root):
    return (
        "\n".join(
            f"{sha(p)}  ./{p.relative_to(root)}"
            for p in sorted(p for p in Path(root).rglob("*") if p.is_file())
        )
        + "\n"
    )


@contextmanager
def private_shader_cache(
    *, seed, seed_sha, private_root, name, argv=None, clock=time.time, warmup=False
):
    """Append Kit flags to real argv before launch; remove owned files/flags on exit."""
    argv = sys.argv if argv is None else argv
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError("Private cache name must be one path component")
    directory = Path(private_root) / name
    target = directory / "nv_shadercache"
    directory.mkdir(parents=True, exist_ok=False)
    started = clock()
    flags = []
    original = list(argv)
    try:
        if warmup:
            target.mkdir()
            expected = ""
        else:
            expected = Path(seed_sha).read_text()
            shutil.copytree(seed, target)
            if digest_tree(target) != expected:
                raise RuntimeError("Private shader cache copy differs from the verified seed")
        flags = [
            f"--/rtx/shaderDb/driverAppShaderCachePath={target}",
            f"--/rtx/shaderDb/driverShaderCachePath={target}",
            "--/renderer/shadercache/driverDiskCache/sizeInMB=12000",
        ]
        argv.extend(flags)
        import hashlib

        yield dict(
            seed=str(seed),
            seed_sha256_list=hashlib.sha256(expected.encode()).hexdigest(),
            files=expected.count("\n"),
            private_cache=str(target),
            copy_and_verify_s=clock() - started,
            driver_cache_size_mb=12000,
            kit_args=flags,
            pid=os.getpid(),
        )
    finally:
        argv[:] = original
        if not warmup:
            shutil.rmtree(directory, ignore_errors=True)


@contextmanager
def process_environment(values, *, unset=()):
    """Explicit process-boundary values, restored even on constructor failure."""
    previous = {key: os.environ.get(key) for key in {*values, *unset}}
    try:
        os.environ.update(values)
        for key in unset:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
