"""Check the pinned SDK and native material libraries without starting Kit."""

import ctypes
import importlib.metadata
import json
import sys
from pathlib import Path

PLUGINS = ("libMaterialXRenderHw.so", "libMaterialXRenderGlsl.so", "libusd_usdBakeMtlx.so")


def load_plugins(paths):
    """Fail on missing libraries or unresolved native dependencies."""
    handles = []
    for name in PLUGINS:
        candidates = paths.get(name, [])
        if not candidates:
            raise RuntimeError(
                f"Isaac MaterialX/USD plugin missing: {name}; reinstall the pinned SDK"
            )
        for path in candidates:
            try:
                handles.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot load Isaac MaterialX/USD plugin {name}: {exc}. "
                    "If libXt.so.6 is missing, install libxt6 (Debian/Ubuntu) or "
                    "conda-forge xorg-libxt and expose only required X11 library symlinks in "
                    "LD_LIBRARY_PATH. See docs/reference/bench-public.md."
                ) from exc
    return list(PLUGINS)


def probe():
    try:
        version = importlib.metadata.version("isaacsim")
        distribution = importlib.metadata.distribution("isaacsim-extscache-kit")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Missing Isaac SDK distribution: {exc.name}; install "
            '"isaacsim[all,extscache]==6.1.0" with Python 3.12, '
            "--default-index https://pypi.org/simple --index https://pypi.nvidia.com "
            "--index-strategy unsafe-best-match. See docs/reference/bench-public.md."
        ) from exc
    identity = dict(python=sys.version.split()[0], isaacsim=version)
    if identity["isaacsim"] != "6.1.0.0" or sys.version_info[:2] != (3, 12):
        raise ValueError(f"Requires Isaac 6.1.0.0 / Python 3.12; found {identity}")
    paths = {name: [] for name in PLUGINS}
    for entry in distribution.files or ():
        if entry.name in paths:
            paths[entry.name].append(Path(distribution.locate_file(entry)))
    identity["material_plugins"] = load_plugins(paths)
    return identity


if __name__ == "__main__":
    print(json.dumps(probe()))
