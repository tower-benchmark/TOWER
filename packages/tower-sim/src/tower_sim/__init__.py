"""TOWER simulator components; importing the package never initializes a simulator."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tower-sim")
except PackageNotFoundError:
    import tomllib
    from pathlib import Path

    try:
        __version__ = tomllib.loads(
            (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
        )["project"]["version"]
    except FileNotFoundError:
        __version__ = "0+unknown"

__all__ = ["SimConfig"]


def __getattr__(name):
    if name == "SimConfig":
        from .config import SimConfig

        return SimConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
