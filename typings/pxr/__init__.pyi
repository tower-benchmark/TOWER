"""USD symbols are provided dynamically by compiled plugins, including Isaac's extensions."""

from typing import Any

def __getattr__(name: str) -> Any: ...
