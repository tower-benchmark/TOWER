"""Stock Isaac 6.1 experience, with the three original dependency removals."""

import copy
import hashlib
import json
import re
import tomllib
from pathlib import Path


def _merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def _expand(value, apps):
    if isinstance(value, dict):
        return {key: _expand(item, apps) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, apps) for item in value]
    if isinstance(value, str):
        return value.replace("${app}", str(apps))
    return value


def local_stock_experience(sdk, output):
    apps = Path(sdk).resolve() / "apps"
    base = apps / "isaacsim.exp.base.kit"
    python_app = apps / "isaacsim.exp.base.python.kit"
    source = base.read_text()
    config = tomllib.loads(source)
    python_config = tomllib.loads(python_app.read_text())
    assert config["package"]["version"] == "6.1.0"
    needle = '"isaacsim.robot.policy.examples" = {}\n'
    assert source.count(needle) == 1
    derived = source.replace(needle, "")
    derived = re.sub(r'^name = "Isaac-Sim Base"$', 'name = "Isaac-Sim Python"', derived, flags=re.M)
    derived = re.sub(
        r"^settings.persistent = true.*$", "settings.persistent = false", derived, flags=re.M
    )
    derived = re.sub(
        r'^title = "Isaac Sim Base"$', 'title = "Isaac Sim Python"', derived, flags=re.M
    )
    derived = derived.replace("${app}", str(apps))
    actual = tomllib.loads(derived)
    expected = _expand(_merge(config["settings"], python_config["settings"]), apps)
    assert actual["settings"] == expected, "Stock base/Python settings changed during derivation"
    dependencies = dict(config["dependencies"])
    dependencies.pop("isaacsim.robot.policy.examples")
    assert actual["dependencies"] == dependencies
    path = Path(output) / "rss.reanchor_render.kit"
    path.write_text(derived)
    manifest = {
        "base_app": str(base),
        "python_app": str(python_app),
        "base_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "derived_app": str(path),
        "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
        "removed_dependencies": ["isaacsim.robot.policy.examples"],
        "all_stock_base_and_python_settings_equal": True,
        "all_other_dependencies_and_order_overrides_equal": True,
    }
    (Path(output) / "startup-experience.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def omit_viewport_overlays(stock_experience):
    def experience(sdk, output):
        path = Path(stock_experience(sdk, output))
        text = path.read_text()
        original = tomllib.loads(text)
        removed = ["omni.kit.viewport.legacy_gizmos", "isaacsim.gui.sensors.icon"]
        for name in removed:
            line = '"' + name + '" = {}\n'
            assert text.count(line) == 1, name
            text = text.replace(line, "")
        actual = tomllib.loads(text)
        assert actual["settings"] == original["settings"]
        assert actual["dependencies"] == {
            k: v for k, v in original["dependencies"].items() if k not in removed
        }
        path.write_text(text)
        manifest_path = Path(output) / "startup-experience.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["removed_dependencies"].extend(removed)
        manifest["derived_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        manifest["headless_overlay_optimization"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path

    return experience
