import json
from types import SimpleNamespace

import pytest

from tools.simulator import benchmark
from tools.simulator import isaac_environment_probe as probe

SUPPORTED_SDK = {"isaacsim": "6.1.0.0"}
UNSUPPORTED_SDK = {"isaacsim": "6.0.0.0"}


def test_loads_every_material_plugin(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(probe.ctypes, "CDLL", lambda path, mode: calls.append((path, mode)))
    paths = {name: [tmp_path / name] for name in probe.PLUGINS}
    assert probe.load_plugins(paths) == list(probe.PLUGINS)
    assert calls == [(str(tmp_path / name), probe.ctypes.RTLD_GLOBAL) for name in probe.PLUGINS]


@pytest.mark.parametrize("failed", probe.PLUGINS)
def test_missing_libxt_is_actionable(monkeypatch, tmp_path, failed):
    def load(path, mode):
        if path.endswith(failed):
            raise OSError("libXt.so.6: cannot open shared object file")

    monkeypatch.setattr(probe.ctypes, "CDLL", load)
    with pytest.raises(RuntimeError, match="libXt.so.6.*libxt6.*xorg-libxt.*LD_LIBRARY_PATH"):
        probe.load_plugins({name: [tmp_path / name] for name in probe.PLUGINS})


def test_missing_plugin_fails():
    with pytest.raises(RuntimeError, match="plugin missing"):
        probe.load_plugins({})


@pytest.fixture
def sdk_metadata(monkeypatch, tmp_path):
    from importlib.metadata import PackagePath

    entries = [PackagePath(f"exts/materials/{name}") for name in probe.PLUGINS]
    entries.append(PackagePath("exts/materials/unrelated.so"))
    located = []
    requested = []

    def locate(entry):
        located.append(entry)
        return tmp_path / entry

    def distribution(name):
        requested.append(name)
        assert name == "isaacsim-extscache-kit"
        return SimpleNamespace(files=entries, locate_file=locate)

    def version(name):
        assert name == "isaacsim"
        return SUPPORTED_SDK["isaacsim"]

    monkeypatch.setattr(probe.importlib.metadata, "version", version)
    monkeypatch.setattr(probe.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(probe.sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(probe.sys, "version", "3.12.3 (test)")
    return entries, located, requested


def test_probe_locates_only_material_plugins(monkeypatch, tmp_path, sdk_metadata):
    entries, located, requested = sdk_metadata
    loaded = []
    monkeypatch.setattr(probe.ctypes, "CDLL", lambda path, mode: loaded.append(path))
    assert probe.probe() == {
        "python": "3.12.3",
        "isaacsim": "6.1.0.0",
        "material_plugins": list(probe.PLUGINS),
    }
    assert requested == ["isaacsim-extscache-kit"]
    assert located == entries[:-1]
    assert loaded == [str(tmp_path / entry) for entry in entries[:-1]]


@pytest.mark.parametrize(
    "version,python",
    [(UNSUPPORTED_SDK["isaacsim"], (3, 12)), (SUPPORTED_SDK["isaacsim"], (3, 11))],
)
def test_probe_rejects_unsupported_versions(monkeypatch, sdk_metadata, version, python):
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: version)
    monkeypatch.setattr(probe.sys, "version_info", python)
    with pytest.raises(ValueError, match="Requires Isaac 6.1.0.0 / Python 3.12"):
        probe.probe()
    assert sdk_metadata[1] == []


@pytest.mark.parametrize("missing", ["isaacsim", "isaacsim-extscache-kit"])
def test_probe_missing_distribution_is_actionable(monkeypatch, sdk_metadata, missing):
    def absent(name):
        raise probe.importlib.metadata.PackageNotFoundError(name)

    method = "version" if missing == "isaacsim" else "distribution"
    monkeypatch.setattr(probe.importlib.metadata, method, absent)
    with pytest.raises(RuntimeError) as error:
        probe.probe()
    message = str(error.value)
    assert missing in message
    assert 'install "isaacsim[all,extscache]==6.1.0"' in message
    assert "https://pypi.org/simple" in message
    assert "https://pypi.nvidia.com" in message
    assert "--index-strategy unsafe-best-match" in message


def test_check_environment_parses_last_nonempty_line(monkeypatch):
    identity = {"isaacsim": "6.1.0.0", "python": "3.12.3"}
    result = SimpleNamespace(
        returncode=0,
        stdout='SDK diagnostic\n{"earlier": true}\n' + json.dumps(identity) + "\n \n",
        stderr="",
    )
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **kw: result)
    assert benchmark.check_environment("python") == identity


@pytest.mark.parametrize(
    "stdout", ["", " \n", "SDK diagnostic", '{"isaacsim": "6.1.0.0"}\ninvalid']
)
def test_check_environment_rejects_invalid_last_line(monkeypatch, stdout):
    result = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **kw: result)
    with pytest.raises(RuntimeError, match="valid JSON on the last non-empty stdout line"):
        benchmark.check_environment("python")


def test_check_environment_reports_failed_probe(monkeypatch):
    result = SimpleNamespace(returncode=1, stdout="not JSON", stderr="missing native library\n")
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **kw: result)
    with pytest.raises(
        RuntimeError, match="Isaac environment check failed:\nmissing native library"
    ):
        benchmark.check_environment("python")
