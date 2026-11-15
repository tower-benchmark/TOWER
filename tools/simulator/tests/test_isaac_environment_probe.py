import pytest

from tools.simulator import isaac_environment_probe as probe


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
