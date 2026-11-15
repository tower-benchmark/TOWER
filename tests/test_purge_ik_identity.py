"""The public runner and independent IK service bind the same unchanged identity."""

import pytest
from tower_sim.bench import ik_identity, ik_service


def test_single_shared_identity(monkeypatch):
    import importlib.util
    import sys
    from importlib.resources import files
    from types import SimpleNamespace as NS

    # This test verifies the actual SharedIK identity constructor; solver math
    # has its separate native suite. Keep the pure identity check runnable in CI.
    monkeypatch.setitem(
        sys.modules,
        "nero_kinematics.solvers.posture_ipopt",
        NS(Solvers=lambda *args: NS(fk_error=0)),
    )
    path = files("nero_kinematics.solvers").joinpath("shared.py")
    spec = importlib.util.spec_from_file_location("nero_kinematics.solvers._identity_test", path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    assert (
        ik_identity.shared_spec_identity()
        == shared.SharedIK().identity["spec_sha256"]
        == ik_service.service_identity()["command_source_sha256"]
    )


def test_tampered_spec_fails(monkeypatch, tmp_path):
    path = tmp_path / "benchmark/runtime/shared/shared-ik-spec.json"
    path.parent.mkdir(parents=True)
    path.write_bytes((ik_identity.DATA / path.relative_to(tmp_path)).read_bytes() + b"\n")
    monkeypatch.setattr(ik_identity, "DATA", tmp_path)
    with pytest.raises(ValueError, match="Changed redacted shared IK"):
        ik_identity.shared_spec_identity()
