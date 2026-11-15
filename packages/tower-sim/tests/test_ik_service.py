import contextlib
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from tower_sim.bench.ik_client import IKServiceClient, IKServiceOptions, _write_all
from tower_sim.bench.ik_service import SCHEMA, service_identity, solve_request, validate_request

from scripts.maintainer.archive_root import ArchivedToolPath
from tests.characterization.runtime_scenarios import ROOT


def request(count=3):
    return dict(
        schema=SCHEMA,
        request_id="fixture:1",
        dt_s=0.1,
        poses=np.tile(np.eye(4), (count, 2, 1, 1)).tolist(),
        seed=np.zeros((2, 7)).tolist(),
    )


@pytest.mark.archive
@pytest.mark.parametrize("rejected_at", [None, 0, 1, 3])
def test_service_dispatch_matches_frozen(frozen_loader, monkeypatch, rejected_at):
    original = frozen_loader("_p705_service", ArchivedToolPath("generation_ik_service.py"))
    all_calls = []

    def make_solver():
        calls = []
        all_calls.append(calls)

        def encoder(target, seed, arm):
            calls.append((target.tolist(), seed.tolist(), arm))
            accepted = len(calls) - 1 != rejected_at
            return dict(
                accepted=accepted,
                joints=(seed + 0.001).tolist() if accepted else None,
                reason="fixture",
            )

        return SimpleNamespace(encoder=encoder, identity=dict(fallback="none"))

    monkeypatch.setitem(
        sys.modules, "shared_ik", SimpleNamespace(SharedIK=make_solver, SPEC_SHA="fixture")
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request()) + "\n"))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        original.main()
    old = json.loads(output.getvalue())
    new = solve_request(request(), make_solver())
    for key in ("accepted", "joints", "diagnostics", "identity"):
        assert new[key] == old[key]
    assert all_calls[0] == all_calls[1]
    assert len(all_calls[1]) == (6 if rejected_at is None else rejected_at + 1)


@pytest.mark.parametrize(
    "case", ["count", "shape", "seed", "nan", "dt", "rotation", "reflection", "bottom", "schema"]
)
def test_service_rejects_invalid_before_solving(case):
    req = request()
    if case == "count":
        req["poses"] = []
    if case == "shape":
        req["poses"] = [np.eye(4).tolist()]
    if case == "seed":
        req["seed"] = [0] * 7
    if case == "nan":
        req["poses"][0][0][0][0] = float("nan")
    if case == "dt":
        req["dt_s"] = 0.2
    if case == "rotation":
        req["poses"][0][0][0][0] = 2
    if case == "reflection":
        req["poses"][0][0][0][0] = -1
    if case == "bottom":
        req["poses"][0][0][3][3] = 0
    if case == "schema":
        req["schema"] = "unversioned"
    with pytest.raises(ValueError):
        validate_request(req)


@pytest.mark.parametrize(
    "behavior",
    [
        "accept",
        "reject",
        "error",
        "error-id",
        "error-matching",
        "profile_sha256",
        "id",
        "malformed",
        "eof",
        "partial",
        "nonfinite",
        "tower-sim",
        "nero-kinematics",
        "command_source_sha256",
        "solver_profile_id",
        "missing-identity",
        "changed-identity",
    ],
)
def test_subprocess_lifecycle_and_fail_closed(tmp_path, behavior):
    worker = tmp_path / "worker"
    worker.write_text(f"""#!{sys.executable}
import json, os, sys, time
assert all(os.environ[k] == '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
for line in sys.stdin:
    request = json.loads(line)
    behavior = {behavior!r}
    result = dict(schema={SCHEMA!r}, request_id=request['request_id'], accepted=behavior != 'reject',
                  joints=[request['seed']] if behavior != 'reject' else None, diagnostics=[],
                  service_identity={service_identity()!r})
    if behavior == 'error': result = dict(service_error='fixture parse failure', request_id=None)
    if behavior in ('error-id', 'error-matching'): result = dict(service_error='fixture parse failure', request_id='wrong' if behavior == 'error-id' else request['request_id'])
    if behavior == 'profile_sha256': result['service_identity']['profile_sha256']['command.json'] = 'wrong'
    if behavior in ('tower-sim', 'nero-kinematics'): result['service_identity']['packages'][behavior] = 'wrong'
    if behavior in ('command_source_sha256', 'solver_profile_id'): result['service_identity'][behavior] = 'wrong'
    if behavior == 'missing-identity': del result['service_identity']
    if behavior == 'changed-identity' and request['request_id'] != '1': result['service_identity']['solver_profile_id'] = 'wrong'
    if behavior == 'id': result['request_id'] = 'wrong'
    if behavior == 'eof': sys.exit(0)
    if behavior == 'partial': sys.stdout.write('{{'); sys.stdout.flush(); time.sleep(10)
    if behavior == 'nonfinite': result['joints'][0][0][0] = float('nan')
    print('not json' if behavior == 'malformed' else json.dumps(result), flush=True)
""")
    worker.chmod(0o700)
    client = IKServiceClient(
        IKServiceOptions(worker, response_timeout_s=0.25 if behavior == "partial" else 5.0)
    )
    req = request(1)
    if behavior in ("accept", "reject"):
        with client:
            for _ in range(2):
                result = client.solve(req["poses"], req["seed"])
                assert result["accepted"] is (behavior == "accept")
            process = client._process
        assert process.poll() is not None
    else:
        if behavior == "changed-identity":
            assert client.solve(req["poses"], req["seed"])["accepted"]
        match = (
            "fixture parse failure"
            if behavior in ("error", "error-matching")
            else "identity mismatch"
            if behavior == "error-id"
            else None
        )
        with pytest.raises((ValueError, RuntimeError), match=match):
            client.solve(req["poses"], req["seed"])
        assert client._process is None
    client.close()


def test_invalid_request_does_not_start_process(tmp_path):
    client = IKServiceClient(IKServiceOptions(tmp_path / "does-not-exist"))
    with pytest.raises(ValueError):
        client.solve([], np.zeros((2, 7)))
    assert client._process is None


@pytest.mark.parametrize(
    "reference", [False, pytest.param(True, marks=pytest.mark.archive)], ids=["public", "frozen"]
)
def test_native_service_goldens_and_real_process(reference):
    python = os.environ.get("TOWER_IK_TEST_PYTHON", sys.executable)
    if (
        os.environ.get("TOWER_REQUIRE_NATIVE_IK") != "1"
        and "TOWER_IK_TEST_PYTHON" not in os.environ
    ):
        pytest.skip("Native posture-IPOPT service runs in the two IK CI environments")
    if reference:
        from scripts.maintainer.archive_root import archive_bundle

        archive_bundle()
    # Fresh interpreter: preserve native ABI isolation and single-thread BLAS.
    script = "import runpy; runpy.run_path('packages/tower-sim/tests/native_ik_equivalence.py', run_name='__main__')"
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    result = subprocess.run(
        [python, "-c", script, *(["--reference"] if reference else [])],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"Native probe failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_partial_pipe_writes_preserve_frame():
    data = bytearray()
    stream = SimpleNamespace(
        write=lambda chunk: data.extend(chunk[:3]) or len(chunk[:3]),
        flush=lambda: None,
    )
    payload = (json.dumps(request()) + "\n").encode()
    _write_all(stream, payload)
    assert bytes(data) == payload


@pytest.mark.parametrize("written", [None, 0])
def test_pipe_write_no_progress_fails(written):
    stream = SimpleNamespace(write=lambda chunk: written, flush=lambda: None)
    with pytest.raises(RuntimeError, match="closed input transport"):
        _write_all(stream, b"request\n")


def test_native_stdout_cannot_corrupt_protocol():
    script = """
import ctypes, os, sys
from types import SimpleNamespace
from tower_sim.bench.ik_service import main

def noise():
    print('python diagnostic', flush=True)
    os.write(1, b'fd1 diagnostic\\n')
    libc = ctypes.CDLL(None)
    libc.puts(b'C diagnostic')
    libc.fflush(None)

class FakeSolver:
    identity = {'fallback': 'none'}
    def __init__(self): noise()
    def encoder(self, target, seed, arm):
        noise()
        return dict(accepted=True, joints=seed.tolist(), reason='fixture')

sys.modules['nero_kinematics.solvers.shared'] = SimpleNamespace(SharedIK=FakeSolver)
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(request(1)) + "\n{invalid json\n",
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    frames = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(frames) == 2 and frames[0]["accepted"]
    assert frames[0]["service_identity"] == service_identity()
    assert frames[1]["request_id"] is None and "JSONDecodeError" in frames[1]["service_error"]
    for diagnostic in ("python diagnostic", "fd1 diagnostic", "C diagnostic"):
        assert result.stderr.count(diagnostic) == 3


def test_identity_hashes_loaded_profiles_without_installed_distributions(monkeypatch):
    import hashlib
    from importlib.metadata import PackageNotFoundError
    from importlib.resources import files

    from tower_sim.bench import ik_service

    def absent(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(ik_service, "version", absent)
    identity = ik_service.service_identity()
    assert identity["packages"] == {"tower-sim": "0.1.0", "nero-kinematics": "0.2.0"}
    for name in ("command.json", "posture-ipopt.json"):
        assert (
            identity["profile_sha256"][name]
            == hashlib.sha256(
                files("nero_kinematics.profiles").joinpath(name).read_bytes()
            ).hexdigest()
        )
