"""Renderer failures and native teardown crashes remain distinct from policy outcomes."""

import json
import signal
import subprocess
import sys
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest
from tower_sim.bench import policy_entry
from tower_sim.bench.camera_quality import CameraQuality, camera_quality
from tower_sim.bench.evidence import CameraArchive
from tower_sim.bench.observation_validity import DeliveryGuard, renderer_failure

from scripts.maintainer import candidate_policy


def views():
    return np.random.default_rng(4).integers(0, 255, (4, 240, 320, 3), dtype=np.uint8)


def quality(pixels):
    return {str(i): camera_quality(image) for i, image in enumerate(pixels)}


def validate(pixels, reference=None, *, qualities=None):
    return DeliveryGuard().validate(
        pixels,
        1.0,
        quality(pixels) if qualities is None else qualities,
        render_step=1,
        synchronized=True,
        paused=True,
        delta_time_s=0,
        independent_reference=lambda: (
            pixels.copy() if reference is None else reference,
            dict(physical_state_preserved=True),
        ),
    )


@pytest.mark.parametrize("value", [0, 7, 128])
def test_renderer_wide_failure_is_not_physical_occlusion(value):
    pixels = np.full((4, 240, 320, 3), value, dtype=np.uint8)
    result = validate(pixels)
    assert result["valid"] is False
    assert result["reason"] == "Renderer failure in delivered camera views"


@pytest.mark.parametrize("count", [1, 2, 3])
def test_confirmed_partial_occlusion_remains_valid(count):
    pixels = views()
    pixels[:count] = 7
    result = validate(pixels)
    assert result["valid"] is True
    assert result["photometric_visibility"] is False
    assert result["comparison"]["passed"] is True
    assert len(result["reference_quality"]) == 4


@pytest.mark.parametrize("kind", ["zero", "format", "missing"])
def test_one_invalid_delivered_view_is_rejected(kind):
    pixels = views()
    qualities = quality(pixels)
    if kind == "zero":
        pixels[0] = 0
        qualities = quality(pixels)
    elif kind == "format":
        qualities["0"] = camera_quality(np.zeros((1,), dtype=np.uint8))
    else:
        qualities.pop("0")
    assert renderer_failure(qualities)
    assert validate(pixels, qualities=qualities)["valid"] is False


@pytest.mark.parametrize("kind", ["all-failed", "all-zero", "one-zero", "format"])
def test_independent_renderer_failure_is_rejected_even_with_matching_pixels(kind):
    # One healthy view just above the quality threshold; a flat reference is
    # within the historical MAE/bias tolerance but is still a renderer failure.
    pixels = np.full((4, 240, 320, 3), 5, np.uint8)
    pixels[3, ::2] = 21
    reference = pixels.copy()
    if kind == "all-failed":
        reference[3] = 13
    elif kind == "all-zero":
        reference[:] = 0
    elif kind == "one-zero":
        reference[0] = 0
    else:
        reference = reference.astype(float)
    result = validate(pixels, reference)
    assert result["valid"] is False
    if kind in ("all-failed", "one-zero"):
        assert result["comparison"]["passed"] is True
    if kind != "format":
        assert result["reason"] == "Renderer failure in independent camera views"


def test_blackout_uses_existing_delivery_failure_evidence(tmp_path):
    pixels = views()
    guard = DeliveryGuard()
    camera = NS(
        names=tuple(str(i) for i in range(4)),
        observe=lambda: dict(rgb=pixels[None]),
        validate_observation=lambda p, stamp, q: guard.validate(
            p,
            stamp,
            q,
            render_step=round(stamp * 30) + 1,
            synchronized=True,
            paused=True,
            delta_time_s=0,
            independent_reference=lambda: (p.copy(), dict(physical_state_preserved=True)),
        ),
    )
    path = tmp_path / "camera-frames.hdf5"
    archive = CameraArchive(camera, path, quality=CameraQuality())
    assert archive.append(0) is not None
    pixels[:] = 0
    with pytest.raises(RuntimeError, match="Invalid camera observation"):
        archive.append(1)
    assert archive.latest is None
    assert archive.all_observations_valid is False
    archive.close()
    detail = json.loads((tmp_path / "camera-frames.hdf5.diagnostics/00000001.json").read_text())
    assert detail["delivery"]["valid"] is False
    assert len(detail["cameras"]) == 4
    with h5py.File(path) as handle:
        assert not handle.attrs["complete"] and not handle.attrs["streams_consistent"]
        assert len(handle["time"]) == 1


FAKE_WORKER = """
import json, os, resource, signal, sys
from pathlib import Path
from tower_sim.bench.resources import save_exclusive
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
out, mode, diagnostic = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'
run = out / 'simulation/seed-1'
run.mkdir(parents=True)
if mode != 'before':
    save_exclusive(run / 'status.json', dict(evidence_complete=True,
        valid_trial=not diagnostic, diagnostic_only=diagnostic))
    save_exclusive(out / 'simulation/batch-policy-results.json', dict(completed=True))
if mode not in ('before', 'writing'):
    save_exclusive(out / 'worker-complete.json', dict(completed=mode != 'uncommitted',
        phase='evidence' if mode == 'wrong-phase' else 'teardown'))
os.kill(os.getpid(), signal.SIGSEGV)
"""


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize(
    "mode", ["complete", "before", "writing", "wrong-phase", "uncommitted", "bad-evidence"]
)
def test_native_crash_requires_durable_completion_and_valid_evidence(
    tmp_path, monkeypatch, diagnostic, mode
):
    output = tmp_path / "output"
    child = subprocess.run(
        [sys.executable, "-c", FAKE_WORKER, str(output), mode, str(diagnostic)],
        capture_output=True,
        check=False,
    )
    assert child.returncode == -signal.SIGSEGV, child.stderr.decode()
    scored = []

    def score(*a, **kw):
        scored.append(True)
        if mode == "bad-evidence":
            raise ValueError("Incomplete camera archive")
        return dict(physical_recording_valid=True)

    monkeypatch.setattr(policy_entry, "score_diagnostic", score)
    if mode == "bad-evidence" and not diagnostic:
        path = output / "simulation/seed-1/status.json"
        path.write_text(json.dumps(dict(valid_trial=True, evidence_complete=False)))
    request = dict(output=str(output), task="task1-2", seeds=[1], assets=str(candidate_policy.ROOT))
    code = candidate_policy.finish(request, child.returncode, diagnostic=diagnostic)
    receipt = json.loads(
        (output / ("diagnostic-result.json" if diagnostic else "result.json")).read_text()
    )
    complete = mode == "complete"
    assert (code == 0) == complete
    assert receipt["evidence_complete"] is complete
    assert receipt["worker_returncode"] == -signal.SIGSEGV
    assert receipt["worker_exit"] == dict(
        returncode=-signal.SIGSEGV,
        signal=signal.SIGSEGV,
        phase="teardown" if complete else "execution",
    )
    assert bool(scored) == (diagnostic and mode in ("complete", "bad-evidence"))


@pytest.mark.parametrize("code", [0, 7])
def test_process_exit_without_signal_is_recorded(code):
    assert policy_entry.worker_exit(code, completed=True) == dict(
        returncode=code, signal=None, phase="teardown"
    )


@pytest.mark.parametrize("failure", [None, "evidence", "writer-close", "teardown"])
def test_worker_publishes_boundary_after_evidence_and_before_teardown(
    tmp_path, monkeypatch, failure
):
    import benchmark_assets

    output = tmp_path / "output"
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            dict(
                output=str(output),
                task="task1-2",
                assets=str(tmp_path),
                manifest="unused",
                gpu=0,
                duration=1,
                trial_id="test",
            )
        )
    )
    marker = output / "worker-complete.json"
    events = []

    def record(name):
        events.append(name)
        assert marker.exists() == (name == "teardown" and failure != "evidence")
        if name == failure:
            raise RuntimeError(name)

    monkeypatch.setattr(candidate_policy.benchmark, "verify_bundle", lambda: None)
    monkeypatch.setattr(benchmark_assets, "load_rows", lambda *a: [])
    monkeypatch.setattr(candidate_policy, "build_identity", lambda *a: ("test", {}, {}, {}))
    monkeypatch.setattr(candidate_policy, "data_identity", lambda *a: {})
    monkeypatch.setattr(candidate_policy, "runtime_profile", lambda *a: {})
    monkeypatch.setattr(
        candidate_policy, "PolicyIK", lambda *a: NS(close=lambda: record("writer-close"))
    )
    monkeypatch.setattr(candidate_policy.J2Compensation, "from_cache", lambda *a: None)
    monkeypatch.setattr(candidate_policy, "observed_joint_validator", lambda *a, **kw: NS())
    monkeypatch.setattr(policy_entry, "validate_entry", lambda *a: None)
    monkeypatch.setattr(policy_entry, "prepare", lambda *a, **kw: (NS(close=lambda: None), None))
    monkeypatch.setattr(policy_entry, "execute", lambda *a, **kw: record("evidence") or 0)
    # The actual paths are irrelevant to this lifecycle test, not its ordering.
    resource_path = candidate_policy.Resources.path
    monkeypatch.setattr(
        candidate_policy.Resources,
        "path",
        lambda self, name: (
            tmp_path if name in ("scene_cache", "scene") else resource_path(self, name)
        ),
    )
    prepared = NS(
        context=NS(_drive_checks=1),
        collect_sources=lambda: {},
        close=lambda: record("teardown"),
    )
    args = NS(request=request_path, private_kit_root=None, ik_python=tmp_path, diagnostic=True)
    runner = NS(setup=lambda *a, **kw: prepared)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            candidate_policy.worker(args, runner=runner)
    else:
        candidate_policy.worker(args, runner=runner)
    assert marker.exists() == (failure in (None, "teardown"))
    if marker.exists():
        assert json.loads(marker.read_text()) == dict(
            completed=True, phase="teardown", gain_checks=1
        )
        assert (output / "fast-observed-joints.json").is_file()
