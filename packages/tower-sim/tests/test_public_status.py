"""The identity accompanies valid, invalid and diagnostic trial receipts."""

import json
from types import SimpleNamespace as NS

import pytest
from test_policy_evaluation import authorization as authorization
from test_replay_lifecycle import simulation as simulation
from tower_sim.bench.evaluation import EvaluationContext
from tower_sim.bench.verification import ReleaseVerification, envelope


@pytest.fixture
def provenance():
    public = envelope(dict(commit="a" * 40, manifest_sha256="b" * 64))
    return dict(
        public=public,
        **{k: public[k] for k in ("release_manifest_sha256", "commit", "data_manifest_sha256")},
    )


def test_replay_status_and_summary_include_identity(simulation, provenance):
    simulation.context.public_provenance = provenance
    assert simulation.run() == 0
    for path in (
        simulation.row["output"] / "status.json",
        simulation.context.output / "batch-policy-results.json",
    ):
        body = json.loads(path.read_text())
        assert {k: body[k] for k in provenance} == provenance


@pytest.mark.parametrize("complete", [False, True])
def test_formal_status_includes_identity_on_success_or_invalidation(
    authorization, provenance, tmp_path, monkeypatch, complete
):
    request, health, rows, task = authorization
    output = tmp_path / "trial"
    output.mkdir()
    rows[0].update(output=output)
    rows[0]["candidate"]["episode_spec"] = {}
    verifier = object.__new__(ReleaseVerification)
    verifier.identity = dict(public=provenance["public"])
    context = EvaluationContext(request, health, rows, task, verification=verifier)
    for name in (
        "episode_spec.json",
        "episode/states.hdf5",
        "commands.json",
        "finger-link-poses.npz",
        "finger-feedback.npz",
        "finger_geometry.json",
        "camera-frames.hdf5",
        "camera-readback.json",
        "runtime-sources-final.json",
    ):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    if complete:
        stamp = context.duration_ticks / 400
        commands = [dict(time=0.0), dict(time=stamp)]
        evidence = NS(records=[commands], env=NS(finger_geometry={}))
        monkeypatch.setattr(context, "write_evidence", lambda *a: output)
        context.scorer = NS(
            recording=lambda *a: NS(status={}),
            score=lambda recording: dict(valid=True, scores={"s": 0}, protocol="fixture"),
        )
        context.finalize_trial(
            0,
            evidence,
            commands,
            NS(times=[0.0, stamp], close=lambda **kw: None),
            {},
            {},
            tick=context.duration_ticks,
        )
    else:
        context.closed_archives.add(0)
        context.invalidate_unfinished(0, None, [], None, {}, {}, reason="startup rejected")
    result = json.loads((output / "status.json").read_text())
    assert result["valid_trial"] is complete
    assert {k: result[k] for k in provenance} == provenance
