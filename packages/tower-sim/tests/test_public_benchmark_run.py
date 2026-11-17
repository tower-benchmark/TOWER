"""Public scene selection and aggregation retain evidence and qualification."""

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import h5py
import pytest
from tower_sim.bench import run_summary
from tower_sim.bench.resources import sha

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools/simulator"))
    return importlib.import_module("benchmark_run")


def test_scene_selection(entry):
    assert entry.parse_scenes("3,7,10-12") == [3, 7, 10, 11, 12]
    assert entry.parse_scenes("0-49") == list(range(50))
    for bad in ("", "50", "-1", "4-2", "3,3", "1-3,2", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            entry.parse_scenes(bad)


@pytest.mark.parametrize("keep", [False, True])
def test_batches_stop_and_keep_going(entry, monkeypatch, tmp_path, keep, capsys):
    public = importlib.import_module("benchmark_candidate")
    prepared = []

    def prepare(args, **kwargs):
        prepared.append(args.seeds)
        assert kwargs == dict(diagnostic=False)
        path = tmp_path / f"request-{len(prepared)}.json"
        path.write_text("{}")
        return path

    monkeypatch.setattr(public, "prepare", prepare)
    monkeypatch.setattr(public, "launch", lambda *a, **kw: 1)
    monkeypatch.setattr(run_summary, "aggregate", lambda *a: None)
    monkeypatch.setattr(entry.benchmark, "require_eula", lambda *a: None)
    result = entry.run(
        NS(
            scenes=[0, 1, 2],
            batch_size=2,
            output=tmp_path / "out",
            keep_going=keep,
            accept_eula=True,
        )
    )
    assert result == 1
    assert prepared == ([[0, 1], [2]] if keep else [[0, 1]])
    if not keep:
        assert "[2]" in capsys.readouterr().out


def batch(tmp_path, seed=0, *, complete=True, task="task6-2", qualification="unreviewed"):
    root = tmp_path / f"batch-{seed}"
    episode = root / "simulation" / f"seed-{seed}"
    (episode / "episode").mkdir(parents=True)
    protocol = root / "protocol.json"
    protocol.write_text(
        json.dumps(
            dict(
                schema=run_summary.UNREVIEWED,
                qualification=qualification,
                task=task,
                seeds=[seed],
                package_digest="package",
                public=dict(benchmark="2026.11.1"),
            )
        )
    )
    (root / "benchmark-request.json").write_text(
        json.dumps(
            dict(protocol=str(protocol), protocol_sha256=sha(protocol), qualification="fake")
        )
    )
    score = episode / "score.json"
    score.write_text(
        json.dumps(
            dict(
                task=task,
                valid=True,
                scores=dict(g=1, p=0, s=1, C=1, E=1, D=2),
                planned_g=1,
                planned_p=1,
            )
        )
    )
    with h5py.File(episode / "episode/states.hdf5", "w"):
        pass
    status = dict(
        seed=seed,
        valid_trial=complete,
        evidence_complete=complete,
        status="duration_reached" if complete else "infrastructure_failure",
        protocol_sha256=sha(protocol),
        package_digest="package",
        source_hashes={name: sha(episode / name) for name in ("score.json", "episode/states.hdf5")},
    )
    (episode / "status.json").write_text(json.dumps(status))
    return root


def test_aggregate_metrics_exclusions_and_bound_qualification(tmp_path):
    runs = [batch(tmp_path), batch(tmp_path, 1, complete=False)]
    value = run_summary.aggregate(runs, tmp_path / "out")
    assert value["qualification"] == "unreviewed"
    assert value["metrics"] == dict(GSR=1, PSR=0, LHPR=0.5, TSR=1, ODR=2, C=1, E=1, trials=1)
    assert len(value["excluded"]) == 1
    assert value["scenes"][0]["posture"]["status"] == "not_evaluated"
    assert (runs[0] / "simulation/seed-0/posture-flags.json").exists()


def test_aggregate_sha_and_duplicates(tmp_path):
    root = batch(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        run_summary.aggregate([root, root], tmp_path / "duplicate")
    score = root / "simulation/seed-0/score.json"
    score.write_text(score.read_text() + " ")
    with pytest.raises(ValueError, match="SHA"):
        run_summary.aggregate([root], tmp_path / "tampered")


def test_aggregate_mixed_and_faked_qualification(tmp_path):
    one = batch(tmp_path)
    two = batch(tmp_path, 1, task="task1-2")
    with pytest.raises(ValueError, match="Mixed"):
        run_summary.aggregate([one, two], tmp_path / "mixed")
    fake = batch(tmp_path, 2, qualification="posture-reviewed")
    with pytest.raises(ValueError, match="Qualification"):
        run_summary.aggregate([fake], tmp_path / "fake")
