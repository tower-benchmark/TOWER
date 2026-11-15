# SPDX-License-Identifier: Apache-2.0
"""Analytic counterexamples to common deviations from the manuscript."""

import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from reproduction.paper_metrics.metrics import (
    ARTIFACTS,
    aggregate,
    ccc,
    directed_mmrv,
    pearson,
    source_agreement,
    summarize,
    validate,
)
from reproduction.paper_metrics.protocol import TASKS, group_keys, operation_plan
from reproduction.paper_metrics.__main__ import COLUMNS, main, read_csv


def episode(task="A", layers=2, **kw):
    ng, np = TASKS[task]["counts"][layers]
    return {
        "policy": "pi05",
        "task": task,
        "layers": layers,
        "domain": "SIM",
        "episode_id": "1",
        "valid": True,
        "invalid_reason": "",
        "s": 0,
        "g": 1,
        "p": 0,
        "d": 2,
        "planned_g": ng,
        "planned_p": np,
        "annotation_status": "agreed",
        "episode_artifact": "video:1",
        "checkpoint_artifact": "weights:1",
        "preprocessing_artifact": "preprocessing:1",
        "run_config_artifact": "config:1",
        **kw,
    }


def cohort():
    rows = []
    for policy, task, layers, domain in group_keys():
        ng, np = TASKS[task]["counts"][layers]
        for i in range(10):
            eid = f"{policy}/{task}/{layers}/{domain}/{i}"
            rows.append(
                episode(
                    task,
                    layers,
                    policy=policy,
                    domain=domain,
                    episode_id=eid,
                    episode_artifact=eid,
                    s=1,
                    g=ng,
                    p=np,
                    d=0,
                )
            )
    return rows


def write_episodes(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row | {"valid": int(row["valid"])})


def test_operation_totals_and_tool_cycle():
    for t in TASKS:
        for h in (2, 3):
            plan = operation_plan(t, h)
            assert (
                sum(p["group"] == "G" for p in plan),
                sum(p["group"] == "P" for p in plan),
            ) == TASKS[t]["counts"][h]
            assert len({p["operation_id"] for p in plan}) == len(plan)
    assert [p["group"] for p in operation_plan("F", 2)] == list("GGGPGPGP")
    assert operation_plan("C", 2)[0]["primitives"] == ["GRASP", "EXTRACT"]


def test_nonprefix_progress_terminal_failure_and_repeated_disturbance():
    # All 4 operations credited, but final structure collapsed; D may exceed E.
    r = aggregate([episode(g=2, p=2, s=0, d=7)])
    assert r["lhpr"] == 1 and r["tsr"] == 0
    assert r["sums"]["E"] == 5 and r["odr"] == 7 / 5
    # Early failure keeps every unstarted operation in the denominator.
    r = aggregate([episode(g=0, p=0, s=0, d=2)])
    assert r["lhpr"] == 0 and r["odr"] == 2


def test_pool_counts_instead_of_averaging_ratios():
    a, b = episode(g=0, p=0, d=5), episode(g=2, p=2, s=1, d=0)
    r = aggregate([a, b])
    assert r["odr"] == 1  # 5/(1+4), not (5+0)/2
    assert r["statistics"]["s"]["sample_sd"] == pytest.approx(2**-0.5)


def test_missing_fields_invalidate_only_dependent_metrics():
    r = aggregate([episode(d=None), episode(episode_id="2", s=1, g=2, p=2, d=0)])
    assert r["odr"] is None and r["tsr"] == 0.5 and r["lhpr"] == 5 / 8
    r = aggregate([episode(g=None)])
    assert r["lhpr"] is None and r["odr"] is None and r["psr"] == 0
    assert all(aggregate([])[k] is None for k in ("tsr", "gsr", "psr", "lhpr", "odr"))


def test_invalid_trials_excluded_from_every_metric():
    r = aggregate(
        [
            episode(s=1, g=2, p=2, d=0),
            episode(valid=False, invalid_reason="hardware failure", d=200),
        ]
    )
    assert r["tsr"] == 1 and r["odr"] == 0 and r["invalid_episodes"] == 1


def test_directed_ties_are_not_symmetric():
    assert directed_mmrv([0.5, 0.5], [0, 1]) == 0.5
    assert directed_mmrv([0, 1], [0.5, 0.5]) == 0
    assert directed_mmrv([1, 0], [0, 1]) == 1
    assert directed_mmrv([0, 0.5, 1], [0, 0.5, 1]) == 0


def test_correlation_and_ccc_degenerate_cases():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1)
    assert ccc([1, 2, 3], [2, 4, 6]) < 1
    assert pearson([1, 1], [2, 3]) is None
    assert ccc([1, 1], [1, 1]) is None


def test_complete_cohort_and_missing_group_not_silently_dropped():
    rows = cohort()
    result = summarize(rows)
    assert result["cohort_complete"] and result["cross_domain"]["paired_groups"] == 48
    assert all(row["rank"] == 1 for row in result["pooled_model_odr"])
    assert result["cross_domain"]["pearson_raw"] == pytest.approx(1)
    assert result["cross_domain"]["mmrv_normalized"] == 0
    with pytest.raises(ValueError, match="expected 10"):
        summarize(rows[:-1])
    partial = summarize(rows[:-1], strict=False)
    assert partial["cross_domain"]["pearson_raw"] is None
    assert partial["cross_domain"]["mmrv_normalized"] is None


@pytest.mark.parametrize(
    "update",
    [
        {"g": True},
        {"s": 2},
        {"g": 3},
        {"d": -1},
        {"planned_g": 3},
        {"s": 1, "g": 1},
        {"valid": None},
        {"valid": False, "invalid_reason": ""},
    ],
)
def test_malformed_labels_rejected_even_in_partial_mode(update):
    with pytest.raises(ValueError):
        validate([episode(**update)], strict=False)


def test_duplicate_identity_and_duplicate_evidence_rejected():
    with pytest.raises(ValueError, match="duplicate episode identity"):
        validate([episode(), episode()], strict=False)
    with pytest.raises(ValueError, match="Duplicate episode artifact"):
        validate([episode(), episode(episode_id="2")], strict=False)


def test_missing_annotation_or_provenance_blocks_strict_mode():
    rows = cohort()
    rows[0]["annotation_status"] = "pending"
    rows[1]["checkpoint_artifact"] = ""
    with pytest.raises(ValueError, match="Incomplete paper cohort"):
        validate(rows)


@pytest.mark.parametrize("field,other", [("g", "p"), ("p", "g")])
def test_incomplete_export_preserves_known_success_with_missing_counts(tmp_path, field, other):
    rows = cohort()
    rows[0][field] = None
    with pytest.raises(ValueError, match=f"missing {field}"):
        validate(rows)
    path = tmp_path / "episodes.csv"
    write_episodes(path, rows)
    out = tmp_path / "out"
    assert (
        main(["aggregate", "--episodes", str(path), "--output", str(out), "--allow-incomplete"])
        == 0
    )
    result = json.loads((out / "summary.json").read_text())
    group = result["groups"][0]
    assert not result["cohort_complete"]
    assert group["tsr"] == 1 and group[other + "sr"] == 1
    assert all(group[k] is None for k in (field + "sr", "lhpr", "odr"))
    assert result["cross_domain"]["pearson_raw"] is None
    # A supplied count still must agree with the success predicate.
    rows[0][other] = 1
    with pytest.raises(ValueError, match="Success requires all prescribed operations"):
        validate(rows, strict=False)


@pytest.mark.parametrize("field", ARTIFACTS)
def test_whitespace_artifact_references_block_strict_validation(tmp_path, field):
    rows = cohort()
    rows[0][field] = " \t\n "
    with pytest.raises(ValueError, match=f"missing {field}"):
        validate(rows)
    assert not summarize(rows, strict=False)["cohort_complete"]
    path = tmp_path / "episodes.csv"
    write_episodes(path, rows)
    parsed = read_csv(path)
    assert parsed[0][field] == ""
    with pytest.raises(ValueError, match=f"missing {field}"):
        validate(parsed)


def test_padding_cannot_hide_duplicate_episode_evidence():
    rows = [episode(), episode(episode_id="2", episode_artifact=" \tvideo:1 \n")]
    with pytest.raises(ValueError, match="Duplicate episode artifact"):
        validate(rows, strict=False)


def test_agreement_uses_source_labels_and_unicode_names():
    rows = cohort()
    ratings = []
    for r in rows:
        if r["task"] != "A":
            continue
        for annotator, p in [("z", 0), ("b", r["p"]), ("a", r["p"])]:
            ratings.append({**r, "annotator": annotator, "p": p})
    result = source_agreement(rows, ratings)
    assert len(result["sample"]) == 80 and result["agreement"]["combined"]["p"] == 1
    assert all(r["annotators"] == ["a", "b"] for r in result["sample"])
    assert source_agreement(rows, list(reversed(ratings))) == result
    assert source_agreement(rows, [])["agreement"]["combined"]["s"] is None
    with pytest.raises(ValueError, match="Duplicate source rating"):
        source_agreement(rows, ratings + [ratings[0]])


@pytest.mark.parametrize("missing", ["half", "one_rating", "one_field"])
def test_agreement_requires_the_full_source_population_before_sampling(missing):
    rows = cohort()
    ratings = []
    for row in rows:
        row["s"] = 0
        if row["task"] != "A":
            continue
        trial = int(row["episode_id"].rsplit("/", 1)[1])
        for annotator in ("a", "b"):
            ratings.append(
                {
                    **row,
                    "annotator": annotator,
                    "p": 0 if annotator == "b" and trial >= 5 else row["p"],
                }
            )
    complete = source_agreement(rows, ratings)
    assert complete["status"] == "complete" and len(complete["sample"]) == 80
    assert 0 < complete["agreement"]["combined"]["p"] < 1
    if missing == "half":
        # Removing the disagreements previously produced 100% agreement.
        ratings = [r for r in ratings if int(r["episode_id"].rsplit("/", 1)[1]) < 5]
    elif missing == "one_rating":
        ratings = ratings[1:]
    else:
        ratings[0]["p"] = None
    result = source_agreement(rows, ratings)
    assert result["status"] == "unavailable" and result["issues"]
    assert "expected 10 eligible episodes" in result["issues"][0]
    # No later domain is sampled from a shifted shared RNG stream.
    assert result["sample"] == []
    assert all(
        domain[field] is None
        for domain in result["agreement"].values()
        for field in ("s", "g", "p", "lhpr")
    )


def test_csv_cli_exports_and_refuses_overwrite(tmp_path):
    path = tmp_path / "episodes.csv"
    rows = cohort()
    for field in ARTIFACTS:
        rows[0][field] = " \t" + rows[0][field] + " \n"
    write_episodes(path, rows)
    assert len(read_csv(path)) == 960
    out = tmp_path / "out"
    assert main(["aggregate", "--episodes", str(path), "--output", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["cohort_complete"] and not summary["historical_results_verified"]
    for field in ARTIFACTS:
        assert summary["result_to_artifact"][0][field] == rows[0][field].strip()
    assert (out / "tables-14-17.csv").is_file()
    with pytest.raises(SystemExit):
        main(["aggregate", "--episodes", str(path), "--output", str(out)])


def test_template_does_not_fabricate_labels(tmp_path):
    out = tmp_path / "template.csv"
    assert main(["template", "--output", str(out)]) == 0
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 960 and all(r["valid"] == r["s"] == r["d"] == "" for r in rows)
