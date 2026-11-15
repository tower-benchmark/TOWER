# SPDX-License-Identifier: Apache-2.0
"""Pure standard-library aggregation of final human labels; no trajectory proxy."""

from collections import Counter, defaultdict
import math
import random
import statistics

from .protocol import DOMAINS, POLICIES, TASKS, TRIALS_PER_GROUP, group_keys

IDENTITY = ("policy", "task", "layers", "domain", "episode_id")
LABELS = ("s", "g", "p", "d")
ARTIFACTS = (
    "episode_artifact",
    "checkpoint_artifact",
    "preprocessing_artifact",
    "run_config_artifact",
)


def identity(row):
    return tuple(row[k] for k in IDENTITY)


def validate(rows, *, strict=True):
    seen = set()
    artifacts = set()
    groups = Counter()
    issues = []
    for row in rows:
        key = identity(row)
        policy, task, layers, domain, episode_id = key
        if (
            policy not in POLICIES
            or task not in TASKS
            or layers not in (2, 3)
            or domain not in DOMAINS
        ):
            raise ValueError(f"Unknown configuration: {key}")
        if not isinstance(episode_id, str) or not episode_id.strip() or key in seen:
            raise ValueError(f"Empty or duplicate episode identity: {key}")
        seen.add(key)
        artifact = row.get("episode_artifact")
        if isinstance(artifact, str):
            artifact = artifact.strip()
        if artifact:
            # Independent episodes cannot point to the same evidence in a domain.
            artifact_key = (domain, artifact)
            if artifact_key in artifacts:
                raise ValueError(f"Duplicate episode artifact: {artifact_key}")
            artifacts.add(artifact_key)
        if type(row.get("valid")) is not bool:
            raise ValueError(f"Validity must be explicit for {key}")
        ng, np = TASKS[task]["counts"][layers]
        if (row.get("planned_g"), row.get("planned_p")) != (ng, np):
            raise ValueError(f"Planned counts disagree with Table 4: {key}")
        for field in ("planned_g", "planned_p"):
            if type(row[field]) is not int:
                raise ValueError(f"{field} must be an integer: {key}")
        for field, upper in (("s", 1), ("g", ng), ("p", np), ("d", None)):
            value = row.get(field)
            if value is not None and (
                type(value) is not int or value < 0 or (upper is not None and value > upper)
            ):
                raise ValueError(f"Invalid {field}={value!r}: {key}")
        if row.get("s") == 1 and any(
            row.get(field) is not None and row[field] != planned
            for field, planned in (("g", ng), ("p", np))
        ):
            raise ValueError(f"Success requires all prescribed operations: {key}")
        if row["valid"]:
            groups[key[:4]] += 1
            if row.get("invalid_reason"):
                raise ValueError(f"Valid episode has invalid_reason: {key}")
            missing = [f for f in LABELS if row.get(f) is None]
            missing.extend(
                f for f in ARTIFACTS if not isinstance(row.get(f), str) or not row[f].strip()
            )
            if missing:
                issues.append(f"{key}: missing {', '.join(missing)}")
            if row.get("annotation_status") not in ("agreed", "adjudicated"):
                issues.append(f"{key}: final human annotation is not agreed/adjudicated")
        elif not row.get("invalid_reason"):
            raise ValueError(f"Invalid episode needs an external-to-policy reason: {key}")
    for key in group_keys():
        if groups[key] != TRIALS_PER_GROUP:
            issues.append(f"{key}: expected 10 valid episodes, found {groups[key]}")
    if strict and issues:
        raise ValueError("Incomplete paper cohort:\n" + "\n".join(issues))
    return issues


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def total(rows, field):
    values = [row.get(field) for row in rows]
    return sum(values) if values and all(v is not None for v in values) else None


def derived(row, name):
    if name == "C":
        return (
            row["g"] + row["p"] if row.get("g") is not None and row.get("p") is not None else None
        )
    if name == "E":
        c = derived(row, "C")
        return c + 1 - row["s"] if c is not None and row.get("s") is not None else None
    return row.get(name)


def aggregate(rows):
    valid = [r for r in rows if r["valid"]]
    values = [{**r, "C": derived(r, "C"), "E": derived(r, "E")} for r in valid]
    sums = {
        name: total(values, name)
        for name in ("s", "g", "p", "d", "C", "E", "planned_g", "planned_p")
    }

    def metric(num, den):
        return ratio(num, den) if num is not None and den is not None else None

    stats = {}
    for name in ("s", "g", "p", "C", "d"):
        v = [r[name] for r in values]
        complete = bool(v) and all(x is not None for x in v)
        stats[name] = {
            "mean": statistics.mean(v) if complete else None,
            "sample_sd": statistics.stdev(v) if complete and len(v) > 1 else None,
        }
    return {
        "valid_episodes": len(valid),
        "invalid_episodes": len(rows) - len(valid),
        "tsr": metric(sums["s"], len(valid)),
        "gsr": metric(sums["g"], sums["planned_g"]),
        "psr": metric(sums["p"], sums["planned_p"]),
        "lhpr": metric(sums["C"], sum(r["planned_g"] + r["planned_p"] for r in valid)),
        "odr": metric(sums["d"], sums["E"]),
        "mean_successful_operations": stats["C"]["mean"],
        "sums": sums,
        "statistics": stats,
    }


def pearson(x, y):
    if len(x) < 2:
        return None
    xm, ym = statistics.mean(x), statistics.mean(y)
    xx, yy = sum((v - xm) ** 2 for v in x), sum((v - ym) ** 2 for v in y)
    return (
        sum((a - xm) * (b - ym) for a, b in zip(x, y)) / math.sqrt(xx * yy) if xx and yy else None
    )


def ccc(x, y):
    if not x:
        return None
    xm, ym = statistics.mean(x), statistics.mean(y)
    vx = statistics.mean((v - xm) ** 2 for v in x)
    vy = statistics.mean((v - ym) ** 2 for v in y)
    cov = statistics.mean((a - xm) * (b - ym) for a, b in zip(x, y))
    return ratio(2 * cov, vx + vy + (xm - ym) ** 2)


def directed_mmrv(sim, real):
    """Equation 15: strict directed comparisons, including asymmetric ties."""
    return statistics.mean(
        max(
            [
                abs(real[m] - real[j])
                for j in range(len(real))
                if j != m and (sim[m] > sim[j]) != (real[m] > real[j])
            ]
            or [0.0]
        )
        for m in range(len(real))
    )


def summarize(rows, *, strict=True):
    issues = validate(rows, strict=strict)
    grouped = defaultdict(list)
    for r in rows:
        grouped[identity(r)[:4]].append(r)
    groups = {key: aggregate(grouped[key]) for key in group_keys()}
    paired = []
    mmrv = []
    for task in TASKS:
        for layers in (2, 3):
            k = sum(TASKS[task]["counts"][layers])
            sim, real = [], []
            for policy in POLICIES:
                sg, rg = (
                    groups[(policy, task, layers, "SIM")],
                    groups[(policy, task, layers, "REAL")],
                )
                x, y = sg["mean_successful_operations"], rg["mean_successful_operations"]
                complete = sg["valid_episodes"] == rg["valid_episodes"] == TRIALS_PER_GROUP
                if x is None or y is None or not complete:
                    continue
                paired.append(
                    {
                        "policy": policy,
                        "task": task,
                        "layers": layers,
                        "planned": k,
                        "sim": x,
                        "real": y,
                    }
                )
                sim.append(x / k)
                real.append(y / k)
            if len(sim) == len(POLICIES):
                mmrv.append(directed_mmrv(sim, real))
    cross = {
        "paired_groups": len(paired),
        "expected_pairs": 48,
        "pearson_raw": None,
        "mae_operations": None,
        "mmrv_normalized": None,
        "pearson_normalized": None,
        "mae_percentage_points": None,
        "ccc_normalized": None,
    }
    # Never silently shrink the fixed comparison set for partial exports.
    if len(paired) == 48:
        x, y = [p["sim"] for p in paired], [p["real"] for p in paired]
        u, v = [p["sim"] / p["planned"] for p in paired], [p["real"] / p["planned"] for p in paired]
        cross.update(
            pearson_raw=pearson(x, y),
            mae_operations=statistics.mean(abs(a - b) for a, b in zip(x, y)),
            mmrv_normalized=statistics.mean(mmrv),
            pearson_normalized=pearson(u, v),
            mae_percentage_points=100 * statistics.mean(abs(a - b) for a, b in zip(u, v)),
            ccc_normalized=ccc(u, v),
        )
    pooled = []
    for policy in POLICIES:
        for domain in DOMAINS:
            subset = [r for r in rows if r["policy"] == policy and r["domain"] == domain]
            complete = all(
                groups[(policy, t, h, domain)]["valid_episodes"] == 10
                for t in TASKS
                for h in (2, 3)
            )
            result = aggregate(subset)
            pooled.append(
                {
                    "policy": policy,
                    "domain": domain,
                    "valid_episodes": result["valid_episodes"],
                    "odr": result["odr"] if complete else None,
                }
            )
    for domain in DOMAINS:
        entries = [r for r in pooled if r["domain"] == domain]
        complete = all(r["odr"] is not None for r in entries)
        for r in entries:
            r["rank"] = (
                (1 + sum(other["odr"] < r["odr"] for other in entries)) if complete else None
            )
    return {
        "issues": issues,
        "cohort_complete": not issues,
        "groups": [dict(zip(IDENTITY[:4], key)) | value for key, value in groups.items()],
        "pooled_model_odr": pooled,
        "cross_domain": cross,
        "paired_means": paired,
        "annotation_status": dict(Counter(r.get("annotation_status", "missing") for r in rows)),
        "result_to_artifact": [
            {k: r.get(k) for k in (*IDENTITY, *ARTIFACTS, "valid", "annotation_status")}
            for r in rows
        ],
    }


def source_agreement(rows, ratings):
    """Appendix B.2: seed-42 sample; deterministic sort then a single RNG stream.

    Sampling order is explicitly versioned here; original selected identities are
    needed to establish identity with the historical Table 7 sample.
    """
    episodes = {identity(r): r for r in rows}
    by_episode = defaultdict(list)
    seen = set()
    for rating in ratings:
        key = identity(rating)
        annotator = rating.get("annotator")
        if key not in episodes or not isinstance(annotator, str) or not annotator.strip():
            raise ValueError(f"Unknown episode or missing annotator: {key}")
        if (key, annotator) in seen:
            raise ValueError(f"Duplicate source rating: {key}, {annotator}")
        seen.add((key, annotator))
        ng, np = TASKS[rating["task"]]["counts"][rating["layers"]]
        for field, upper in (("s", 1), ("g", ng), ("p", np), ("d", None)):
            value = rating.get(field)
            if value is not None and (
                type(value) is not int or value < 0 or (upper is not None and value > upper)
            ):
                raise ValueError(f"Invalid source {field}: {key}")
        by_episode[key].append(rating)
    sampled = []
    unavailable = []
    candidate_groups = []
    for domain in DOMAINS:
        for policy in POLICIES:
            for layers in (2, 3):
                candidates = []
                for key, episode in sorted(episodes.items()):
                    rr = sorted(by_episode[key], key=lambda r: r["annotator"])
                    if key[:4] != (policy, "A", layers, domain) or not episode["valid"]:
                        continue
                    # Select first two names before considering their scores.
                    if len(rr) >= 2 and all(
                        r.get(f) is not None for r in rr[:2] for f in ("s", "g", "p")
                    ):
                        candidates.append((key, rr[0], rr[1]))
                if len(candidates) != TRIALS_PER_GROUP:
                    unavailable.append(
                        f"{domain}/{policy}/A{layers}: expected {TRIALS_PER_GROUP} "
                        f"eligible episodes, found {len(candidates)}"
                    )
                candidate_groups.append(candidates)
    # Validate all 80 candidates per domain before consuming the shared RNG.
    # Skipping an incomplete group would also change later groups' selections.
    if not unavailable:
        rng = random.Random(42)
        for candidates in candidate_groups:
            for key, a, b in rng.sample(candidates, 5):
                sampled.append(
                    {
                        **dict(zip(IDENTITY, key)),
                        "annotators": [a["annotator"], b["annotator"]],
                        "s": a["s"] == b["s"],
                        "g": a["g"] == b["g"],
                        "p": a["p"] == b["p"],
                        "lhpr": a["g"] + a["p"] == b["g"] + b["p"],
                    }
                )
    out = {
        "status": "unavailable" if unavailable else "complete",
        "seed": 42,
        "sample_order": "REAL,SIM; pi05,starvla,fastwam,openwam; 2,3; sorted episode identity",
        "issues": unavailable,
        "sample": sampled,
        "agreement": {},
    }
    for domain in (*DOMAINS, "combined"):
        selected = [r for r in sampled if domain == "combined" or r["domain"] == domain]
        required = 80 if domain == "combined" else 40
        out["agreement"][domain] = {
            "episodes": len(selected),
            **{
                f: statistics.mean(r[f] for r in selected) if len(selected) == required else None
                for f in ("s", "g", "p", "lhpr")
            },
        }
    return out
