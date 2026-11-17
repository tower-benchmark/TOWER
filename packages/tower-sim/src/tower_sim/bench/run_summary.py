"""Read-only aggregation of durable benchmark attempts; metrics are conditional."""

import re
from pathlib import Path
from typing import Any

from tower_eval.tasks import benchmark_task

from .resources import save_exclusive, sha
from .run_bundle import Bundle, inside, scene_ids
from .verification import envelope, read

UNREVIEWED = "tower-public-benchmark-protocol-unreviewed-v1"
REVIEWED = "tower-public-benchmark-protocol-v1"
REQUIRED = {
    "episode_spec.json",
    "episode/states.hdf5",
    "commands.json",
    "finger-link-poses.npz",
    "finger-feedback.npz",
    "finger_geometry.json",
    "score.json",
    "camera-frames.hdf5",
    "camera-readback.json",
    "runtime-sources-final.json",
}


def qualification(protocol, bundle):
    schema = protocol.get("schema")
    if schema not in (UNREVIEWED, REVIEWED):
        raise ValueError("Unsupported protocol schema")
    expected = "unreviewed" if schema == UNREVIEWED else "posture-reviewed"
    if protocol.get("qualification", "posture-reviewed") != expected:
        raise ValueError("Qualification differs from bound protocol schema")
    if schema == REVIEWED:
        from .ik_identity import shared_spec_identity

        if protocol["ik"]["shared_spec_sha256"] != shared_spec_identity():
            raise ValueError("Review uses a noncanonical IK identity")
        binding = protocol["pilot_review"]
        review = bundle.read(binding["path"], binding["sha256"])
        for name in (
            "passed",
            "native_inference_qualified",
            "scene_reference_review_passed",
            "shared_ik_parity_passed",
        ):
            if review.get(name) is not True:
                raise ValueError("Unverified maintainer review")
        if (
            review["package_digest"] != protocol["package_digest"]
            or review["shared_spec_sha256"] != protocol["ik"]["shared_spec_sha256"]
        ):
            raise ValueError("Review identity mismatch")
        posture = review["posture_review"]
        if (
            posture.get("schema") != "tower-no-reversed-grasp-review-v1"
            or posture.get("task") != protocol["task"]
            or posture.get("package_digest") != protocol["package_digest"]
        ):
            raise ValueError("Unbound posture review")
        for name in (
            "planned_approach_carry_passed",
            "recorded_approach_carry_passed",
            "reverse_180_grasp_absent",
            "task_contact_geometry_preserved",
        ):
            if posture.get(name) is not True:
                raise ValueError("Unverified posture review")
        for name in ("planned_trajectory", "recorded_approach_carry"):
            item = posture[name]
            bundle.path(item["path"], item["sha256"])
    return expected


def identity(bundle) -> tuple[dict, dict[str, Any]]:
    request = bundle.request
    protocol = bundle.field("protocol")
    runtime = bundle.field("runtime_manifest")
    if runtime.get("schema") != "tower-evaluation-inputs-v2":
        raise ValueError("Unsupported runtime inputs")
    for path, digest in runtime["files"].items():
        bundle.path(path, digest)
    package_sha = runtime["files"][request["package"]]
    package = bundle.read(request["package"], package_sha)
    checkpoint = package.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or re.fullmatch(r"[0-9a-f]{64}", checkpoint) is None:
        raise ValueError("Missing or malformed checkpoint identity")
    if not isinstance(package.get("package_digest"), str) or not package["package_digest"]:
        raise ValueError("Missing package identity")
    public = protocol["public"]
    if public != request["public"] or public.get("schema") != "tower-public-envelope-v1":
        raise ValueError("Public identity mismatch")
    for key in (
        "commit",
        "release_manifest_sha256",
        "data_manifest_sha256",
        "packages",
        "data_version",
        "benchmark",
        "request_schema",
    ):
        if not public.get(key):
            raise ValueError("Missing public identity")
    if (
        not isinstance(public["commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", public["commit"]) is None
    ):
        raise ValueError("Missing or malformed release commit")
    frozen = envelope(benchmark=public["benchmark"])
    for key in ("data_version", "request_schema", "packages", "data_manifest_sha256"):
        if public[key] != frozen[key]:
            raise ValueError("Public identity differs from frozen benchmark")
    release = bundle.field("release_manifest")
    if (
        release.get("schema") != "tower-release-v2"
        or release["commit"] != public["commit"]
        or request["release_manifest_sha256"] != public["release_manifest_sha256"]
    ):
        raise ValueError("Release identity mismatch")
    for key in ("dependency_manifest", "ik_dependency_manifest", "ik_bounds"):
        bundle.field(key)
    if protocol.get("ruleset", "bench-2026.11") != "bench-2026.11":
        raise ValueError("Mixed or unsupported ruleset")
    if package["package_digest"] != protocol["package_digest"]:
        raise ValueError("Package content differs from protocol")
    if request["task"] != protocol["task"] or request["seeds"] != protocol["seeds"]:
        raise ValueError("Request differs from protocol")
    from .ik_identity import shared_spec_identity

    if (
        protocol.get("scene_version") != "tower_v4"
        or protocol["ik"]["shared_spec_sha256"] != shared_spec_identity()
        or protocol["duration_s"] != request["duration"]
        or protocol["success_rule"] != request["success_rule"]
    ):
        raise ValueError("Protocol runtime contract mismatch")
    scene_ids(protocol["seeds"])
    return protocol, dict(
        benchmark=public["benchmark"],
        ruleset="bench-2026.11",
        task=benchmark_task(protocol["task"]).id,
        package_digest=package["package_digest"],
        package_sha256=package_sha,
        checkpoint_sha256=package["checkpoint_sha256"],
        qualification=qualification(protocol, bundle),
        public=public,
        policy=protocol["policy"],
        duration_s=protocol["duration_s"],
        success_rule=protocol["success_rule"],
        runtime={
            key: request[key + "_sha256"]
            for key in ("dependency_manifest", "ik_dependency_manifest", "ik_bounds")
        },
    )


def admitted(batch, request):
    # Use the parent's completion predicate, including valid nonzero teardown.
    from scripts.maintainer.candidate_policy import completion_state

    batch_path = batch / "simulation/batch-policy-results.json"
    if batch_path.is_file():
        for seed, binding in read(batch_path).get("committed_formal_trials", {}).items():
            if seed not in [str(n) for n in request["seeds"]]:
                raise ValueError("Unexpected committed scene")
            expected = str(Path(request["output"]) / f"simulation/seed-{seed}/status.json")
            if binding["path"] != expected:
                raise ValueError("Committed status path differs from request")
            read(batch / f"simulation/seed-{seed}/status.json", binding["sha256"])
    result_path = batch / "result.json"
    if not result_path.is_file():
        return False, "Missing parent completion receipt"
    result = read(result_path)
    boundary, complete = completion_state(dict(request, output=str(batch)))
    if not boundary or not complete or result.get("evidence_complete") is not True:
        return False, "Batch lacks qualified durable completion"
    rows = read(batch / "simulation/batch-policy-results.json")
    if rows.get("completed") is not True or rows.get("diagnostic_only") is not False:
        return False, "Batch is incomplete or diagnostic"
    for seed in request["seeds"]:
        status = read(batch / f"simulation/seed-{seed}/status.json")
        if status.get("valid_trial") is not True or status.get("evidence_complete") is not True:
            return False, "Batch contains an ineligible trial"
    return True, None


def annotation(states, score, output):
    try:
        from .annotations import publish

        return publish(states, score, output)
    except Exception as error:  # noqa: BLE001 -- no optional import/serialization can reject scores
        return dict(
            status="not_evaluated", operations=[], reason=f"Annotation unavailable: {error}"
        )


def aggregate(runs, output, *, initial=False):
    roots = []
    for directory in map(Path, runs):
        root = directory if (directory / "run.json").is_file() else directory.parent.parent
        if not (root / "run.json").is_file():
            raise ValueError("A durable run.json and retained attempt inputs are required")
        # Selecting a batch must not hide the rest of its original attempt history.
        if root.resolve() in [p.resolve() for p in roots]:
            raise ValueError("Duplicate scene source")
        roots.append(root)
    output = Path(output).resolve()
    if any(
        output.is_relative_to(root.resolve()) and not (initial and output == root.resolve())
        for root in roots
    ):
        raise ValueError("Aggregate output must not be inside input evidence")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "score.json").exists():
        raise ValueError("Aggregate output already exists")
    scenes, excluded, not_run, requested, attempted, sources = [], [], [], [], [], []
    seen = set()
    common = None
    task = None
    totals = dict(g=0, p=0, s=0, D=0, planned_g=0, planned_p=0)
    for root in roots:
        plan = read(root / "run.json")
        if plan.get("schema") != "tower-benchmark-run-v1":
            raise ValueError("Unsupported run schema")
        selected = scene_ids(plan["requested"])
        canonical = benchmark_task(plan["task"]).id
        if task is not None and task != canonical:
            raise ValueError("Mixed task")
        task = canonical
        if seen.intersection(selected):
            raise ValueError("Duplicate scene")
        seen.update(selected)
        requested += selected
        if [s for row in plan["batches"] for s in row["seeds"]] != plan["requested"]:
            raise ValueError("Batch plan differs from requested scenes")
        for row in plan["batches"]:
            name = row["id"]
            attempt = inside(root, "attempts/" + name)
            batch = inside(root, "runs/" + name)
            ids = scene_ids(row["seeds"])
            source = dict(
                run_id=plan["run_id"],
                run_manifest_sha256=sha(root / "run.json"),
                batch=name,
                scenes=ids,
                state="not_run",
            )
            sources.append(source)
            if not (attempt / "started.json").is_file():
                not_run += ids
                continue
            source["started_sha256"] = sha(attempt / "started.json")
            if (attempt / "finished.json").is_file():
                source["finished_sha256"] = sha(attempt / "finished.json")
                source["finish"] = read(attempt / "finished.json")
            attempted += ids
            source["state"] = "unprepared"
            if not (attempt / "prepared.json").is_file():
                excluded += [
                    dict(
                        scene=s,
                        status="infrastructure_failure",
                        reason="Preparation failed or interrupted",
                        source=source,
                    )
                    for s in ids
                ]
                continue
            bundle = Bundle(attempt)
            protocol, current = identity(bundle)
            if protocol["seeds"] != row["seeds"] or current["task"] != task:
                raise ValueError("Prepared request differs from selected attempt")
            if common is not None and current != common:
                raise ValueError(
                    "Mixed task/package/benchmark/ruleset/qualification/runtime identity"
                )
            common = current
            source.update(
                state="prepared",
                request_sha256=bundle.prepared["request_sha256"],
                protocol_sha256=bundle.request["protocol_sha256"],
                runtime_manifest_sha256=bundle.request["runtime_manifest_sha256"],
            )
            eligible, reason = admitted(batch, bundle.request)
            if not eligible:
                excluded += [
                    dict(scene=s, status="infrastructure_failure", reason=reason, source=source)
                    for s in ids
                ]
                continue
            batch_report = read(batch / "simulation/batch-policy-results.json")
            if (
                batch_report.get("public") != common["public"]
                or batch_report.get("checkpoint_package_digest") != common["package_digest"]
            ):
                raise ValueError("Batch public identity mismatch")
            source.update(
                state="completed",
                result_sha256=sha(batch / "result.json"),
                worker_complete_sha256=sha(batch / "worker-complete.json"),
                batch_results_sha256=sha(batch / "simulation/batch-policy-results.json"),
            )
            for seed, scene in zip(protocol["seeds"], ids, strict=True):
                episode = batch / "simulation" / f"seed-{seed}"
                status_path = episode / "status.json"
                committed = batch_report["committed_formal_trials"][str(seed)]
                status = read(status_path)
                if status.get("schema") != "tower-v4-standalone-policy-trial-v1":
                    raise ValueError("Unsupported trial schema")
                if (
                    status["protocol_sha256"] != bundle.request["protocol_sha256"]
                    or status["package_digest"] != current["package_digest"]
                    or type(status["seed"]) is not int
                    or status["seed"] != seed
                ):
                    raise ValueError("Episode differs from bound protocol")
                if (
                    status.get("public") != current["public"]
                    or status["checkpoint_sha256"] != current["checkpoint_sha256"]
                    or any(
                        status.get(k) != current["public"][k]
                        for k in ("commit", "release_manifest_sha256", "data_manifest_sha256")
                    )
                ):
                    raise ValueError("Episode provenance differs from bound identity")
                hashes = status["source_hashes"]
                if not REQUIRED.issubset(hashes):
                    raise ValueError("Missing required retained evidence")
                score_path = episode / "score.json"
                score_sha = sha(score_path)
                if score_sha != hashes["score.json"]:
                    raise ValueError("Aggregate score SHA mismatch")
                for name, digest in hashes.items():
                    if name == "score.json":
                        continue
                    if sha(inside(episode, name)) != digest:
                        raise ValueError("Aggregate retained evidence SHA mismatch")
                report = read(score_path)
                if report.get("ruleset", "bench-2026.11") != current["ruleset"]:
                    raise ValueError("Scene ruleset differs from benchmark")
                if report.get("valid") is not True or benchmark_task(report["task"]).id != task:
                    raise ValueError("Invalid scene score")
                posture = annotation(
                    episode / "episode/states.hdf5",
                    score_path,
                    output / "annotations" / (scene + ".json"),
                )
                scenes.append(
                    dict(
                        scene=scene,
                        status=status["status"],
                        scores=report["scores"],
                        score_sha256=score_sha,
                        status_sha256=committed["sha256"],
                        posture=posture,
                        source=source,
                    )
                )
                for key in totals:
                    totals[key] += (
                        report[key] if key.startswith("planned_") else report["scores"][key]
                    )
    if task is None:
        raise ValueError("No benchmark runs")
    c = totals["g"] + totals["p"]
    e = c + len(scenes) - totals["s"]

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    metrics = dict(
        GSR=ratio(totals["g"], totals["planned_g"]),
        PSR=ratio(totals["p"], totals["planned_p"]),
        LHPR=ratio(c, totals["planned_g"] + totals["planned_p"]),
        TSR=ratio(totals["s"], len(scenes)),
        ODR=ratio(totals["D"], e),
        C=c,
        E=e,
        trials=len(scenes),
    )
    groups = dict(
        requested=requested,
        attempted=attempted,
        scored=[s["scene"] for s in scenes],
        excluded=[s["scene"] for s in excluded],
        not_run=not_run,
    )
    coverage: dict[str, Any] = {
        key: dict(ids=sorted(value), count=len(value)) for key, value in groups.items()
    }
    coverage.update(
        requested_complete=len(scenes) == len(requested), full_task_coverage=len(scenes) == 50
    )
    result = dict(
        schema="tower-benchmark-score-v1",
        **(common or dict(task=task, qualification=None, benchmark=None, ruleset="bench-2026.11")),
        scenes=sorted(scenes, key=lambda row: row["scene"]),
        metrics=metrics,
        metric_scope="conditional_on_scored_eligible_trials",
        coverage=coverage,
        sources=sources,
        excluded=sorted(excluded, key=lambda row: row["scene"]),
    )
    save_exclusive(output / "score.json", result)
    return result
