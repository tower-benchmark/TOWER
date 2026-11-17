"""Standalone policy evidence and authorization, using the canonical scorer.

The caller binds candidate sources in its runtime manifest. Frozen manifests
cannot authorize new code. This entry never grants official episode credit.
"""

import json
from pathlib import Path

from .ik_identity import shared_spec_identity
from .resources import save_exclusive, sha
from .scoring import BenchScorer
from .verification import read as read_bound


def immutable(path, value):
    # Preserve frozen NumPy-to-JSON evidence encoding with atomic publication.
    save_exclusive(
        path, json.loads(json.dumps(value, allow_nan=False, default=lambda v: v.tolist()))
    )


def read_json(path):
    return json.loads(Path(path).read_text())


class EvaluationContext:
    def __init__(self, request, health, rows, task, *, verification=None):
        self.verification = verification
        self.request = request
        self.health = health
        self.rows = rows
        self.completed = {}
        self.closed_archives = set()
        self.task = task.identity.legacy_id
        self.scorer = BenchScorer(task)
        if request["duration"] != task.identity.deadline_s:
            raise ValueError("Formal duration differs from benchmark deadline")
        self.success_rule = request["success_rule"]
        self.duration_ticks = round(request["duration"] * 400)
        self.package = read_json(request["package"])
        self.protocol = read_bound(request["protocol"], request["protocol_sha256"])
        self.files = read_bound(request["runtime_manifest"], request["runtime_manifest_sha256"])[
            "files"
        ]
        if not (sha(request["protocol"]) == request["protocol_sha256"]):
            raise ValueError(
                "Invalid evaluation evidence: sha(request['protocol']) == request['protocol_sha256']"
            )
        if not (sha(request["runtime_manifest"]) == request["runtime_manifest_sha256"]):
            raise ValueError(
                "Invalid evaluation evidence: sha(request['runtime_manifest']) == request['runtime_manifest_sha256']"
            )
        if not ([r["candidate"]["seed"] for r in rows] == request["seeds"]):
            raise ValueError(
                "Invalid evaluation evidence: [r['candidate']['seed'] for r in rows] == request['seeds']"
            )
        if not (set(request["seeds"]) <= set(self.protocol["seeds"])):
            raise ValueError(
                "Invalid evaluation evidence: set(request['seeds']) <= set(self.protocol['seeds'])"
            )
        if not (sha(request["manifest"]) == self.files[request["manifest"]]):
            raise ValueError(
                "Invalid evaluation evidence: sha(request['manifest']) == self.files[request['manifest']]"
            )
        if not (
            health["identity"]["package_digest"]
            == self.package["package_digest"]
            == self.protocol["package_digest"]
        ):
            raise ValueError(
                "Invalid evaluation evidence: health['identity']['package_digest'] == self.package['package_digest'] == self.protocol['package_digest']"
            )
        if not (
            self.protocol["scene_version"] == "tower_v4"
            and self.protocol["duration_s"] == request["duration"]
        ):
            raise ValueError(
                "Invalid evaluation evidence: self.protocol['scene_version'] == 'tower_v4' and self.protocol['duration_s'] == request['duration']"
            )
        if not (self.protocol["success_rule"] == self.success_rule):
            raise ValueError(
                "Invalid evaluation evidence: self.protocol['success_rule'] == self.success_rule"
            )
        if not (self.protocol["authorization"]["scene_selection"] == "Current TOWER v4 scene"):
            raise ValueError(
                "Invalid evaluation evidence: self.protocol['authorization']['scene_selection'] == 'Current TOWER v4 scene'"
            )
        if self.protocol.get("schema") == "tower-public-benchmark-protocol-unreviewed-v1":
            if self.protocol["ik"]["shared_spec_sha256"] != shared_spec_identity():
                raise ValueError("Protocol changes the canonical shared IK identity")
        else:
            review = read_bound(
                self.protocol["pilot_review"]["path"], self.protocol["pilot_review"]["sha256"]
            )
            if not (
                sha(self.protocol["pilot_review"]["path"])
                == self.protocol["pilot_review"]["sha256"]
            ):
                raise ValueError(
                    "Invalid evaluation evidence: sha(self.protocol['pilot_review']['path']) == self.protocol['pilot_review']['sha256']"
                )
            if not (
                review["passed"]
                and review["native_inference_qualified"]
                and (review["package_digest"] == self.package["package_digest"])
            ):
                raise ValueError(
                    "Invalid evaluation evidence: review['passed'] and review['native_inference_qualified'] and (review['package_digest'] == self.package['package_digest'])"
                )
            if not (review["scene_reference_review_passed"] and review["shared_ik_parity_passed"]):
                raise ValueError(
                    "Invalid evaluation evidence: review['scene_reference_review_passed'] and review['shared_ik_parity_passed']"
                )
            if not (
                self.protocol["ik"]["shared_spec_sha256"]
                == review["shared_spec_sha256"]
                == shared_spec_identity()
            ):
                raise ValueError(
                    "Invalid evaluation evidence: protocol and review must match the canonical shared IK identity"
                )
        from concurrent.futures import ThreadPoolExecutor

        def verify_bound_file(item):
            p, h = item
            if not (sha(p) == h):
                raise ValueError(("Runtime changed", p))
            return p

        with ThreadPoolExecutor(max_workers=8) as verifier:
            verified = list(verifier.map(verify_bound_file, self.files.items()))
        if not (len(verified) == len(self.files)):
            raise ValueError("Invalid evaluation evidence: len(verified) == len(self.files)")

    def verify_prediction(self, identity):
        if not (identity == self.health["identity"]):
            raise ValueError("Inference identity changed during evaluation")

    def closure(self, sources):
        if not (sources):
            raise ValueError("Invalid evaluation evidence: sources")
        if self.verification is not None:
            self.verification.verify()
            # Input/evidence bindings remain mandatory. Executables are bound by
            # the release/dependency manifests; the collector rejects frozen imports.
            for path, digest in self.files.items():
                if sha(path) != digest:
                    raise ValueError(("Runtime input changed", path))
            return self.verification.verify_sources(sources)
        from concurrent.futures import ThreadPoolExecutor

        def verify_source(item):
            p, h = item
            if not (self.files.get(p) == h and sha(p) == h):
                raise ValueError(("Unbound/changed runtime source", p))

        with ThreadPoolExecutor(max_workers=8) as verifier:
            list(verifier.map(verify_source, sources.items()))
        return sources

    def write_evidence(self, index, evidence, commands, camera, sources):
        output = self.rows[index]["output"]
        for n in ["episode/states.hdf5", "commands.json", "status.json"]:
            if (output / n).exists():
                raise ValueError("Invalid evaluation evidence: not (output / n).exists()")
        self.closure(sources)
        evidence.write(index, output)
        immutable(output / "commands.json", commands)
        immutable(
            output / "camera-readback.json",
            dict(camera, env_id=index, terminal_capture_scope="actual per-environment terminal"),
        )
        immutable(output / "runtime-sources-final.json", sources)
        return output

    def finalize_trial(
        self,
        index,
        evidence,
        commands,
        archive,
        camera_report,
        source_files,
        *,
        tick,
        failure=None,
        success=None,
    ):
        seed = self.rows[index]["candidate"]["seed"]
        if not (seed not in self.completed):
            raise ValueError("Invalid evaluation evidence: seed not in self.completed")
        if not (type(tick) is int and 0 < tick <= self.duration_ticks):
            raise ValueError(
                "Invalid evaluation evidence: type(tick) is int and 0 < tick <= self.duration_ticks"
            )
        rows = evidence.records[index]
        stamp = tick / 400
        if not (len(rows) > 1 and len(archive.times) > 1):
            raise ValueError(
                "Invalid evaluation evidence: len(rows) > 1 and len(archive.times) > 1"
            )
        if not (rows[-1]["time"] == float(archive.times[-1]) == stamp):
            raise ValueError(
                "Invalid evaluation evidence: rows[-1]['time'] == float(archive.times[-1]) == stamp"
            )
        if not ([r["time"] for r in rows] == [r["time"] for r in commands]):
            raise ValueError(
                "Invalid evaluation evidence: [r['time'] for r in rows] == [r['time'] for r in commands]"
            )
        if failure and success:
            raise ValueError("Invalid evaluation evidence: not (failure and success)")
        if success:
            if not (success["protocol"] == self.success_rule and success["terminal_tick"] == tick):
                raise ValueError(
                    "Invalid evaluation evidence: success['protocol'] == self.success_rule and success['terminal_tick'] == tick"
                )
        elif failure:
            failed = failure["failure_tick"]
            if not (0 <= failed <= tick and failure["rejected_action_applied"] is False):
                raise ValueError(
                    "Invalid evaluation evidence: 0 <= failed <= tick and failure['rejected_action_applied'] is False"
                )
            if not (tick == (20 if failed == 0 else failed)):
                raise ValueError(
                    "Invalid evaluation evidence: tick == (20 if failed == 0 else failed)"
                )
        else:
            if not (tick == self.duration_ticks):
                raise ValueError("Invalid evaluation evidence: tick == self.duration_ticks")
        output = self.write_evidence(index, evidence, commands, camera_report, source_files)
        scorer = self.scorer
        recording = scorer.recording(
            self.rows[index]["candidate"]["episode_spec"], rows, evidence.env.finger_geometry
        )
        if failure and self.task != "task1-2":
            recording.status["status"] = "task_failure"
        report = scorer.score(recording)
        if report.get("valid") is not True:
            raise ValueError("Scorer rejected physical recording evidence")
        if success and not (
            report["scores"]["s"] == 1 and report.get("success_time_s", stamp) == stamp
        ):
            raise ValueError("Offline success/time disagrees with online detector")
        final_state_revision = self.task.startswith(("task4-", "task6-", "task7-"))
        guard_terminal = bool(failure and failure.get("category") == "camera_near_plane_guard")
        if (
            not success
            and not final_state_revision
            and not guard_terminal
            and report["scores"]["s"] != 0
        ):
            raise ValueError("Offline success disagrees with online detector")
        scored_success = bool(report["scores"]["s"])
        immutable(output / "score.json", report)
        names = [
            "episode_spec.json",
            "episode/states.hdf5",
            "commands.json",
            "finger-link-poses.npz",
            "finger-feedback.npz",
            "finger_geometry.json",
            "score.json",
        ]
        if success:
            immutable(output / "task-success.json", dict(success, report=report))
        if failure:
            immutable(output / "policy-action-failure.json", failure)
        if failure and failure.get("camera_guard"):
            immutable(output / "camera-near-plane-failure.json", failure["camera_guard"])
            names += ["camera-near-plane-failure.json"]
        self.closed_archives.add(index)
        archive.close(complete=True)
        names += ["camera-frames.hdf5"]
        status = dict(
            schema="tower-v4-standalone-policy-trial-v1",
            status="task_success" if success else "task_failure" if failure else "duration_reached",
            process_status="completed",
            valid_trial=True,
            evidence_complete=True,
            seed=seed,
            terminal_tick=tick,
            simulated_seconds=stamp,
            terminal_reason="task_success"
            if success
            else "camera_near_plane_guard"
            if guard_terminal
            else "policy_action_failure"
            if failure
            else "duration_reached",
            task_success=scored_success,
            scoring_revision=report.get("scoring_revision", report["protocol"]),
            scored_outcome="success" if scored_success else "failure",
            checkpoint_sha256=self.package["checkpoint_sha256"],
            package_digest=self.package["package_digest"],
            protocol_sha256=self.request["protocol_sha256"],
            execution_adapter="tower_sim_candidate_v4",
            profiling_only=bool(self.request.get("baseline_profile")),
            official_episode_credit=False,
            returncode=0,
            source_hashes={n: sha(output / n) for n in names},
        )
        if self.verification is not None:
            status.update(self.verification.provenance)
        immutable(output / "status.json", status)
        from tower_eval.posture_flags import flags

        immutable(
            output / "posture-flags.json",
            flags(output / "episode/states.hdf5", output / "score.json"),
        )
        self.completed[seed] = dict(
            path=str(output / "status.json"), sha256=sha(output / "status.json")
        )

    def invalidate_unfinished(
        self, index, evidence, commands, archive, camera_report, source_files, *, reason
    ):
        seed = self.rows[index]["candidate"]["seed"]
        if seed in self.completed:
            b = self.completed[seed]
            if not (sha(b["path"]) == b["sha256"]):
                raise ValueError("Invalid evaluation evidence: sha(b['path']) == b['sha256']")
            return
        output = self.rows[index]["output"]
        errors = []
        try:
            if not (output / "episode/states.hdf5").exists():
                self.write_evidence(index, evidence, commands, camera_report, source_files)
        except BaseException as e:  # noqa: BLE001 -- retain partial evidence
            errors.append(repr(e))
        if index not in self.closed_archives:
            self.closed_archives.add(index)
            try:
                archive.close(complete=False)
            except BaseException as e:  # noqa: BLE001 -- retain partial evidence
                errors.append(repr(e))
        if not (output / "status.json").exists():
            immutable(
                output / "status.json",
                dict(
                    schema="tower-v4-standalone-policy-trial-v1",
                    status="infrastructure_failure",
                    valid_trial=False,
                    evidence_complete=False,
                    seed=seed,
                    reason=reason,
                    errors=errors,
                    **(self.verification.provenance if self.verification is not None else {}),
                ),
            )
