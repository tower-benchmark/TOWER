"""Content-addressed online computations, explicitly injected into canonical detectors.

Every value comes from the reference implementation. No module/class patching,
process globals, quantization or scoring rules live here. The runner shares one cache across its online scorers per process, matching
the frozen capacity. Final scoring always uses reference computation.
"""

import dataclasses
import json
from typing import Any, cast

import numpy as np
from tower_eval.scoring.computation import REFERENCE, ReferenceComputation


class CachedComputation(ReferenceComputation):
    def __init__(
        self,
        *,
        reference=REFERENCE,
        audit_samples=4,
        full_audit_every=1024,
        angular_audit_every=512,
        max_entries=1_000_000,
    ):
        if any(
            type(v) is not int or v <= 0
            for v in (full_audit_every, angular_audit_every, max_entries)
        ):
            raise ValueError("Cache audit periods and capacity must be positive integers")
        if type(audit_samples) is not int or audit_samples < 0:
            raise ValueError("audit_samples must be a nonnegative integer")
        self.reference = reference
        self.audit_samples = audit_samples
        self.full_audit_every = full_audit_every
        self.angular_audit_every = angular_audit_every
        self.max_entries = max_entries
        self.cache = {}
        self.angles = {}
        self.stats = dict(
            calls=0,
            candidate_samples=0,
            hits=0,
            computed=0,
            audited_samples=0,
            full_audits=0,
            fallback=0,
            cache_clears=0,
            entries=0,
            angular_hits=0,
            angular_computed=0,
            angular_audited=0,
        )

    @staticmethod
    def sub(r, idx):
        return dataclasses.replace(
            r, times=r.times[idx], poses=r.poses[idx], velocities=r.velocities[idx]
        )

    def support_mask(self, r, target, supporters, dimensions, candidate, rules):
        self.stats["calls"] += 1
        n = len(r.times)
        candidate = np.asarray(candidate)
        blocks = [target, *supporters]
        if (
            candidate.dtype != np.bool_
            or candidate.shape != (n,)
            or not isinstance(r.poses, np.ndarray)
            or r.poses.dtype not in (np.float32, np.float64)
            or r.poses.ndim != 3
            or r.poses.shape[0] != n
            or r.poses.shape[2] != 7
            or not isinstance(r.velocities, np.ndarray)
            or r.velocities.dtype != r.poses.dtype
            or r.velocities.ndim != 3
            or r.velocities.shape[0] != n
            or r.velocities.shape[2] != 6
            or not all(
                isinstance(j, (int, np.integer)) and 0 <= j < r.poses.shape[1] for j in blocks
            )
        ):
            self.stats["fallback"] += 1
            return self.reference.support_mask(r, target, supporters, dimensions, candidate, rules)
        static = json.dumps(
            [
                int(target),
                [int(j) for j in supporters],
                [np.asarray(dimensions[j], dtype=float).tolist() for j in blocks],
                dataclasses.asdict(rules),
                cast(Any, r.poses.dtype).str,
            ],
            sort_keys=True,
        )
        table = self.cache.setdefault(static, {})
        result = np.zeros(n, dtype=bool)
        idx = np.flatnonzero(candidate)
        self.stats["candidate_samples"] += len(idx)
        if len(idx):
            rows = np.ascontiguousarray(
                np.concatenate(
                    (r.poses[idx][:, blocks].reshape(len(idx), -1), r.velocities[idx, target]),
                    axis=1,
                )
            )
            width = rows.shape[1] * rows.itemsize
            raw = rows.tobytes()
            keys = [raw[p * width : (p + 1) * width] for p in range(len(idx))]
            values = [table.get(key) for key in keys]
            missing = [p for p, v in enumerate(values) if v is None]
            hits = [p for p, v in enumerate(values) if v is not None]
            if missing:
                computed = self.reference.support_mask(
                    self.sub(r, idx[missing]),
                    target,
                    supporters,
                    dimensions,
                    np.ones(len(missing), dtype=bool),
                    rules,
                )
                if self.stats["entries"] + len(missing) > self.max_entries:
                    for other in self.cache.values():
                        other.clear()
                    self.stats["entries"] = 0
                    self.stats["cache_clears"] += 1
                for p, v in zip(missing, computed):
                    v = bool(v)
                    values[p] = v
                    table[keys[p]] = v
                self.stats["entries"] += len(missing)
                self.stats["computed"] += len(missing)
            self.stats["hits"] += len(hits)
            if hits and self.audit_samples:
                chosen = [
                    hits[(k * len(hits)) // self.audit_samples]
                    for k in range(min(self.audit_samples, len(hits)))
                ]
                check = self.reference.support_mask(
                    self.sub(r, idx[chosen]),
                    target,
                    supporters,
                    dimensions,
                    np.ones(len(chosen), dtype=bool),
                    rules,
                )
                self.stats["audited_samples"] += len(chosen)
                if [bool(v) for v in check] != [values[p] for p in chosen]:
                    raise RuntimeError("fast_early_stop sample audit mismatch")
            result[idx] = values
        if self.stats["calls"] % self.full_audit_every == 0:
            self.stats["full_audits"] += 1
            if not np.array_equal(
                self.reference.support_mask(r, target, supporters, dimensions, candidate, rules),
                result,
            ):
                raise RuntimeError("fast_early_stop full audit mismatch")
        return result

    def angular_error(self, q, reference):
        # stable_dwell calls this once per sample with two single quaternions. The result is a pure
        # function of their exact values, so it is cached by dtype and bytes. Other call shapes go to the original.
        if (
            type(q) is not np.ndarray
            or type(reference) is not np.ndarray
            or q.shape != (4,)
            or reference.shape != (4,)
            or q.dtype not in (np.float32, np.float64)
            or reference.dtype != q.dtype
        ):
            return self.reference.angular_error(q, reference)
        key = cast(Any, q.dtype).str.encode() + q.tobytes() + reference.tobytes()
        value = self.angles.get(key)
        if value is None:
            value = self.reference.angular_error(q, reference)
            if len(self.angles) >= self.max_entries:
                self.angles.clear()
                self.stats["cache_clears"] += 1
            self.angles[key] = value
            self.stats["angular_computed"] += 1
        else:
            self.stats["angular_hits"] += 1
            if self.stats["angular_hits"] % self.angular_audit_every == 0:
                self.stats["angular_audited"] += 1
                check = self.reference.angular_error(q, reference)
                if type(check) is not type(value) or check.tobytes() != value.tobytes():
                    raise RuntimeError("fast_early_stop angular audit mismatch")
        return value
