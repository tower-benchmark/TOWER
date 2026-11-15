"""JSON-lines adapter for nero-kinematics posture-IPOPT in its own interpreter."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from typing import Any, Protocol

import numpy as np
from nero_kinematics.profiles import load_profile

SCHEMA = "tower-bench-ik-v1"


def service_identity():
    """Build and loaded profile bytes, also for source-only PYTHONPATH installs."""

    def package_version(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return importlib.import_module(name.replace("-", "_")).__version__

    command = load_profile("command")
    return dict(
        packages={name: package_version(name) for name in ("tower-sim", "nero-kinematics")},
        command_profile_id=command["name"],
        command_source_sha256=command["source_sha256"],
        solver_profile_id=load_profile("posture-ipopt")["name"],
        profile_sha256={
            name: hashlib.sha256(
                files("nero_kinematics.profiles").joinpath(name).read_bytes()
            ).hexdigest()
            for name in ("command.json", "posture-ipopt.json")
        },
    )


class EncoderSolver(Protocol):
    identity: dict[str, Any]

    def encoder(self, target: Any, seed: Any, arm: int) -> dict[str, Any]: ...


def validate_request(request):
    if request.get("schema") != SCHEMA or not isinstance(request.get("request_id"), str):
        raise ValueError("Invalid IK protocol/request ID")
    poses, seed = np.asarray(request["poses"], float), np.asarray(request["seed"], float)
    if (
        poses.ndim != 4
        or poses.shape[1:] != (2, 4, 4)
        or not 1 <= len(poses) <= 32
        or seed.shape != (2, 7)
        or request["dt_s"] != 0.1
    ):
        raise ValueError("Invalid IK request")
    if not np.isfinite(poses).all() or not np.isfinite(seed).all():
        raise ValueError("Nonfinite IK request")
    for transform in poses.reshape(-1, 4, 4):
        if (
            not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or np.linalg.det(transform[:3, :3]) < 0.99999
        ):
            raise ValueError("Invalid transform")
    return poses, seed.copy()


def solve_request(request, solver: EncoderSolver):
    """Attempt each supplied target once; rejection aborts the entire chunk."""
    poses, seed = validate_request(request)
    rows, diagnostics = [], []
    accepted = True
    for index, pair in enumerate(poses):
        joints = []
        for arm in (0, 1):
            result = solver.encoder(pair[arm], seed[arm], arm)
            diagnostics.append(
                dict(row=index, arm=arm, variant="posture_ipopt_v5_shared", **result)
            )
            if not result["accepted"]:
                accepted = False
                break
            joints.append(result["joints"])
        if not accepted:
            break
        seed = np.asarray(joints)
        rows.append(joints)
    return dict(
        schema=SCHEMA,
        request_id=request["request_id"],
        accepted=accepted,
        joints=rows if accepted else None,
        diagnostics=diagnostics,
        identity=solver.identity,
        service_identity=service_identity(),
    )


def main():
    # Save the protocol pipe before redirecting fd 1. Python-only redirection
    # does not catch C/C++ diagnostics from native imports and IPOPT.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as protocol:
        os.dup2(sys.stderr.fileno(), 1)
        with contextlib.redirect_stdout(sys.stderr):
            serve(protocol)


def serve(protocol):
    # All native imports happen after both Python and fd-level redirection.
    from nero_kinematics.solvers.shared import SharedIK

    solver = SharedIK()
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            result = solve_request(request, solver)
        except Exception as exc:  # noqa: BLE001 -- one error envelope per request
            result = dict(
                schema=SCHEMA,
                request_id=request.get("request_id") if isinstance(request, dict) else None,
                service_error=repr(exc),
            )
        protocol.write(json.dumps(result, allow_nan=False) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
