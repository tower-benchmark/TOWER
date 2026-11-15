"""Owned subprocess transport; no environment lookup, global process or IK fallback."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ik_service import SCHEMA, service_identity, validate_request


def _subprocess_environment(environment):
    """The only environment-read boundary; control choices never come from it."""
    result = dict(os.environ if environment is None else environment)
    result.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return result


def _write_all(stream, payload):
    remaining = memoryview(payload)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise RuntimeError("IK service closed input transport")
        remaining = remaining[written:]
    stream.flush()


@dataclass(frozen=True)
class IKServiceOptions:
    python: Path
    response_timeout_s: float = 120.0

    def __post_init__(self):
        if not np.isfinite(self.response_timeout_s) or self.response_timeout_s <= 0:
            raise ValueError("IK response timeout must be finite and positive")


class IKServiceClient:
    """One serial request stream per instance. Use as a context manager or call close.

    Both wheels must be installed in the selected IK interpreter. The environment
    is inherited only at this subprocess boundary; BLAS is always single-threaded.
    """

    def __init__(self, options: IKServiceOptions, *, environment: Mapping[str, str] | None = None):
        self.options = options
        self.environment = None if environment is None else dict(environment)
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._sequence = 0
        self._expected_identity = service_identity()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._lock:
            process, self._process = self._process, None
            if process is None:
                return
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()

    def _receive(self):
        process = self._process
        assert process is not None and process.stdout is not None
        deadline = time.monotonic() + self.options.response_timeout_s
        data = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b"\n" not in data:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(timeout=remaining):
                    raise RuntimeError("IK service response timeout")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("IK service closed transport")
                data.extend(chunk)
                if len(data) > 4 * 1024 * 1024:
                    raise RuntimeError("IK service response exceeds frame limit")
        line, extra = bytes(data).split(b"\n", 1)
        if extra:
            raise RuntimeError("IK service sent an unsolicited response")
        return json.loads(line)

    def solve(self, poses, seed, dt_s=0.1):
        with self._lock:
            self._sequence += 1
            request = dict(
                schema=SCHEMA,
                request_id=str(self._sequence),
                poses=np.asarray(poses).tolist(),
                seed=np.asarray(seed).tolist(),
                dt_s=dt_s,
            )
            validate_request(request)
            payload = (json.dumps(request, allow_nan=False) + "\n").encode()
            if self._process is None:
                self._process = subprocess.Popen(
                    [str(self.options.python), "-u", "-m", "tower_sim.bench.ik_service"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    env=_subprocess_environment(self.environment),
                    bufsize=0,
                )
            try:
                if self._process.poll() is not None:
                    raise RuntimeError("IK service exited")
                assert self._process.stdin is not None
                _write_all(self._process.stdin, payload)
                result = self._receive()
                # Parse errors may have no request ID or service identity. Keep
                # their actual error instead of masking it as an identity failure.
                if isinstance(result, dict) and "service_error" in result:
                    if result.get("request_id") not in (None, request["request_id"]):
                        raise RuntimeError("IK service response identity mismatch")
                    raise RuntimeError(result["service_error"])
                if (
                    not isinstance(result, dict)
                    or result.get("schema") != SCHEMA
                    or result.get("request_id") != request["request_id"]
                    or result.get("service_identity") != self._expected_identity
                ):
                    raise RuntimeError("IK service response identity mismatch")
                if type(result.get("accepted")) is not bool or not isinstance(
                    result.get("diagnostics"), list
                ):
                    raise RuntimeError("Invalid IK service result")
                if result["accepted"]:
                    joints = np.asarray(result.get("joints"), float)
                    if joints.shape != (len(poses), 2, 7) or not np.isfinite(joints).all():
                        raise RuntimeError("Invalid IK service joints")
                elif result.get("joints") is not None:
                    raise RuntimeError("Rejected IK response contains executable joints")
                return result
            except (OSError, ValueError, RuntimeError, TypeError):
                # Discard a broken channel; never replay this request or change its target.
                self.close()
                raise
