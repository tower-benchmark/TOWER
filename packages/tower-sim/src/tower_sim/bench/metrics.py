"""Injected event sink, request IDs and optional profiling; no import-time I/O."""

from __future__ import annotations

import collections
import contextlib
import functools
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class MetricsSink(Protocol):
    def emit(self, record: dict[str, Any]) -> None: ...
    def flush(self) -> None: ...


class Profiler(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def range_push(self, name: str) -> None: ...
    def range_pop(self) -> None: ...


@dataclass(frozen=True)
class TrialRequestID:
    trial: str

    def __call__(self, seed: int, tick: int) -> str:
        return f"{self.trial}:{seed}:{tick}"


class Metrics:
    def __init__(
        self,
        sink: MetricsSink,
        request_id: Callable[[int, int], str],
        *,
        profiler: Profiler | None = None,
        trace: bool = False,
        epoch_ns=time.time_ns,
        monotonic_ns=time.monotonic_ns,
    ):
        if trace and profiler is None:
            raise ValueError("Tracing requires an explicitly supplied profiler")
        self.sink, self.request_id, self.profiler = sink, request_id, profiler
        self.trace, self.epoch_ns, self.monotonic_ns = trace, epoch_ns, monotonic_ns
        self.lock = threading.RLock()
        self.stats = collections.defaultdict(lambda: [0.0, 0])
        self.tick_value = 0
        self.active = self.finished = self.closed = False
        self.loop_start: int | None = None

    def emit(self, event, **fields):
        with self.lock:
            record = dict(
                event=event,
                tick=self.tick_value,
                sim_s=self.tick_value / 400,
                epoch_ns=self.epoch_ns(),
                mono_ns=self.monotonic_ns(),
                **fields,
            )
            # Preserve the frozen JSON sink's strict serialization contract even
            # when the caller supplies an in-memory or permissive sink.
            json.dumps(record, allow_nan=False)
            self.sink.emit(record)

    def tick(self, tick):
        self.tick_value = tick
        if self.loop_start is None:
            self.loop_start = self.monotonic_ns()
            self.emit("simulation_loop_start")
        if self.trace and tick >= 1280 and not self.active and not self.finished:
            assert self.profiler is not None
            self.profiler.start()
            self.active = True
            self.emit("nsight_start")
        if self.active and tick >= 2560:
            assert self.profiler is not None
            self.profiler.stop()
            self.active = False
            self.finished = True
            self.emit("nsight_stop")
        if tick % 640 == 0:
            self.flush()

    @contextlib.contextmanager
    def span(self, name):
        started = self.monotonic_ns()
        if self.active and self.profiler is not None:
            self.profiler.range_push(name)
        try:
            yield
        finally:
            elapsed = (self.monotonic_ns() - started) / 1e9
            with self.lock:
                self.stats[name][0] += elapsed
                self.stats[name][1] += 1
            if self.active and self.profiler is not None:
                self.profiler.range_pop()

    def call(self, name, fn, *args, **kwargs):
        with self.span(name):
            return fn(*args, **kwargs)

    def decorated(self, name):
        def wrap(fn):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                return self.call(name, fn, *args, **kwargs)

            return wrapped

        return wrap

    def headers(self, seed, tick):
        capture = (
            ("start" if tick == 1280 else "stop" if tick == 2560 else "") if self.trace else ""
        )
        return {
            "Content-Type": "application/json",
            "X-Baseline-Request-ID": self.request_id(seed, tick),
            "X-Baseline-Capture": capture,
        }

    def barrier(self, read_state, assert_unchanged, *, wait_for_release=None):
        """Wait without advancing physics; no callback means no barrier.

        After ready_for_release is emitted and flushed, wait_for_release must
        publish the runner's READY_FOR_RELEASE marker (including any cohort ID)
        and block until its matching release arrives. The caller owns that I/O,
        polling and timeout policy, and supplies a full physical-state snapshot
        and unchanged-state assertion. Metrics performs no file/environment I/O.
        """
        if wait_for_release is None:
            return
        before = read_state()
        self.emit("ready_for_release")
        self.flush()
        with self.span("startup.barrier_wait"):
            wait_for_release()
        assert_unchanged(before, read_state())
        self.emit("released_with_physics_unchanged")
        self.flush()

    def predict(self, adapter, payload, *, headers):
        """Measure the per-environment WebSocket exchange before response validation."""
        request_id = headers["X-Baseline-Request-ID"]
        started = None
        timer = contextlib.ExitStack()

        def observe(event, **fields):
            nonlocal started
            if event == "rpc_start":
                started = self.monotonic_ns()
                self.emit("rpc_start", request_id=request_id)
                timer.enter_context(self.span("policy.rpc_elapsed_sum"))
                return
            timer.close()
            rpc_s = (self.monotonic_ns() - started) / 1e9
            self.emit("rpc_end", request_id=request_id, rpc_s=rpc_s, **fields)

        with timer:
            return adapter.predict(payload, headers=headers, rpc_observer=observe)

    def flush(self):
        self.emit("stage_totals", totals=dict(self.stats))
        self.sink.flush()

    def end_loop(self):
        if self.loop_start is not None:
            self.emit(
                "simulation_loop_end", loop_wall_s=(self.monotonic_ns() - self.loop_start) / 1e9
            )
        self.flush()

    def close(self):
        if self.closed:
            return
        if self.active and self.profiler is not None:
            self.profiler.stop()
            self.active = False
        self.flush()
        self.closed = True
        # The injected sink is owned (and closed) by its caller.
