"""Verify that a Replicator step actually delivered its scheduled frame.

Pinned Replicator can log a render timeout and return normally. A Python return
or our own step counter is therefore not delivery evidence. This helper reads
its real SDG dispatcher and scheduled timestamp without pumping or advancing the
simulator. Initialization is explicitly unverified when the graph is warming up.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import math
import numbers
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

DISPATCHER = "/Render/PostProcess/SDGPipeline/PostProcessDispatcher"


def snapshot_dispatcher() -> dict[str, Any]:
    import omni.graph.core as og  # pyright: ignore[reportMissingImports]

    result: dict[str, Any] = dict(
        exists=False, compute_count=None, reference_numerator=None, reference_denominator=None
    )
    try:
        node = og.get_node_by_path(DISPATCHER)
        if node is None or not node.is_valid():
            return result
        result.update(
            exists=True,
            compute_count=int(node.get_compute_count()),
            reference_numerator=node.get_attribute("outputs:referenceTimeNumerator").get(),
            reference_denominator=node.get_attribute("outputs:referenceTimeDenominator").get(),
        )
        for name in ("reference_numerator", "reference_denominator"):
            if isinstance(result[name], numbers.Integral):
                result[name] = int(result[name])
    except Exception as exc:  # noqa: BLE001 -- latch native callback/delivery failures
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def source_time_epsilon(source):
    # The pinned SDK defines _TIME_EPS locally inside step(), not as a module
    # constant. Extract that exact literal; reject ambiguous or changed code.
    return _source_time_epsilon(Path(source).read_text())


@lru_cache(maxsize=1)
def _source_time_epsilon(text):
    # Frozen rule, verbatim: exactly one _TIME_EPS assignment anywhere inside the
    # top-level step() functions, equal to 1e-7. The pinned SDK also assigns it in
    # step_async(); that, and module-level code, are deliberately not consulted.
    tree = ast.parse(text)
    steps = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "step"
    ]
    values = [
        ast.literal_eval(node.value)
        for function in steps
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_TIME_EPS" for target in node.targets
        )
    ]
    if len(values) != 1 or type(values[0]) not in (int, float) or values[0] != 1e-7:
        raise RuntimeError("Pinned Replicator step time tolerance changed or is ambiguous")
    return values[0]


def scheduled_target() -> dict[str, Any]:
    module = importlib.import_module("omni.replicator.core.scripts.orchestrator")
    if module.__file__ is None:
        raise RuntimeError("Pinned Replicator source is unavailable")
    source = Path(module.__file__).resolve()
    result: dict[str, Any] = dict(
        available=False,
        source_path=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        tolerance_s=source_time_epsilon(source),
    )
    stamps = module._orchestrator._sim_times_to_write
    if stamps:
        numerator, denominator = stamps[-1]
        result.update(available=True, numerator=int(numerator), denominator=int(denominator))
    return result


def _rational(numerator, denominator):
    if (
        not isinstance(numerator, numbers.Integral)
        or isinstance(numerator, bool)
        or not isinstance(denominator, numbers.Integral)
        or isinstance(denominator, bool)
        or denominator <= 0
    ):
        raise RuntimeError("Renderer reference/scheduled timestamp is not a finite rational")
    stamp = cast(float, numerator / denominator)
    if not math.isfinite(stamp):
        raise RuntimeError("Nonfinite renderer timestamp")
    return stamp


def verified_step(step, *, initializing=False):
    """Call the original step once; raise before callers increment capture counts.

    No additional render, app update, retry or physics operation is performed.
    The caller must retain this receipt for successful observation delivery.
    """
    before = snapshot_dispatcher()
    step()
    after = snapshot_dispatcher()
    target: dict[str, Any]
    try:
        target = scheduled_target()
    except Exception as exc:  # noqa: BLE001 -- latch native callback/delivery failures
        target = dict(available=False, error=f"{type(exc).__name__}: {exc}")
    receipt = dict(
        schema="tower-v3-render-delivery-v1",
        verified=False,
        initializing=bool(initializing),
        before=before,
        after=after,
        scheduled=target,
    )
    if initializing:
        return dict(receipt, reason="Initialization warmup; not observation delivery evidence")
    if (
        before.get("exists") is not True
        or after.get("exists") is not True
        or before.get("error")
        or after.get("error")
    ):
        raise RuntimeError(f"Renderer dispatcher unavailable: {receipt}")
    counts = [before.get("compute_count"), after.get("compute_count")]
    if any(type(count) is not int or count < 0 for count in counts) or cast(int, counts[1]) <= cast(
        int, counts[0]
    ):
        raise RuntimeError(f"Renderer dispatcher did not advance: {receipt}")
    if target.get("available") is not True:
        raise RuntimeError(f"Renderer scheduled timestamp unavailable: {receipt}")
    tolerance = cast(float, target.get("tolerance_s"))
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise RuntimeError(f"Invalid pinned renderer time tolerance: {receipt}")
    actual = _rational(after["reference_numerator"], after["reference_denominator"])
    expected = _rational(target["numerator"], target["denominator"])
    if actual < expected - tolerance:
        raise RuntimeError(f"Renderer returned before scheduled frame delivery: {receipt}")
    return dict(
        receipt,
        verified=True,
        delivered_time_s=actual,
        scheduled_time_s=expected,
        reason="Actual dispatcher advanced and reached the scheduled timestamp",
    )
