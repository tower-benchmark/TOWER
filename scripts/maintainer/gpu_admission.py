"""Opt-in admission for the owner-approved four-client desktop GPU; never runtime policy."""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

DESKTOP_PATHS = frozenset(
    (
        "/usr/lib/xorg/Xorg",
        "/usr/bin/gnome-shell",
        "/usr/libexec/gnome-remote-desktop-daemon",
        "/usr/share/FlClash/FlClash",
    )
)
COMPUTE_PATH = "/usr/libexec/gnome-remote-desktop-daemon"


def desktop_sample(xml: str) -> dict[str, Any]:
    """Fail closed on missing metrics, process lists, paths or process types."""
    gpus = ET.fromstring(xml).findall("gpu")
    if len(gpus) != 1:
        raise ValueError("Admission must describe exactly one selected GPU")
    gpu = gpus[0]

    def metric(path: str, unit: str) -> int:
        value = gpu.findtext(path, "")
        match = re.fullmatch(r"(\d+) " + unit, value)
        if not match:
            raise ValueError(f"Missing/invalid GPU metric: {path}")
        return int(match[1])

    processes = gpu.find("processes")
    if processes is None or any(p.tag != "process_info" for p in processes):
        raise ValueError("Missing/unknown GPU process inventory")
    clients = []
    for process in processes:
        name, kind, pid = (process.findtext(key, "") for key in ("process_name", "type", "pid"))
        if (
            name not in DESKTOP_PATHS
            or kind not in ("G", "C", "C+G")
            or ("C" in kind and name != COMPUTE_PATH)
            or not pid.isdecimal()
        ):
            raise ValueError(f"Unapproved GPU client: {name!r} ({kind!r})")
        clients.append(dict(path=name, type=kind, pid=int(pid)))
    memory = metric("fb_memory_usage/used", "MiB")
    utilization = metric("utilization/gpu_util", "%")
    if memory >= 1024 or utilization > 5:
        raise ValueError("Desktop GPU exceeds memory/utilization admission limits")
    return dict(memory_mib=memory, utilization_percent=utilization, processes=clients)


def admit_desktop(gpu: int, receipt: Path) -> dict[str, Any]:
    """Ten XML snapshots spanning at least ten seconds; retain even failed samples."""
    document: dict[str, Any] = dict(
        schema="tower-desktop-gpu-admission-v2",
        gpu=gpu,
        allowlist=sorted(DESKTOP_PATHS),
        compute_path=COMPUTE_PATH,
        limits=dict(memory_mib_exclusive=1024, max_utilization_percent=5, median_percent=0),
        samples=[],
        admitted=False,
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive publication prevents a later admission attempt replacing evidence.
    with receipt.open("x") as stream:
        try:
            start = time.monotonic()
            for index in range(10):
                if index:
                    time.sleep(max(0.0, start + index * 1.12 - time.monotonic()))
                sample: dict[str, Any] = dict(time_unix=time.time(), monotonic_s=time.monotonic())
                document["samples"].append(sample)
                sample["xml"] = subprocess.check_output(
                    ["nvidia-smi", f"--id={gpu}", "-q", "-x"], text=True
                )
                sample.update(desktop_sample(sample["xml"]))
            samples = document["samples"]
            document["sample_span_s"] = samples[-1]["monotonic_s"] - samples[0]["monotonic_s"]
            document["median_utilization_percent"] = statistics.median(
                row["utilization_percent"] for row in samples
            )
            if document["sample_span_s"] < 10 or document["median_utilization_percent"] != 0:
                raise ValueError("Desktop GPU is not idle across the sampled window")
            document["admitted"] = True
        finally:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return document
