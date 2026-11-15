"""The opt-in desktop rule admits only an idle, fully inventoried GPU."""

import json

import pytest

from scripts.maintainer import gpu_admission as admission
from scripts.maintainer import gpu_baseline


def xml(*, memory: int | str = 564, utilization: int | str = 0, clients=None):
    if clients is None:
        clients = [
            (name, "C+G" if name == admission.COMPUTE_PATH else "G")
            for name in sorted(admission.DESKTOP_PATHS)
        ]
    processes = "".join(
        f"<process_info><pid>{i + 1}</pid><process_name>{name}</process_name>"
        f"<type>{kind}</type></process_info>"
        for i, (name, kind) in enumerate(clients)
    )
    return (
        "<nvidia_smi_log><gpu><fb_memory_usage>"
        f"<used>{memory} MiB</used></fb_memory_usage>"
        f"<utilization><gpu_util>{utilization} %</gpu_util></utilization>"
        f"<processes>{processes}</processes></gpu></nvidia_smi_log>"
    )


@pytest.mark.parametrize("memory,utilization", [(564, 0), (1023, 5)])
def test_desktop_inventory_accepts_only_explicit_limits(memory, utilization):
    row = admission.desktop_sample(xml(memory=memory, utilization=utilization))
    assert row["memory_mib"] == memory and len(row["processes"]) == 4


@pytest.mark.parametrize(
    "body",
    [
        xml(memory=1024),
        xml(utilization=6),
        xml(memory="N/A"),
        xml(utilization="N/A"),
        xml(clients=[("/" + "tmp/gnome-shell", "G")]),
        xml(clients=[("/usr/bin/python", "C")]),
        xml(clients=[("/usr/lib/xorg/Xorg", "C+G")]),
        xml(clients=[(admission.COMPUTE_PATH, "unknown")]),
        xml().replace("<processes>", "<missing>").replace("</processes>", "</missing>"),
        xml().replace("<pid>1</pid>", "<pid>N/A</pid>"),
        "<nvidia_smi_log/>",
        xml().replace("</gpu>", "</gpu><gpu/>"),
    ],
)
def test_desktop_inventory_rejects_unknown_or_busy(body):
    with pytest.raises(ValueError):
        admission.desktop_sample(body)


def clock_and_gpu(monkeypatch, utilizations):
    clock = [0.0]
    monkeypatch.setattr(admission.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        admission.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    samples = iter(utilizations)
    commands = []

    def snapshot(command, **kwargs):
        commands.append(command)
        return xml(utilization=next(samples))

    monkeypatch.setattr(admission.subprocess, "check_output", snapshot)
    return commands


def test_sampled_desktop_admission_records_every_xml(monkeypatch, tmp_path):
    commands = clock_and_gpu(monkeypatch, [0] * 6 + [2, 0, 5, 0])
    path = tmp_path / "admission.json"
    result = admission.admit_desktop(0, path)
    assert result["schema"] == "tower-desktop-gpu-admission-v2"
    assert result["admitted"] and result["sample_span_s"] >= 10
    assert len(result["samples"]) == 10 and result["median_utilization_percent"] == 0
    assert all("<processes>" in row["xml"] for row in result["samples"])
    assert commands == [["nvidia-smi", "--id=0", "-q", "-x"]] * 10
    assert json.loads(path.read_text()) == result
    with pytest.raises(FileExistsError):
        admission.admit_desktop(0, path)


@pytest.mark.parametrize("values,count", [([1] * 10, 10), ([0, 0, 6], 3)])
def test_failed_window_is_recorded(monkeypatch, tmp_path, values, count):
    clock_and_gpu(monkeypatch, values)
    path = tmp_path / "rejected.json"
    with pytest.raises(ValueError):
        admission.admit_desktop(0, path)
    result = json.loads(path.read_text())
    assert not result["admitted"] and len(result["samples"]) == count
    assert all("xml" in row for row in result["samples"])


def test_default_admission_keeps_original_512_and_zero(monkeypatch):
    monkeypatch.setattr(gpu_baseline.subprocess, "check_output", lambda command, **kw: "511, 0")
    gpu_baseline.gpu_idle(0, False)
    for value in ("512, 0", "511, 1"):
        monkeypatch.setattr(
            gpu_baseline.subprocess,
            "check_output",
            lambda command, value=value, **kw: xml() if "-x" in command else value,
        )
        with pytest.raises(RuntimeError):
            gpu_baseline.gpu_idle(0, False)
        with pytest.raises(RuntimeError):
            gpu_baseline.gpu_idle(0, True)


def test_policy_desktop_option_reaches_admission(monkeypatch, tmp_path):
    from scripts.maintainer import candidate_policy

    path = tmp_path / "request.json"
    path.write_text(json.dumps(dict(gpu=0, output=str(tmp_path / "output"))))
    monkeypatch.setenv("TOWER_GPU_LOCK", str(tmp_path / "gpu.lock"))
    called = []

    def reject(gpu, receipt):
        called.append((gpu, receipt))
        raise ValueError("desktop inventory rejected")

    monkeypatch.setattr(admission, "admit_desktop", reject)
    with pytest.raises(ValueError, match="desktop inventory rejected"):
        candidate_policy.main(
            [
                "run",
                "--request",
                str(path),
                "--ik-python",
                "unused",
                "--diagnostic",
                "--allow-desktop-gpu",
            ]
        )
    assert called == [(0, path.with_suffix(".admission.json"))]
