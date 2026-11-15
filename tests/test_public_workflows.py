"""Public workflows cannot expose GPU runners or mutable action dependencies."""

import copy
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def check_workflow(document):
    assert document.get("permissions") == {"contents": "read"}
    for job in document["jobs"].values():
        assert isinstance(job.get("runs-on"), str)
        assert job["runs-on"] in {"ubuntu-latest", "ubuntu-24.04"}
        for entry in [job, *job.get("steps", [])]:
            if "uses" in entry:
                assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", entry["uses"])
                if entry["uses"].startswith("actions/checkout@"):
                    assert entry.get("with", {}).get("persist-credentials") is False


@pytest.mark.parametrize(
    "path",
    sorted(p for p in (ROOT / ".github/workflows").iterdir() if p.suffix in {".yml", ".yaml"}),
    ids=lambda p: p.name,
)
def test_public_workflow_permissions_actions_and_hosts(path):
    check_workflow(yaml.safe_load(path.read_text()))
    for line in path.read_text().splitlines():
        if "uses:" in line:
            assert re.search(r"@[0-9a-f]{40}\s+# v\d+\.\d+\.\d+$", line)


@pytest.mark.parametrize("change", ["permissions", "action", "runner", "credentials"])
def test_workflow_guard_rejects_unsafe_configuration(change):
    document = copy.deepcopy(yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text()))
    job = document["jobs"]["quality"]
    if change == "permissions":
        document.pop("permissions")
    elif change == "action":
        job["steps"][0]["uses"] = "actions/checkout@v4"
    elif change == "runner":
        job["runs-on"] = ["self-hosted", "tower-gpu"]
    else:
        job["steps"][0]["with"]["persist-credentials"] = True
    with pytest.raises(AssertionError):
        check_workflow(document)


def test_public_ci_requires_public_data():
    assert (
        yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["env"][
            "TOWER_REQUIRE_PUBLIC_DATA"
        ]
        == "1"
    )


@pytest.mark.parametrize(
    "runner",
    [
        "tower-gpu",
        {"group": "tower-gpu-runners"},
        "${{ vars.RUNNER }}",
        None,
        ["ubuntu-latest", "tower-gpu"],
        "self-hosted",
    ],
)
def test_workflow_guard_rejects_unapproved_runner_forms(runner):
    with pytest.raises(AssertionError):
        check_workflow(dict(permissions={"contents": "read"}, jobs={"job": {"runs-on": runner}}))


@pytest.mark.parametrize("runner", ["ubuntu-latest", "ubuntu-24.04"])
def test_workflow_guard_accepts_hosted_runner_labels(runner):
    check_workflow(dict(permissions={"contents": "read"}, jobs={"job": {"runs-on": runner}}))
