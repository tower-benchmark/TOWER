"""Reject private lineage on every public distribution surface."""

import ast
import ipaddress
import re
from pathlib import PurePosixPath


def check_lineage(name, payload):
    # Split tokens keep the enforcement code itself admissible in wheels.
    schemas = (b"tower-scene-" + b"lineage-v1", b"tower-internal-" + b"lineage-manifest-v1")
    if name == "scripts/maintainer/scene_v2_release.py":
        return
    if any(token in payload for token in schemas):
        raise ValueError(f"Internal lineage on public surface: {name}")


# Historical Python package/schema identifiers are compatibility contracts,
# not machine host names. Other run/host names with that prefix are scanned.
FORBIDDEN = re.compile(
    r"/(?:mnt|root|Users)/|/(?:home)/[^/\s\"']+"
    r"|aliy[u]n|\bjenga[-_]sim(?!_v1\b)|anonymous[-]review"
    r"|submission\W{0,3}\d|\b1378[4]\b|\bICL[R]\b|double[- ]blind|after paper acceptanc[e]"
    r"|\.(?:ecs\.)?internal\b|\b[a-z][\w-]*[.]lan\b|/na[s]/"
    r"|paper-isaac5[1]|refs/ta[g]s/|tag benchmark-2026[.]11[.]0|benchmark-2026[.]11[.]0:|jenga[-]desktop"
    r"|datagen-archive-v[1]|simulator-assets-v[12]|simulator-v2[.]0[.]1|git checkout benchmark-2026[.]11[.]0"
    r"|anonymous author[s]|(?<![\w/])mnt/na[s]|pypi[.]nvidia[.]cn",
    re.IGNORECASE,
)
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)")
# Dotted package versions are not IP literals. Recognize syntax, not arbitrary
# address values or file exemptions, so a new host/IP cannot inherit an exception.
VERSION = re.compile(
    r"""(?ix)
    (?:\bversion\s*[=:]\s*["']
       |["'](?:version|isaac_?sim|opencv-python(?:-headless)?)["']\s*:\s*["']
       |\bisaac(?:\s+sim)?\s+
       |\bisaacsim(?:\[[^\]\n]+\])?==
       |\bopencv-python(?:-headless)?(?:-noffmpeg)?==
       |\["isaacsim"\]\s*!=\s*"
       |\.get\("isaacsim"\)\s*!=\s*"
    )(?P<value>(?:\d{1,3}\.){3}\d{1,3})
    """
)
DISTRIBUTION_VERSION = re.compile(
    r"(?:[/\"])(?:isaacsim|opencv|nvidia)[a-zA-Z0-9_-]*-"
    r"(?P<value>(?:\d{1,3}\.){3}\d{1,3})(?:rc\d+)?(?:-(?:cp|py)\d|\.tar)"
)
DOCUMENTATION_NETWORKS = tuple(
    ipaddress.ip_network(value) for value in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
)


EMAIL = re.compile(
    r"[A-Z0-9._%+-]+@(?:gmail|yahoo|outlook|hotmail|qq|163|icloud|protonmail)[.]com", re.I
)


# A revision used as a data identity by this archive-only corpus builder.
# The exact value and source file are listed; no path/glob exception is allowed.
ARCHIVE_IDENTITY_ALLOWLIST = {
    "scripts/maintainer/build_golden_corpus.py": (
        "benchmark-2026.11.0" + ":finalize_trial/from_physical_rows/score_recording",
    ),
}


def violations(line, *, name=None):
    for identity in ARCHIVE_IDENTITY_ALLOWLIST.get(name or "", ()):
        line = line.replace(identity, "archive data identity")
    findings = [match.group() for match in FORBIDDEN.finditer(line)]
    addresses = list(IPV4.finditer(line))
    versions = (
        {
            match.span("value")
            for pattern in (VERSION, DISTRIBUTION_VERSION)
            for match in pattern.finditer(line)
        }
        if addresses
        else set()
    )
    for match in addresses:
        if match.span() in versions:
            continue
        try:
            address = ipaddress.IPv4Address(match.group())
        except ipaddress.AddressValueError:
            continue
        if str(address) in {"127.0.0.1", "0.0.0.0"}:
            continue
        if any(address in network for network in DOCUMENTATION_NETWORKS):
            continue
        findings.append(str(address))
    if "@" in line:
        findings.extend(match.group() for match in EMAIL.finditer(line))
    return findings


REMOVED_MODULES = (
    "jenga_" + "generator",
    "replay_runner_" + "isaac_sim",
    "tower_sim." + "generation",
    "tower_sim.backend." + "model",
    "tower_eval.simulator." + "isaac",
    "cu" + "robo",
    "curobo_" + "planner",
)
REMOVED_SYMBOLS = {
    "Generation" + "Config",
    "plan_" + "free_space",
    "Curobo" + "Planner",
    "Generator" + "Simulator",
    "Motion" + "Policy",
    "Tower" + "Transfer",
    "generation_" + "profile",
    "fit_" + "profile",
    "procedural_" + "profile",
    "planner_" + "factory",
    "policy_" + "factory",
    "build_" + "layout",
    "TaskLayout" + "Config",
    "episode_" + "seed",
}


def check_generation(name, payload):
    parts = PurePosixPath(name).parts
    dotted = name.replace("/", ".").removesuffix(".py")
    if "tower-" + "datagenerator" in parts or any(token in dotted for token in REMOVED_MODULES):
        raise ValueError(f"Generation module on public surface: {name}")
    if not name.endswith(".py"):
        return
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names = [node.name]
        elif isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [a.name for a in node.names]
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
        ):
            names = [str(node.args[0].value)]
        if any(
            n in REMOVED_SYMBOLS or any(n == m or n.startswith(m + ".") for m in REMOVED_MODULES)
            for n in names
        ):
            raise ValueError(
                f"Generation reference on public surface: {name}:{getattr(node, 'lineno', 0)}"
            )


def check_archive(name, payload):
    if name.startswith("tests/golden/") and name.endswith(".tar.xz"):
        raise ValueError(f"Archived original recording on public surface: {name}")
    if PurePosixPath(name).parts[:2] == ("simulator", "benchmark-202611"):
        raise ValueError(f"Archived bundle on public surface: {name}")
    token = "archive_" + "compat"
    if token in PurePosixPath(name).parts or PurePosixPath(name).stem == token:
        raise ValueError(f"Archive compatibility module on public surface: {name}")
    if name.endswith(".py"):
        for node in ast.walk(ast.parse(payload)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or "", *(a.name for a in node.names)]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ):
                names.append(str(node.args[0].value))
            if any(token in n.split(".") for n in names):
                raise ValueError(f"Archive compatibility import on public surface: {name}")


def check_public_archive_paths(name, payload):
    """Public simulator routes must obtain pinned evaluation inputs from package data."""
    path = PurePosixPath(name)
    if path.parent != PurePosixPath("tools/simulator") or path.suffix != ".py":
        return
    forbidden = {"BUNDLE", "RUNTIME", "ArchivedPath"}
    for node in ast.walk(ast.parse(payload)):
        names = (
            [node.id]
            if isinstance(node, ast.Name)
            else [node.attr]
            if isinstance(node, ast.Attribute)
            else [item.name for item in node.names]
            if isinstance(node, ast.ImportFrom)
            else []
        )
        if forbidden.intersection(names):
            raise ValueError(
                f"Archive path in public simulator route: {name}:{getattr(node, 'lineno', 0)}"
            )


def check_public(name, payload):
    check_generation(name, payload)
    check_archive(name, payload)
    check_public_archive_paths(name, payload)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError:
        text = ""
    findings = violations(name) + [
        v for line in text.splitlines() for v in violations(line, name=name)
    ]
    if findings:
        raise ValueError(f"Private identifiers on public surface: {name}: {findings}")
