"""Process-scoped Kit crash-upload suppression for maintainer experiments."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

SETTINGS = {
    "/crashreporter/enabled": False,
    "/crashreporter/skipOldDumpUpload": True,
    "/crashreporter/devOnlyOverridePrivacyAndForceUpload": False,
    "/privacy/performance": False,
    "/privacy/usage": False,
    "/privacy/personalization": False,
    "/privacy/extraDiagnosticDataOptIn": "",
    "/telemetry/enableAnonymousData": False,
    "/exts/omni.kit.telemetry/skipDeferredStartup": True,
}


def configure(root: Path) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = root / "config"
    config.mkdir()
    privacy = config / "privacy.toml"
    privacy.write_text(
        '[privacy]\nversion = "1.0"\nperformance = false\nusage = false\n'
        'personalization = false\nextraDiagnosticDataOptIn = ""\n'
    )
    environment = dict(
        OMNI_CONFIG_PATH=str(config),
        OMNI_CRASHREPORTER_ENABLED="0",
        OMNI_CRASHREPORTER_SKIPOLDDUMPUPLOAD="1",
        OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA="1",
    )
    os.environ.update(environment)
    overrides = [f"--{key}={str(value).lower()}" for key, value in SETTINGS.items()]
    overrides += [
        "--portable-root",
        str(root),
        f"--/crashreporter/dumpDir={root / 'dumps'}",
        f"--/log/file={root / 'kit.log'}",
        f"--/structuredLog/privacySettingsFile={privacy}",
    ]
    sys.argv.extend(overrides)
    receipt = dict(
        settings=SETTINGS,
        argv=overrides,
        private_root=str(root),
        environment=environment,
        privacy_sha256=hashlib.sha256(privacy.read_bytes()).hexdigest(),
    )
    (root / "requested.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("TOWER_KIT_UPLOAD_SUPPRESSION_REQUESTED " + json.dumps(receipt), flush=True)


def verify(root: Path, *, phase: str = "verified") -> None:
    settings = importlib.import_module("carb.settings").get_settings()
    actual = {key: settings.get(key) for key in SETTINGS}
    if actual != SETTINGS:
        raise RuntimeError(f"Kit upload suppression differs: {actual!r}")
    if settings.get("/crashreporter/dumpDir") != str(root.resolve() / "dumps"):
        raise RuntimeError("Kit crash dump directory is not private")
    receipt = dict(verified=True, phase=phase, effective=actual, private_root=str(root.resolve()))
    (root / f"{phase}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("TOWER_KIT_UPLOAD_SUPPRESSION_VERIFIED " + json.dumps(receipt), flush=True)
