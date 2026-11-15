"""Re-pin the explicitly versioned paper reference after an intentional edit.

Bump reproduction.paper_metrics.VERSION first. Frozen benchmark evidence is
outside this manifest and is never rewritten by this tool.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "reproduction/paper_metrics"


def main():
    import hashlib
    import runpy

    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    version = runpy.run_path(str(ROOT / "__init__.py"))["VERSION"]
    if version == manifest["version"]:
        raise ValueError("Bump paper_metrics.VERSION before re-pinning")
    names = sorted({*manifest["files_sha256"], "annotations.py", "scoring_cli.py"})
    manifest["version"] = version
    manifest["files_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"version": version, "pinned_files": len(names)}))


if __name__ == "__main__":
    main()
