"""Check a tracked public tree and optional wheels without importing their code."""

import argparse
import subprocess
import zipfile
from pathlib import Path

from tower_sim.bench.public_surface import check_public

from scripts.maintainer.scan_public_assets import Scanner


def check_tree(root):
    scanner = Scanner()
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    for name in names:
        path = root / name
        if name and path.is_file():
            check_public(name, path.read_bytes())
            scanner.scan(path, name)

    if not scanner.report()["passed"]:
        raise ValueError(f"Public asset scan failed: {scanner.hits}; {scanner.errors}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--wheel", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    check_tree(args.root)
    scanner = Scanner()
    for path in args.wheel:
        scanner.scan(path)
        with zipfile.ZipFile(path) as wheel:
            for name in wheel.namelist():
                check_public(name, wheel.read(name))
    if not scanner.report()["passed"]:
        raise ValueError(f"Public wheel scan failed: {scanner.hits}; {scanner.errors}")
    print("Public tree and wheel checks passed")


if __name__ == "__main__":
    main()
