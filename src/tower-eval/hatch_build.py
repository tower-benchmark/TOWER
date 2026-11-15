# SPDX-License-Identifier: Apache-2.0
"""Ship the canonical task registry without maintaining a second source copy."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        source = Path(self.root).parents[1] / "benchmarks/2026.11/tasks.json"
        destination = "tower_eval/_data/tasks-2026.11.json"
        if self.target_name == "sdist":
            destination = "src/" + destination
        if source.is_file():
            build_data["force_include"][str(source)] = destination
        elif not (Path(self.root) / "src/tower_eval/_data/tasks-2026.11.json").is_file():
            raise FileNotFoundError("The benchmark 2026.11 task registry is required to build")
