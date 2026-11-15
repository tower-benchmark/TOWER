"""Create NEW characterization snapshots; never overwrite archived scores."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.maintainer.golden_support import (
    GOLDEN,
    benchmark,
    online_score,
    online_stream_sha256,
    package_score,
    read,
    unpack,
    write,
)
from tests.characterization.controller_scenarios import (
    controller_streams,
)
from tests.characterization.runtime_scenarios import (
    command_queue_stream,
    encoder_observation_output,
)
from tests.characterization.scene_resolution import (
    scene_resolution_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=GOLDEN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--units-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, result in (
        ("controllers", controller_streams()),
        ("command-queue", command_queue_stream()),
        ("encoder-observation", encoder_observation_output()),
        ("scene-resolve-sha256", scene_resolution_table()),
    ):
        write(args.output / (name + ".json"), result)
    if args.units_only:
        return
    for trial in read(GOLDEN / "MANIFEST.json")["trials"]:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            unpack(args.root / trial["archive"], run)
            for kind, function in (
                ("paper-v1", lambda run=run, trial=trial: package_score(run, trial["task"])),
                (
                    "benchmark-score-cli",
                    lambda run=run, trial=trial: benchmark.score_run(run, trial["task"]),
                ),
                ("online", lambda run=run: online_score(run)),
            ):
                result = function()
                suffix = ".json"
                if kind == "online" and "events" in result:
                    result["archived_stream_sha256"] = online_stream_sha256(result["events"])
                    suffix += ".xz"
                write(args.output / (trial["id"] + "." + kind + suffix), result)
                print(f"Snapshot {trial['id']} {kind}", flush=True)


if __name__ == "__main__":
    main()
