# SPDX-License-Identifier: Apache-2.0
"""Persistent stdin/stdout JSONL converter. Never connects to a robot."""

import argparse
import json
import sys

from . import ArmIK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--identity", action="store_true", help="Print model/solver identity and exit"
    )
    args = parser.parse_args()
    arm = ArmIK(args.config)
    if args.identity:
        print(json.dumps(arm.identity, allow_nan=False))
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {
                "pose",
                "measured_joints",
                "frame",
            }:
                raise ValueError("Each request requires exactly pose, measured_joints, frame")
            result = arm.solve(request["pose"], request["measured_joints"], frame=request["frame"])
            response = json.dumps(result, allow_nan=False)
        except (ValueError, TypeError, OverflowError) as exc:
            response = json.dumps({"success": False, "joints": None, "error": str(exc)})
        print(response, flush=True)


if __name__ == "__main__":
    main()
