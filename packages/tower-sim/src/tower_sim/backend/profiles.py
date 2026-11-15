# SPDX-License-Identifier: Apache-2.0
"""Verified historical motion-profile reader and digest contract."""

import json
from pathlib import Path

from tower_eval.simulator.spec import digest


def load_profile(path: Path) -> dict:
    profile = json.loads(path.read_text())
    expected = profile.pop("profile_sha256")
    if digest(profile) != expected:
        raise ValueError("motion profile checksum mismatch")
    split_by_id, split_by_content = {}, {}
    for source in profile["sources"]:
        for key, splits in (
            (source["episode_id"], split_by_id),
            (source["channel_sha256"], split_by_content),
        ):
            if key in splits and splits[key] != source["split"]:
                raise ValueError(
                    "demonstration identity or identical channel content crosses profile splits"
                )
            splits[key] = source["split"]
    if any(split_by_id.get(p["source_episode_id"]) != "fit" for p in profile["primitives"]):
        raise ValueError("motion primitive is not from a fitting episode")
    if any(
        split_by_id.get(e["source_episode_id"]) != "fit" for e in profile.get("grasp_episodes", [])
    ):
        raise ValueError("grasp episode is not from a fitting episode")
    profile["profile_sha256"] = expected
    return profile
