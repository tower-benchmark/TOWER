"""Original reference windows and exact actor matching; no generation imports."""

import numpy as np

from .evidence import BatchContacts, ScreenContacts


def reference_contact_windows(parameters, candidate):
    by_name = {b["name"]: i for i, b in enumerate(candidate["episode_spec"]["blocks"])}
    windows = []
    for arm in ("back", "left"):
        keys = parameters["keyframes"][arm]
        grasps = [k for k in keys if k[0] == "grasp"]
        releases = [k for k in keys if k[0] == "release"]
        names = parameters["scene"]["pick_order"][arm]
        for name, grasp, release in zip(names, grasps, releases, strict=True):
            start, end = float(grasp[1]), float(release[2])
            if not np.isfinite([start, end]).all() or not 0 <= start < end:
                raise ValueError("Invalid source grasp/release contact window")
            windows.append(
                dict(
                    arm=arm,
                    block_name=name,
                    block_index=by_name[name],
                    start_time_s=start,
                    end_time_s=end,
                    event_count=0,
                    observed=False,
                )
            )
    return windows


def reference_contact_matches(actors, stamp, window):
    stand = "StandA_Robot" if window["arm"] == "left" else "StandB_Robot"
    return (
        window["start_time_s"] <= stamp <= window["end_time_s"]
        and any(p.endswith(f"/Blocks/block_{window['block_index']:02d}") for p in actors)
        and any(stand in p and p.endswith(("/gripper_link1", "/gripper_link2")) for p in actors)
    )


def make_contacts(env, rows, *, reference_windows=False):
    cls = BatchContacts if reference_windows else ScreenContacts
    return cls(
        env,
        rows,
        reference_windows=reference_windows,
        contact_windows=reference_contact_windows,
        contact_matches=reference_contact_matches,
    )
