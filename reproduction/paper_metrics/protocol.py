# SPDX-License-Identifier: Apache-2.0
"""Fixed manuscript protocol: Appendix A Table 4 and Sections 4.1, B, G."""

POLICIES = ("pi05", "starvla", "fastwam", "openwam")
DOMAINS = ("REAL", "SIM")
TASKS = {
    "A": {"name": "Cross Transfer", "legacy": 2, "counts": {2: (2, 2), 3: (3, 3)}},
    "B": {"name": "Two-Block Transfer", "legacy": 1, "counts": {2: (4, 4), 3: (6, 6)}},
    "C": {"name": "Extract and Stack", "legacy": 6, "counts": {2: (1, 1), 3: (2, 2)}},
    "D": {"name": "Lateral Insertion", "legacy": 5, "counts": {2: (1, 1), 3: (2, 2)}},
    "E": {"name": "Layerwise Pushdown", "legacy": 4, "counts": {2: (3, 3), 3: (6, 3)}},
    "F": {"name": "Tool-Assisted Extraction", "legacy": 7, "counts": {2: (5, 3), 3: (10, 6)}},
}
TRIALS_PER_GROUP = 10


def operation_plan(task, layers):
    """Ordered success-credit units, not individual motion phases or attempts."""
    g, _ = TASKS[task]["counts"][layers]
    if task == "E":
        units = [("G", ["PUSH"])] * g + [("P", ["PUSH"])] * 3
    elif task == "F":
        cycle = [
            ("G", ["GRASP tool"]),
            ("G", ["PUSH tool"]),
            ("G", ["PUSH tool"]),
            ("P", ["PLACE tool"]),
            ("G", ["GRASP", "EXTRACT"]),
            ("P", ["PLACE"]),
            ("G", ["GRASP", "EXTRACT"]),
            ("P", ["PLACE"]),
        ]
        units = cycle * (layers - 1)
    else:
        acquire = ["GRASP", "EXTRACT"] if task == "C" else ["GRASP"]
        place = ["INSERT"] if task == "D" else ["PLACE"]
        units = [("G", acquire), ("P", place)] * g
    return [
        {"operation_id": f"{task}{layers}-{i:02d}", "group": group, "primitives": primitives}
        for i, (group, primitives) in enumerate(units, 1)
    ]


def group_keys():
    return [
        (policy, task, layers, domain)
        for policy in POLICIES
        for task in TASKS
        for layers in (2, 3)
        for domain in DOMAINS
    ]
