"""Registry identity/topology plus the explicitly resolved campaign protocol."""

from dataclasses import dataclass

from tower_eval.tasks import Task, benchmark_task, load_tasks

from .resources import Resources


@dataclass(frozen=True)
class BenchTask:
    identity: Task
    deadline_s: int
    early_success_stop: bool
    success_rule: str
    physics_hz: int

    @property
    def duration_ticks(self):
        return round(self.deadline_s * self.physics_hz)

    @classmethod
    def from_resources(cls, task, resources: Resources):
        identity = benchmark_task(task)
        registry = {t.id: t for t in load_tasks(resources.path("tasks")) if t.status == "benchmark"}
        protocol = resources.read("protocol")
        if protocol["schema"] != "tower-benchmark-202611-v1":
            raise ValueError("Expected the 2026.11 benchmark protocol")
        if {t.legacy_id for t in registry.values()} != set(protocol["tasks"]):
            raise ValueError("Registry and protocol task sets differ")
        for registered in registry.values():
            entry = protocol["tasks"][registered.legacy_id]
            if (
                registered != benchmark_task(registered.id)
                or registered.deadline_s != entry["duration_s"]
                or registered.success_rule != entry["success_rule"]
                or registered.scorer_ruleset != "bench-2026.11"
                or type(entry["early_success_stop"]) is not bool
            ):
                raise ValueError(f"Registry/protocol mismatch: {registered.id}")
        row = protocol["tasks"][identity.legacy_id]
        hz = protocol["simulation"]["physics_hz"]
        if hz != 400:
            raise ValueError("Campaign evidence requires the 400 Hz physics clock")
        return cls(identity, row["duration_s"], row["early_success_stop"], row["success_rule"], hz)


def validate_spec(spec, expected_task=None):
    m = spec["layout_metadata"]
    task = m.get("evaluation_task")
    if task not in {t.legacy_id for t in load_tasks() if t.status == "benchmark"} or (
        expected_task is not None and task != expected_task
    ):
        raise ValueError("Missing/mismatched explicit six-task identity")
    identity = benchmark_task(task)
    assert identity.counts is not None
    number, layers, blocks = identity.number, identity.layers, identity.block_count
    transfers = 0 if number == 4 else 2 * (layers - 1) if number == 7 else identity.counts.planned_p
    if m.get("catalog_task") != number or m.get("movable_layers") != layers:
        raise ValueError("Task family/layers mismatch")
    if len(spec["blocks"]) != blocks or len(spec["transfers"]) != transfers:
        raise ValueError("Task topology/transfer count mismatch")
    if sum(bool(b["fixed_on_table"]) for b in spec["blocks"]) != 6:
        raise ValueError("Six fixed foundation blocks required")
    indices = []
    for t in spec["transfers"]:
        i = t["block_index"]
        indices.append(i)
        if (
            not 0 <= i < blocks
            or spec["blocks"][i]["fixed_on_table"]
            or spec["blocks"][i]["name"] != t["block_id"]
            or t["arm"] not in (0, 1)
        ):
            raise ValueError("Invalid movable target identity")
    if len(set(indices)) != transfers:
        raise ValueError("Duplicate target")
    if number == 4:
        pushes = m.get("push_targets", [])
        if len(pushes) != 3 * layers or len({p["block_index"] for p in pushes}) != 3 * layers:
            raise ValueError("Task4 requires all distinct ordered pushes")
        for order, p in enumerate(pushes):
            i = p["block_index"]
            expected_layer = layers - order // 3
            if (
                not 0 <= i < blocks
                or spec["blocks"][i]["fixed_on_table"]
                or spec["blocks"][i]["name"] != p["block_id"]
            ):
                raise ValueError("Invalid push target identity")
            if p["order"] != order or p["layer"] != expected_layer or p["arm"] != (order // 3) % 2:
                raise ValueError("Task4 push order/arm differs from the qualified left-first scene")
            if p["contact_face"] != "ordinary_near_long_axis_end":
                raise ValueError("Unqualified push contact face")
    if number == 6 and m.get("placement_mode") != "same_tower_roof":
        raise ValueError("Task6 requires its approved same-tower roof geometry")
    if number == 7:
        cycles = m.get("cycles", [])
        if len(cycles) != layers - 1:
            raise ValueError("Task7 cycle metadata missing")
        if (
            number == 7
            and layers == 3
            and m.get("source_center_bond", {}).get("block_indices") != [7, 10]
        ):
            raise ValueError("Task73 retained-center bond missing")
    return task
