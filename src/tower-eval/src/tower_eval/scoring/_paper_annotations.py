"""Read source ratings without passing them into the trajectory detector."""

import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from statistics import mean
from zipfile import ZipFile

from tower_eval.scoring.rules import task_counts

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
FIELDS = {
    "s": "任务成功",
    "g": "抓取成功",
    "p": "放置成功",
    "raw_progress": "长程进度",
    "D": "操作扰动",
}


def worksheet_rows(archive, path, shared):
    root = ET.fromstring(archive.read(path))
    for row in root.findall("s:sheetData/s:row", NS):
        values = []
        for cell in row.findall("s:c", NS):
            coordinate = cell.get("r")
            if coordinate:
                column = 0
                match = re.match(r"[A-Z]+", coordinate)
                if match is None:
                    raise TypeError(f"Invalid worksheet cell reference: {coordinate!r}")
                for letter in match[0]:
                    column = column * 26 + ord(letter) - 64
                while len(values) < column - 1:
                    values.append("")
            value = cell.find("s:v", NS)
            value = "" if value is None else value.text or ""
            if cell.get("t") == "s":
                value = shared[int(value)]
            elif cell.get("t") == "inlineStr":
                value = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
            values.append(value)
        yield values


def read_annotations(path):
    episodes, ratings, skipped = {}, [], Counter()
    seen = set()
    sources_by_identity = {}
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = [
                "".join(t.text or "" for t in item.findall(".//s:t", NS))
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ]
        rels = {
            r.attrib["Id"]: r.attrib["Target"]
            for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in workbook.findall("s:sheets/s:sheet", NS):
            match = re.fullmatch(r"(.+)_sim_(.+)", sheet.attrib["name"])
            if not match:
                continue
            annotator, policy = match.groups()
            rid = sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            target = rels[rid]
            target = (
                target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl") / target)
            )
            rows = worksheet_rows(archive, target, shared)
            headers = next(rows)
            required = {"Task", "No.", "Episode", "源文件", *FIELDS.values()}
            if not required.issubset(headers):
                raise ValueError(f"Missing annotation columns in {sheet.attrib['name']}")
            for row_number, row in enumerate(rows, 2):
                row = dict(zip(headers, row))
                task = row.get("Task", "")
                if not task:
                    continue
                if task.startswith("task7-"):
                    skipped["task7_rows"] += 1
                    continue
                task_counts(task)
                source = row.get("源文件", "")
                if not source:
                    raise ValueError(
                        f"Missing trajectory source in {sheet.attrib['name']}:{row_number}"
                    )
                identity = (task, policy, row["No."])
                if identity in sources_by_identity and sources_by_identity[identity] != source:
                    raise ValueError(f"Conflicting trajectory sources for {identity}")
                sources_by_identity[identity] = source
                if source in episodes and episodes[source]["identity"] != identity:
                    raise ValueError(f"Conflicting identity for trajectory {source}")
                episodes[source] = dict(
                    identity=identity,
                    task=task,
                    policy=policy,
                    episode=row["Episode"],
                    source=source,
                )
                present = [row.get(column, "") != "" for column in FIELDS.values()]
                if not any(present):
                    skipped["unrated_rows"] += 1
                    continue
                if not all(present):
                    raise ValueError(f"Partial rating in {sheet.attrib['name']}:{row_number}")
                values = {}
                for field, column in FIELDS.items():
                    value = float(row[column])
                    if (
                        not value.is_integer()
                        or value < 0
                        or (field == "s" and value not in (0, 1))
                    ):
                        raise ValueError(f"Invalid {field} in {sheet.attrib['name']}:{row_number}")
                    values[field] = int(value)
                if (source, annotator) in seen:
                    raise ValueError("Duplicate source rating from the same annotator")
                seen.add((source, annotator))
                values["C"] = values["g"] + values["p"]
                values["E"] = values["C"] + 1 - values["s"]
                ratings.append(
                    dict(
                        source=source,
                        annotator=annotator,
                        sheet=sheet.attrib["name"],
                        row=row_number,
                        **values,
                    )
                )
    if not episodes:
        raise ValueError("No supported SIM episodes in workbook")
    return list(episodes.values()), ratings, dict(skipped)


def agreement(pairs):
    if not pairs:
        return dict(n=0, exact_match=None, mae=None, bias=None)
    errors = [prediction - label for prediction, label in pairs]
    result = dict(
        n=len(pairs),
        exact_match=mean(e == 0 for e in errors),
        mae=mean(abs(e) for e in errors),
        bias=mean(errors),
    )
    return result


def aggregate(reports):
    valid = [r for r in reports if r.get("valid") and r.get("scores")]

    def ratio(a, b):
        return a / b if b else None

    return dict(
        episodes=len(reports),
        valid_episodes=len(valid),
        TSR=ratio(sum(r["scores"]["s"] for r in valid), len(valid)),
        GSR=ratio(sum(r["scores"]["g"] for r in valid), sum(r["planned_g"] for r in valid)),
        PSR=ratio(sum(r["scores"]["p"] for r in valid), sum(r["planned_p"] for r in valid)),
        LHPR=ratio(
            sum(r["scores"]["C"] for r in valid),
            sum(r["planned_g"] + r["planned_p"] for r in valid),
        ),
        ODR=ratio(sum(r["scores"]["D"] for r in valid), sum(r["scores"]["E"] for r in valid)),
    )


def comparison_scores(report, push_columns):
    scores = report.get("scores")
    if (
        scores is None
        or not report.get("task", "").startswith("task4-")
        or push_columns == "paper-phase"
    ):
        return scores
    arms = report["push_successes_by_arm"]
    return {**scores, "g": arms["0"], "p": arms["1"]}


def validate(episodes, ratings, predictions, push_columns="paper-phase"):
    if push_columns not in ("paper-phase", "arm"):
        raise ValueError("Unknown Task 4 annotation column mapping")
    fields = ("s", "g", "p", "C", "D")
    grouped = defaultdict(list)
    individual = defaultdict(lambda: defaultdict(list))
    agreed = defaultdict(list)
    by_task = defaultdict(lambda: defaultdict(list))
    by_policy = defaultdict(lambda: defaultdict(list))
    disagreements, progress_conflicts, count_conflicts = [], [], []
    metadata = {e["source"]: e for e in episodes}
    for rating in ratings:
        grouped[rating["source"]].append(rating)
        if rating["raw_progress"] != rating["C"]:
            progress_conflicts.append(
                dict(
                    source=rating["source"],
                    annotator=rating["annotator"],
                    raw_progress=rating["raw_progress"],
                    C=rating["C"],
                )
            )
        _, _, g, p = task_counts(metadata[rating["source"]]["task"])
        if push_columns == "arm" and metadata[rating["source"]]["task"].startswith("task4-"):
            planned = predictions[rating["source"]].get("planned_pushes_by_arm")
            if planned is not None:
                g, p = planned["0"], planned["1"]
            else:
                # A failed recording load provides no arm assignment evidence.
                g = p = None
        if (
            g is not None
            and p is not None
            and (
                rating["g"] > g
                or rating["p"] > p
                or (rating["s"] and (rating["g"] != g or rating["p"] != p))
            )
        ):
            count_conflicts.append(dict(source=rating["source"], annotator=rating["annotator"]))
        prediction = comparison_scores(predictions[rating["source"]], push_columns)
        if prediction is not None:
            for field in fields:
                individual[rating["annotator"]][field].append((prediction[field], rating[field]))
    mismatches = []
    for source, rows in grouped.items():
        prediction = comparison_scores(predictions[source], push_columns)
        for field in fields:
            values = [r[field] for r in rows]
            # Agreement requires at least two independent raters and unanimity.
            # Disputes require adjudication; never silently select a majority.
            if len(set(values)) != 1:
                disagreements.append(
                    dict(
                        source=source, field=field, ratings={r["annotator"]: r[field] for r in rows}
                    )
                )
                continue
            if len(rows) < 2 or prediction is None:
                continue
            pair = (prediction[field], values[0])
            agreed[field].append(pair)
            by_task[metadata[source]["task"]][field].append(pair)
            by_policy[metadata[source]["policy"]][field].append(pair)
            if pair[0] != pair[1]:
                mismatches.append(
                    dict(source=source, field=field, automatic=pair[0], human=pair[1])
                )

    def summary(groups):
        return {
            key: {f: agreement(group[f]) for f in fields} for key, group in sorted(groups.items())
        }

    metric_groups = defaultdict(list)
    for source, prediction in predictions.items():
        meta = metadata[source]
        metric_groups[f"{meta['policy']}/{meta['task']}"].append(prediction)
    return dict(
        episodes=len(episodes),
        source_ratings=len(ratings),
        task4_column_mapping=push_columns,
        task4_column_semantics=(
            "g=simulator arm 0 pushes; p=simulator arm 1 pushes"
            if push_columns == "arm"
            else "paper G/P phases"
        ),
        scored_episodes=sum(p.get("scores") is not None for p in predictions.values()),
        individual=summary(individual),
        agreed={f: agreement(agreed[f]) for f in fields},
        agreed_by_task=summary(by_task),
        agreed_by_policy=summary(by_policy),
        disputed_fields=disagreements,
        progress_column_conflicts=progress_conflicts,
        annotation_count_conflicts=count_conflicts,
        mismatches=mismatches,
        automatic_metrics=aggregate(list(predictions.values())),
        automatic_metrics_by_policy_task={
            key: aggregate(group) for key, group in sorted(metric_groups.items())
        },
        unscored_episodes=[
            dict(source=source, reason=p.get("error", p.get("invalid_reason")))
            for source, p in predictions.items()
            if p.get("scores") is None
        ],
        success_confusion={
            f"human_{label}_automatic_{prediction}": sum(
                a == prediction and b == label for a, b in agreed["s"]
            )
            for label in (0, 1)
            for prediction in (0, 1)
        },
    )


def resolve_run(source, source_root=None):
    path = Path(source)
    if source_root is not None:
        parts = path.parts
        if parts.count("runs") != 1:
            raise ValueError("Source remapping requires one /runs/ path component")
        path = Path(source_root).joinpath(*parts[parts.index("runs") + 1 :])
        root = Path(source_root).resolve()
        if not path.resolve().is_relative_to(root):
            raise ValueError("Trajectory source escapes the supplied runs root")
    if path.name != "states.hdf5" or path.parent.name != "episode":
        raise ValueError("Expected a source ending in episode/states.hdf5")
    return path.parent.parent
