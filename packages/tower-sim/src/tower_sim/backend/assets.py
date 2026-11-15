# SPDX-License-Identifier: Apache-2.0
"""Derived kinematics and scene preparation; source USD assets remain untouched."""

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import yaml

from tower_sim.config import DEFAULT_STAGE
from .resources import default_path

ROOT = "/PNP_DUALD1200_STATION"
REPORT = default_path("calibration_report")


def block_reference(source: str) -> tuple[str, str]:
    """Support both packaged block layers and the current flattened station."""
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            members = [name for name in archive.namelist() if name.endswith("/jenga_block.usd")]
        if len(members) == 1:
            return f"{source}[{members[0]}]", ""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(source)
    root = stage.GetPrimAtPath(ROOT + "/ReplayTaskBlocks")
    if root:
        for prim in root.GetChildren():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and any(
                child.IsA(UsdGeom.Mesh)
                for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
            ):
                return source, str(prim.GetPath())
    raise ValueError("station asset contains no reusable physical Jenga block")


def checksum(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def calibration() -> tuple[dict, dict]:
    report = json.loads(REPORT.read_text())
    provenance = {
        "source": str(REPORT),
        "sha256": checksum(REPORT),
        "runtime_units": "kg_mm_seconds",
        "interpretation": "Recovered runtime readbacks of trajectory-domain calibration, not independently measured hardware dynamics.",
    }
    return report, provenance


def normalize_reset_stacks(stage, root) -> None:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache()
    replacements = []
    for prim in Usd.PrimRange(root):
        xform = UsdGeom.Xformable(prim)
        if xform and xform.GetResetXformStack():
            world = cache.GetLocalToWorldTransform(prim)
            parent = cache.GetLocalToWorldTransform(prim.GetParent())
            replacements.append((xform, world * parent.GetInverse()))
    for xform, matrix in replacements:
        xform.ClearXformOpOrder()
        xform.SetResetXformStack(False)
        xform.AddTransformOp().Set(matrix)


def export_kinematics(source: str, output: Path) -> tuple[Path, Path]:
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Gf, Usd, UsdGeom

    from tower_sim.replay.dual_replay_worker import ROBOT_PATHS

    enable_extension("isaacsim.asset.exporter.urdf")
    from nvidia.srl.from_usd.to_urdf import UsdToUrdf

    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.DefinePrim("/Nero", "Xform")
    root.GetReferences().AddReference(source, ROBOT_PATHS["left"])
    xform = UsdGeom.Xformable(root)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d(1))
    # Export only the serial arm. Finger and camera joints are irrelevant to TCP FK.
    for name in ("world", "gripper_link", "gripper_link1", "gripper_link2", "root_joint"):
        stage.GetPrimAtPath(f"/Nero/{name}").SetActive(False)
    for name in ("world_to_base_link", "gripper", "gripper_joint1", "gripper_joint2"):
        prim = stage.GetPrimAtPath(f"/Nero/joints/{name}")
        if prim:
            prim.SetActive(False)
    stage.GetPrimAtPath("/Nero/gripper_base/WristCameraMount").SetActive(False)
    converter = UsdToUrdf(stage, root="/Nero/base_link")
    urdf = output / "nero.urdf"
    converter.save_to_file(urdf, kinematics_only=True, quiet=True)
    tree = ET.parse(urdf)
    tree.getroot().set("name", "Nero")
    for link in tree.getroot().findall("link"):
        link.set("name", link.get("name").removeprefix("Nero_"))
    for joint in tree.getroot().findall("joint"):
        for element in (joint.find("parent"), joint.find("child")):
            element.set("link", element.get("link").removeprefix("Nero_"))
    ET.indent(tree)
    tree.write(urdf, encoding="utf-8", xml_declaration=True)
    description = output / "nero.yaml"
    description.write_text(
        yaml.safe_dump(
            {
                "api_version": 1.0,
                "cspace": [f"joint{i}" for i in range(1, 8)],
                "root_link": "base_link",
                "default_q": [0, -0.58, 0, 1.95, 0, -0.03, 1.25],
                "cspace_to_urdf_rules": [],
                "collision_spheres": [],
            }
        )
    )
    (output / "model.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source": source,
                "source_sha256": checksum(Path(source)),
                "urdf_sha256": checksum(urdf),
            },
            indent=2,
        )
    )
    return urdf, description


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--output", type=Path, default=Path("outputs/kinematics"))
    args = parser.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
    try:
        urdf, description = export_kinematics(args.stage, args.output)
        import lula

        robot = lula.load_robot(str(description), str(urdf))
        print("LULA FRAMES", robot.kinematics().frame_names(), flush=True)
        print("KINEMATICS READY", urdf, flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        import sys

        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
