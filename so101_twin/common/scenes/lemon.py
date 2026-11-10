"""Scene description: lemon (data_so101/Beegbrain__pick_lemon_and_drop_in_bowl), official SO-101, old calibration.

Objects / table / wall / lights are reused read-only from so101_twin/lemon/build_scene.py (same world frame: table
top z = 0, arm pan axis at (0, 0.0452), reaching +y). Camera inits are the builder's fitted values.
"""
from __future__ import annotations

import importlib.util
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import arm as A

_SRC = pathlib.Path(__file__).resolve().parents[2] / "lemon" / "build_scene.py"
_spec = importlib.util.spec_from_file_location("lemon_build_scene_ro", _SRC)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

REAL_SET = "so101_lemon_real"
ARM = A.ArmSpec(calib="old", wrist_mount=False, base_pos=(0.0, 0.0452, 0.0), base_yaw_deg=0.0,
                colors={"body": (0.93, 0.42, 0.10, 1), "jaw": (0.12, 0.42, 0.75, 1), "motor": (0.08, 0.08, 0.08, 1)})

# The real "top" camera (or the arm relative to it) sits differently in episodes 0-5 than in 6-39: the per-episode
# IoU of a single pose drops from ~0.87 to 0.58-0.80 there for both the builder twin and the first SO-101 fit, and
# the "side" camera shows no such step. So "top" gets one pose per group.
CAMERAS = {
    "top": dict(mount="world", pos=B.CAMS["top"]["pos"], fovy=B.CAMS["top"]["fovy"],
                quat=tuple(A.ypr_quat(B.CAMS["top"]["yaw"], B.CAMS["top"]["pitch"], B.CAMS["top"]["roll"])),
                group=lambda ep: "e00-05" if ep < 6 else "e06-39"),
    "side": dict(mount="world", pos=B.CAMS["side"]["pos"], fovy=B.CAMS["side"]["fovy"],
                 quat=tuple(A.ypr_quat(B.CAMS["side"]["yaw"], B.CAMS["side"]["pitch"], B.CAMS["side"]["roll"]))),
}
GROUP_RULE = {"top": "episodes 0-5 -> 'e00-05', 6-39 -> 'e06-39'", "side": "single pose"}
OFFSETS_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]
OFFSET_SIGMA = [4.0, 4.0, 4.0, 4.0, 6.0]


def extras(root: ET.Element, asset: ET.Element, wb: ET.Element) -> None:
    for name, kw in B.MATERIALS.items():
        if not name.startswith(("arm_", "jaw_")):
            ET.SubElement(asset, "material", name=name, **kw)
    ET.SubElement(asset, "material", name="bowl", rgba=B.BOWL["rgba"], specular="0.4", shininess="0.5")
    for name, o in B.OBJECTS.items():
        ET.SubElement(asset, "material", name=f"mat_{name}", rgba=o["rgba"], specular="0.3", shininess="0.4")
    ET.SubElement(wb, "light", name="sun", pos="0.3 0.6 1.6", dir="-0.2 -0.3 -1", directional="false",
                  diffuse="0.42 0.42 0.42", specular="0.1 0.1 0.1", castshadow="true", cutoff="90")
    ET.SubElement(wb, "light", name="fill", pos="0.8 -0.4 1.0", dir="-0.6 0.4 -0.8", diffuse="0.15 0.15 0.15",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.74", material="floor")
    ET.SubElement(wb, "geom", name="table_top", type="box", size=" ".join(map(str, B.TABLE_HALF)),
                  pos=f"{B.TABLE_CENTER[0]} {B.TABLE_CENTER[1]} {-B.TABLE_HALF[2]}", material="wood")
    ET.SubElement(wb, "geom", name="wall", type="box", size="0.01 2.0 1.2", pos=f"{B.WALL_X} 0.2 0.46", material="wall")
    for name, o in B.OBJECTS.items():
        x, y = o["xy"]
        b = ET.SubElement(wb, "body", name=name, pos=f"{x} {y} {o['size'][2]}", mocap="true")
        ET.SubElement(b, "geom", name=name, type=o["type"], size=" ".join(map(str, o["size"])),
                      material=f"mat_{name}", contype="0", conaffinity="0")
    bowl = ET.SubElement(wb, "body", name="bowl", pos="0 0 0")
    B.bowl_geoms(bowl)


def frame_setup(rig, fs, i: int) -> None:
    """Lemon on the table (approach), in the jaws (grasp / transfer_open) or in the bowl (release), as the builder."""
    m, d = rig.arm.m, rig.arm.d
    phase = str(fs.labels["phase"][i])
    r = B.OBJECTS["lemon"]["size"][2]
    if phase in ("grasp", "transfer_open"):
        pos = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")].copy()
    elif phase == "release":
        bx, by = B.BOWL["xy"]
        pos = np.array([bx, by, r + 0.004])
    else:
        x, y = B.OBJECTS["lemon"]["xy"]
        pos = np.array([x, y, r])
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lemon")
    d.mocap_pos[m.body_mocapid[bid]] = pos
    mujoco.mj_forward(m, d)
