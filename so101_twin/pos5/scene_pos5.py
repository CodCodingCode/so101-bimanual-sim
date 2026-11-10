"""Scene description: pos5 (data_so101/hbseong__record-pick-and-place-pos5-so101), official SO-101, new calibration.

White SO-101 in front of a green screen; it picks a cyan cube and drops it into a white open box. 'top' = the
dataset's observation.images.top (overhead, strong barrel distortion), 'side' = observation.images.right (static, low,
looking across the bench from the right). Joint units: LeRobot NORMALISED (-100..100 of each joint's calibrated
travel), so q = deg2rad(scale * state + offset) per joint (FIT_SCALES), gripper q = GRIP_LO + grip_scale * g/100 *
(GRIP_HI - GRIP_LO).

World frame = the fitkit session's (armfit.py): floor z = 0, arm base on the floor with its pan axis at
(0, 0.0452, 0), shoulder_pan = 0 reaching +y. The initial cameras / lens / latency / joint map are the fitkit best
fit (fitkit/out/pos5_so101_gain.json) converted exactly by convert_legacy.py (silhouette IoU legacy vs converted:
median 0.996 top / 0.990 side) -> out/init_from_legacy.json.
"""
from __future__ import annotations

import json
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import arm as A

HERE = pathlib.Path(__file__).resolve().parent
_INIT = json.loads((HERE / "out" / "init_from_legacy.json").read_text())

REAL_SET = "so101_pos5_real"
# The real arm has no Waveshare driver-board mounting plate at the back of the base (the side view shows the pale
# wall and the table clamp where the official model puts it); leaving it out raises the even-episode (fit-set) side
# IoU at fixed parameters (README). DROP_PLATE = False restores the full official model.
DROP_PLATE = True
ARM = A.ArmSpec(calib="new", model="so101", wrist_mount=False, base_pos=tuple(_INIT["arm"]["base_pos"]),
                base_yaw_deg=float(_INIT["arm"]["base_yaw_deg"]),
                colors={"body": (0.93, 0.93, 0.92, 1), "jaw": (0.93, 0.93, 0.92, 1), "motor": (0.08, 0.08, 0.08, 1)},
                drop_meshes=("waveshare_mounting_plate_so101_v2",) if DROP_PLATE else ())

_P = _INIT["params"]
CAMERAS = {c: dict(mount="world", pos=tuple(_P["cams"][c]["groups"]["all"]["pos"]),
                   quat=tuple(_P["cams"][c]["groups"]["all"]["quat"]), fovy=float(_P["cams"][c]["fovy"]))
           for c in ("top", "side")}
GROUP_RULE = {"top": "single pose", "side": "single pose"}
OFFSETS_INIT = list(_P["offsets_deg"])
OFFSET_SIGMA = [6.0, 6.0, 6.0, 6.0, 10.0]
FIT_LAG = True
FIT_SCALES = True
SCALES_INIT = list(_P["scales"])
SCALES_PRIOR = [1.0, 1.0, 1.0, 1.0, 1.0]
SCALE_SIGMA = [0.2, 0.2, 0.2, 0.2, 0.2]   # normalised units: deg/unit = travel/200, unknown per joint
JOINT_PARAMS = ("off", "grip", "gripoff")

# ---------------------------------------------------------------------------------------------- RGB scene (not fitted)
# Objects from the real frames only (objects.py -> out/objects.json): the white open box fitted to the box pixels of
# both median background images; the cyan cube on the floor from the top camera's cube pixels (back-projected), in the
# jaws while the gripper is closed in the grasp phase, else in the box. Green screen = flat green skybox + floor plane.
_OBJ_PATH = HERE / "out" / "objects.json"
_OBJ = json.loads(_OBJ_PATH.read_text()) if _OBJ_PATH.exists() else None
GREEN = (0.30, 0.60, 0.45)


def extras(root: ET.Element, asset: ET.Element, wb: ET.Element) -> None:
    hl = root.find("visual").find("headlight")   # brighter ambient: softer shadows, like the diffuse studio light
    hl.set("ambient", "0.45 0.45 0.45")
    hl.set("diffuse", "0.25 0.25 0.25")
    g = " ".join(f"{v:.3f}" for v in GREEN)
    ET.SubElement(asset, "texture", name="greenscreen", type="skybox", builtin="flat", rgb1=g, rgb2=g, width="16",
                  height="16")
    ET.SubElement(asset, "material", name="floor_green", rgba=f"{g} 1", specular="0.05", shininess="0.1")
    ET.SubElement(asset, "material", name="box_white", rgba="0.86 0.86 0.84 1", specular="0.1", shininess="0.2")
    ET.SubElement(asset, "material", name="cube_cyan", rgba="0.35 0.75 0.85 1", specular="0.2", shininess="0.3")
    ET.SubElement(wb, "light", name="key", pos="0.2 0.0 1.2", dir="0 0.2 -1", directional="false",
                  diffuse="0.30 0.30 0.30", specular="0.05 0.05 0.05", castshadow="true", cutoff="80")
    ET.SubElement(wb, "light", name="fill", pos="0.8 0.3 0.6", dir="-0.8 -0.1 -0.5", diffuse="0.2 0.2 0.2",
                  specular="0 0 0", castshadow="false")
    if _OBJ is None:
        return
    bx = _OBJ["box"]
    z0 = bx["z0"]
    zf = min(z0, min(float(v["pos"][2]) for v in CAMERAS.values())) - 0.01   # below every camera (seen from above)
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos=f"0 0 {zf:.4f}", material="floor_green",
                  contype="0", conaffinity="0")
    yaw = bx["yaw"]
    q = " ".join(f"{v:.6f}" for v in A.quat_from_rotvec([0, 0, yaw]))
    body = ET.SubElement(wb, "body", name="box", pos=f"{bx['center'][0]:.4f} {bx['center'][1]:.4f} {z0:.4f}", quat=q)
    hx, hy, H, t = bx["half"][0], bx["half"][1], bx["height"], 0.003
    for name, pos, size in (("box_floor", (0, 0, t / 2), (hx, hy, t / 2)),
                            ("box_wx0", (-hx + t / 2, 0, H / 2), (t / 2, hy, H / 2)),
                            ("box_wx1", (hx - t / 2, 0, H / 2), (t / 2, hy, H / 2)),
                            ("box_wy0", (0, -hy + t / 2, H / 2), (hx, t / 2, H / 2)),
                            ("box_wy1", (0, hy - t / 2, H / 2), (hx, t / 2, H / 2))):
        ET.SubElement(body, "geom", name=name, type="box", pos=" ".join(f"{v:.4f}" for v in pos),
                      size=" ".join(f"{v:.4f}" for v in size), material="box_white", contype="0", conaffinity="0")
    c = _OBJ.get("cube_half", 0.011)
    cb = ET.SubElement(wb, "body", name="cube", pos="0 0 -1", mocap="true")
    ET.SubElement(cb, "geom", name="cube", type="box", size=f"{c} {c} {c}", material="cube_cyan", contype="0",
                  conaffinity="0")


def frame_setup(rig, fs, i: int) -> None:
    if _OBJ is None:
        return
    m, d = rig.arm.m, rig.arm.d
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if bid < 0:
        return
    c = _OBJ.get("cube_half", 0.011)
    held = str(fs.labels["gripper_state"][i]) == "closed" and str(fs.labels["phase"][i]) == "grasp"
    xyz = _OBJ.get("cube_xyz", [None] * fs.n)[i]
    if held:
        pos = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")].copy()
    elif xyz is not None:
        pos = np.asarray(xyz, float)
    else:   # hidden: dropped into the box
        bx = _OBJ["box"]
        pos = np.array([bx["center"][0], bx["center"][1], bx["z0"] + 0.003 + c])
    d.mocap_pos[m.body_mocapid[bid]] = pos
    mujoco.mj_forward(m, d)
