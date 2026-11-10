"""Scene description: redlego (data_so101/eruedas__so101_3cams_pickplace-red-lego_real, LeRobot v3.0, so101_follower).

Black SO-101 (black jaws, white/red servo cables) on a white table against a white wall; a brown open cardboard box
(left of the arm in the top view, in the foreground of the side view) and a red toy brick. Task "Grab the red lego
block and place it in the brown box". Cameras (both static: 0 px phase-correlation shift between the 40 episode
medians, static_check.py; only the exposure changes between sessions):
  'top'  = observation.images.top   (front, elevated: looks at the arm from the +y side, the box at the image left),
  'side' = observation.images.left  (from the +x side, behind the box, lower and closer; +y to the image right).
  observation.images.gripper (wrist camera) is not used.
World frame: the arm's normalised frame (arm.py): base bottom at z = 0 = table top, pan axis = world z, pan 0 reaches
+y (towards the top camera); positive pan swings the arm towards +x (the box).
Joint units: see README (normalised -100..100 vs degrees: fitted per-joint scales, prior centred on 1 deg / unit).
Camera inits: seed.py (render search against rough masks), refined by jseed.py / fit.py.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

import arm as A

HERE = pathlib.Path(__file__).resolve().parent
REAL_SET = "so101_redlego_real"
BLACK = (0.045, 0.045, 0.05, 1)
ARM = A.ArmSpec(calib="new", model="so101", wrist_mount=False, base_pos=(0.0, 0.0, 0.0), base_yaw_deg=0.0,
                colors={"body": BLACK, "jaw": BLACK, "motor": (0.03, 0.03, 0.03, 1)})


def _seed(name, default):
    p = HERE / "out" / f"seed_{name}.json"
    if p.exists():
        j = json.loads(p.read_text())
        return dict(mount="world", pos=tuple(j["pos"]), quat=tuple(j["quat"]), fovy=float(j["fovy"]))
    return default


CAMERAS = {
    "top": _seed("top", dict(mount="world", pos=(0.0, 0.75, 0.45),
                             quat=tuple(A.look_quat((0.0, 0.75, 0.45), (0.0, 0.1, 0.05))), fovy=55.0)),
    "side": _seed("side", dict(mount="world", pos=(0.55, 0.05, 0.25),
                               quat=tuple(A.look_quat((0.55, 0.05, 0.25), (0.0, 0.12, 0.1))), fovy=60.0)),
}
GROUP_RULE = {"top": "single pose (static camera: 0 px shift between episode medians)",
              "side": "single pose (static camera: 0 px shift between episode medians)"}
OFFSETS_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]
OFFSET_SIGMA = [8.0, 8.0, 8.0, 8.0, 15.0]
FIT_SCALES = True
SCALES_INIT = [1.0, 1.0, 1.0, 1.0, 1.0]
SCALES_PRIOR = [1.0, 1.0, 1.0, 1.0, 1.0]
SCALE_SIGMA = [0.15, 0.15, 0.15, 0.15, 0.08]   # roll: state nearly constant (-55 +- 4 units), scale pinned near 1
JOINT_PARAMS = ("off", "grip", "gripoff")
ROLL_CANDIDATES = None

MASK_OCCLUDERS = False   # fit.py flags: the arm-only twin mask is never occluded by twin objects
CUP_W = 0.0

JOINT_UNITS = ("LeRobot so101_follower, 5 arm joints in LeRobot's NORMALISED mode (-100..100 of each joint's calibrated "
               "travel, use_degrees=False): meta/stats.json reaches -98.7 / +99.7 without saturating at exactly +-100, so "
               "the units were settled by the fitted per-joint scales 0.96-1.18 deg/unit (joint_scales_deg_per_unit; "
               "degrees mode would give 1.0); gripper 0..100 of its travel (0 = closed)")

# ---------------------------------------------------------------------------------------------------- objects (RGB only)
# The metric never uses them (arm-only masks). Box: brown open cardboard box, shape fitted once to the real DINO+SAM box
# masks through the fitted cameras, placed per episode; lego: red 2x4 brick, per frame (table -> jaws -> box). Both from
# objects.py (out/objects.json); table / walls from the layout there (background edges cast through the fitted cameras).
OBJECTS_TEXT = ("so101_twin/redlego/out/objects.json (cardboard box: shape fitted to the real DINO+SAM box masks through "
                "the fitted cameras, placed per episode; red brick per frame: table -> jaws -> box, from the real DINO+SAM "
                "brick masks through the fitted cameras; table / wall planes from background edges; RGB only, never in "
                "the metric)")
_OBJ_PATH = HERE / "out" / "objects.json"
OBJ = json.loads(_OBJ_PATH.read_text()) if _OBJ_PATH.exists() else None
BOX = dict(lx=0.30, ly=0.20, h=0.07, t=0.004)
LEGO = dict(lx=0.0637, ly=0.0317, h=0.0192, stud_r=0.0047, stud_h=0.0045)
LAYOUT = dict(back=None, side=None)
if OBJ is not None:
    BOX.update(OBJ.get("box_shape", {}))
    LAYOUT = OBJ.get("layout", LAYOUT)


def _fmt3(v):
    return " ".join(f"{float(x):.5g}" for x in v)


def extras(root, asset, wb) -> None:
    import xml.etree.ElementTree as ET

    ET.SubElement(asset, "material", name="table", rgba="0.86 0.86 0.85 1", specular="0.05", shininess="0.1")
    ET.SubElement(asset, "material", name="wall", rgba="0.84 0.84 0.85 1", specular="0.0", shininess="0.0")
    ET.SubElement(asset, "material", name="cardboard", rgba="0.62 0.47 0.33 1", specular="0.02", shininess="0.05")
    ET.SubElement(asset, "material", name="cardboard_in", rgba="0.55 0.41 0.29 1", specular="0.02", shininess="0.05")
    ET.SubElement(asset, "material", name="lego", rgba="0.85 0.10 0.08 1", specular="0.3", shininess="0.5")
    ET.SubElement(wb, "light", name="key", pos="0.2 0.5 1.5", dir="-0.1 -0.3 -1", diffuse="0.5 0.5 0.5",
                  specular="0.1 0.1 0.1", castshadow="true")
    te = (LAYOUT or {}).get("table_edge")
    if te:   # the table's -x edge (right in the top view) from objects.py; a dark floor beyond it
        xe = 0.5 * (te[0][0] + te[1][0])
        ET.SubElement(wb, "geom", name="table", type="box", size="1.0 1.2 0.01", pos=f"{xe + 1.0:.4f} 0.2 -0.01",
                      material="table", contype="0", conaffinity="0", group="1")
        ET.SubElement(asset, "material", name="floor", rgba="0.45 0.45 0.46 1", specular="0.0", shininess="0.0")
        ET.SubElement(wb, "geom", name="floor", type="box", size="3 3 0.01", pos="0 0 -0.76", material="floor",
                      contype="0", conaffinity="0", group="1")
    else:
        ET.SubElement(wb, "geom", name="table", type="box", size="1.2 1.2 0.01", pos="0 0.2 -0.01", material="table",
                      contype="0", conaffinity="0", group="1")
    for k in ("back", "side"):
        w = (LAYOUT or {}).get(k)
        if w:
            ET.SubElement(wb, "geom", name=f"wall_{k}", type="box", size=_fmt3(w["size"]), pos=_fmt3(w["pos"]),
                          quat=_fmt3(w["quat"]), material="wall", contype="0", conaffinity="0", group="1")
    lx, ly, h, t = BOX["lx"] / 2, BOX["ly"] / 2, BOX["h"], BOX["t"]
    bb = ET.SubElement(wb, "body", name="box", pos="0.3 0.0 0", mocap="true")
    kw = dict(type="box", contype="0", conaffinity="0", group="1")
    ET.SubElement(bb, "geom", name="box_floor", size=_fmt3((lx, ly, t / 2)), pos=_fmt3((0, 0, t / 2)),
                  material="cardboard_in", **kw)
    for nm, sz, ps in (("box_wx0", (t / 2, ly, h / 2), (-lx + t / 2, 0, h / 2)),
                       ("box_wx1", (t / 2, ly, h / 2), (lx - t / 2, 0, h / 2)),
                       ("box_wy0", (lx, t / 2, h / 2), (0, -ly + t / 2, h / 2)),
                       ("box_wy1", (lx, t / 2, h / 2), (0, ly - t / 2, h / 2))):
        ET.SubElement(bb, "geom", name=nm, size=_fmt3(sz), pos=_fmt3(ps), material="cardboard", **kw)
    lb = ET.SubElement(wb, "body", name="lego", pos="0 0.25 -1", mocap="true")
    L = LEGO
    ET.SubElement(lb, "geom", name="lego_body", size=_fmt3((L["lx"] / 2, L["ly"] / 2, L["h"] / 2)), material="lego", **kw)
    for a_ in range(4):
        for b_ in range(2):
            ET.SubElement(lb, "geom", name=f"lego_stud{a_}{b_}", type="cylinder",
                          size=f"{L['stud_r']:.4f} {L['stud_h'] / 2:.4f}",
                          pos=_fmt3(((a_ - 1.5) * L["lx"] / 4, (b_ - 0.5) * L["ly"] / 2, L["h"] / 2 + L["stud_h"] / 2)),
                          material="lego", contype="0", conaffinity="0", group="1")


def frame_setup(rig, fs, i) -> None:
    """place the box (per episode) and the brick (per frame) from out/objects.json; hidden when unknown."""
    import mujoco

    m, d = rig.arm.m, rig.arm.d
    bb = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")]
    lb = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lego")]
    if OBJ is None:
        d.mocap_pos[bb] = (0, 0, -2)
        d.mocap_pos[lb] = (0, 0, -2)
        return
    e = str(int(fs.labels["episode"][i]))
    bx = OBJ["box"].get(e) or OBJ.get("box_default")
    if bx is None:
        d.mocap_pos[bb] = (0, 0, -2)
    else:
        d.mocap_pos[bb] = (bx[0], bx[1], 0.0)
        d.mocap_quat[bb] = A.quat_from_rotvec([0, 0, float(bx[2])])
    c = OBJ["lego"].get(str(int(i)))
    if c is None:
        d.mocap_pos[lb] = (0, 0, -2)
    elif c[0] == "held":   # in the jaws: at the tool centre point of the posed arm, aligned with the gripper
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        mujoco.mj_kinematics(m, d)
        R = d.site_xmat[sid].reshape(3, 3)
        off = np.asarray(OBJ.get("held_offset", [0.0, 0.0, 0.0]), float)
        d.mocap_pos[lb] = d.site_xpos[sid] + R @ off
        d.mocap_quat[lb] = A.quat_mul(A.quat_from_mat(R), A.quat_from_rotvec(OBJ.get("held_rot", [0.0, 0.0, 0.0])))
    else:
        d.mocap_pos[lb] = c[1]
        d.mocap_quat[lb] = A.quat_from_rotvec([0, 0, float(c[2]) if len(c) > 2 else 0.0])
    mujoco.mj_kinematics(m, d)
