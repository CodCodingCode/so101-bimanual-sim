"""Scene description: grayblock (data_so101/Jeevesh2009__so101_gray_block_pickup_test, LeRobot v2.1, robot 'so101',
OLD 'main_*' calibration in DEGREES).

Black SO-101 (black jaws, red / white servo cable loops) on a white table against a white wall; a blue cardboard box
(white inside) to the arm's right as seen from 'top'; a small gray block. Task "Pick the gray block and keep it in the
box." Cameras (both static: 0 px phase-correlation shift between all 40 episode medians, static_check.py; the side
camera's exposure drops from episode 23 on and the box moves slightly then, the camera does not):
  'top'  = observation.images.top   (front, elevated: looks at the arm from the front; base left of centre, box right),
  'side' = observation.images.side  (elevated, from the arm's left-back; the black chair in the background).
World frame: the arm's normalised frame (arm.py): base bottom at z = 0 = table top, pan axis = world z, pan 0 reaches +y.
Joint units: LeRobot v2.1 OLD calibration, DEGREES (feature names main_*; shoulder_lift 187.9 and elbow_flex 179.4 at
the folded rest pose, impossible in the normalised -100..100 mode; no joint at exactly +-100 in the stats), so
q = deg2rad(OLD_SIGN * state + offset) per joint with scale fixed at 1 (like the lemon / ballcup scenes);
gripper (0..~50 here, 0 = closed): q = GRIP_LO + grip_off + grip_scale * g / 100 * (GRIP_HI - GRIP_LO).
Camera inits: seed.py (render search against twin-free masks), refined by jseed.py / fit.py.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

import arm as A

HERE = pathlib.Path(__file__).resolve().parent
REAL_SET = "so101_grayblock_real"
BLACK = (0.045, 0.045, 0.05, 1)
ARM = A.ArmSpec(calib="old", model="so101", wrist_mount=False, base_pos=(0.0, 0.0, 0.0), base_yaw_deg=0.0,
                colors={"body": BLACK, "jaw": BLACK, "motor": (0.035, 0.035, 0.035, 1)})


def _seed(name, default):
    p = HERE / "out" / f"seed_{name}.json"
    if p.exists():
        j = json.loads(p.read_text())
        return dict(mount="world", pos=tuple(j["pos"]), quat=tuple(j["quat"]), fovy=float(j["fovy"]))
    return default


CAMERAS = {
    "top": _seed("top", dict(mount="world", pos=(0.0, 0.70, 0.30),
                             quat=tuple(A.look_quat((0.0, 0.70, 0.30), (0.0, 0.1, 0.12))), fovy=50.0)),
    "side": _seed("side", dict(mount="world", pos=(-0.50, -0.30, 0.40),
                               quat=tuple(A.look_quat((-0.50, -0.30, 0.40), (0.0, 0.1, 0.08))), fovy=50.0)),
}
GROUP_RULE = {"top": "single pose (static camera: 0 px shift between episode medians)",
              "side": "single pose (static camera: 0 px shift between episode medians; exposure drops from episode 23)"}
OFFSETS_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]
OFFSET_SIGMA = [6.0, 6.0, 6.0, 6.0, 10.0]
FIT_SCALES = False          # degrees: 1 deg per unit by construction (old calibration)
JOINT_PARAMS = ("off", "grip", "gripoff")
ROLL_CANDIDATES = None

# ---------------------------------------------------------------------------------------------------- objects (RGB only)
# The metric never uses them (arm-only masks). Box: blue cardboard box with a white inside, per episode; gray block per
# frame (table -> jaws -> box). Placements come from objects.py (out/objects.json: twin-independent DINO+SAM object
# masks ray-cast through the FITTED cameras).
_OBJ_PATH = HERE / "out" / "objects.json"
OBJ = json.loads(_OBJ_PATH.read_text()) if _OBJ_PATH.exists() else None
BOX = dict(lx=0.20, ly=0.16, h=0.085, t=0.003)   # outer half... (full) sizes in m, filled from objects.json when present
BLOCK = 0.03
LAYOUT = dict(wall=None)
if OBJ is not None:
    BOX = OBJ.get("box_size", BOX)
    BLOCK = OBJ.get("block_size", BLOCK)
    LAYOUT = OBJ.get("layout", LAYOUT)


def _fmt3(v):
    return " ".join(f"{float(x):.5g}" for x in v)


def extras(root, asset, wb) -> None:
    import xml.etree.ElementTree as ET

    ET.SubElement(asset, "material", name="table", rgba="0.86 0.86 0.86 1", specular="0.05", shininess="0.1")
    ET.SubElement(asset, "material", name="wall", rgba="0.74 0.73 0.72 1", specular="0.0", shininess="0.0")
    ET.SubElement(asset, "material", name="boxout", rgba="0.10 0.36 0.70 1", specular="0.1", shininess="0.2")
    ET.SubElement(asset, "material", name="boxin", rgba="0.90 0.91 0.93 1", specular="0.05", shininess="0.1")
    ET.SubElement(asset, "material", name="block", rgba="0.55 0.56 0.58 1", specular="0.1", shininess="0.2")
    ET.SubElement(asset, "material", name="chair", rgba="0.05 0.08 0.07 1", specular="0.1", shininess="0.2")
    ET.SubElement(wb, "light", name="key", pos="0.2 0.4 1.5", dir="-0.1 -0.2 -1", diffuse="0.55 0.55 0.55",
                  specular="0.1 0.1 0.1", castshadow="true")
    ET.SubElement(wb, "geom", name="table", type="box", size="1.2 1.2 0.01", pos="0 0 -0.01", material="table",
                  contype="0", conaffinity="0", group="1")
    lay = LAYOUT or {}
    for k in ("wall", "wall2", "chair"):
        if lay.get(k):
            w = lay[k]
            ET.SubElement(wb, "geom", name=k, type="box", size=_fmt3(w["size"]), pos=_fmt3(w["pos"]),
                          quat=_fmt3(w.get("quat", (1, 0, 0, 0))), material=w.get("material", "wall"),
                          contype="0", conaffinity="0", group="1")
    # box: open-top, 4 walls (blue outside, white inside) + white floor; body origin at the floor centre
    bb = ET.SubElement(wb, "body", name="box", pos="0.2 0.0 0", mocap="true")
    lx, ly, h, t = BOX["lx"], BOX["ly"], BOX["h"], BOX["t"]
    for nm, pos, size in (("bx_px", (lx / 2 - t / 2, 0, h / 2), (t / 2, ly / 2, h / 2)),
                          ("bx_nx", (-lx / 2 + t / 2, 0, h / 2), (t / 2, ly / 2, h / 2)),
                          ("bx_py", (0, ly / 2 - t / 2, h / 2), (lx / 2, t / 2, h / 2)),
                          ("bx_ny", (0, -ly / 2 + t / 2, h / 2), (lx / 2, t / 2, h / 2))):
        ET.SubElement(bb, "geom", name=nm, type="box", size=_fmt3(size), pos=_fmt3(pos), material="boxout",
                      contype="0", conaffinity="0", group="1")
        # white inner skin, 0.6 mm inside the wall
        p_in = list(pos)
        s_in = list(size)
        ax = 0 if nm in ("bx_px", "bx_nx") else 1
        sgn = -1 if nm in ("bx_px", "bx_py") else 1
        p_in[ax] = pos[ax] + sgn * (t / 2 + 0.0003)
        s_in[ax] = 0.0003
        s_in[1 - ax] = size[1 - ax] - t
        s_in[2] = size[2] - 0.0005
        ET.SubElement(bb, "geom", name=nm + "_in", type="box", size=_fmt3(s_in), pos=_fmt3(p_in), material="boxin",
                      contype="0", conaffinity="0", group="1")
    ET.SubElement(bb, "geom", name="bx_floor", type="box", size=_fmt3((lx / 2 - t, ly / 2 - t, 0.001)),
                  pos="0 0 0.001", material="boxin", contype="0", conaffinity="0", group="1")
    kb = ET.SubElement(wb, "body", name="block", pos="0 0.25 -1", mocap="true")
    ET.SubElement(kb, "geom", name="block_g", type="box", size=_fmt3((BLOCK / 2,) * 3), material="block",
                  contype="0", conaffinity="0", group="1")


def frame_setup(rig, fs, i) -> None:
    """place the box (per episode) and the block (per frame) from out/objects.json; hidden when unknown."""
    import mujoco

    m, d = rig.arm.m, rig.arm.d
    bb = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")]
    kb = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")]
    if OBJ is None:
        d.mocap_pos[bb] = (0, 0, -2)
        d.mocap_pos[kb] = (0, 0, -2)
        return
    e = str(int(fs.labels["episode"][i]))
    bx = OBJ["box"].get(e)
    if bx is None:
        d.mocap_pos[bb] = (0, 0, -2)
    else:
        d.mocap_pos[bb] = (bx[0], bx[1], 0.0)
        d.mocap_quat[bb] = A.quat_from_rotvec([0, 0, float(bx[2])])
    c = OBJ["block"].get(str(int(i)))
    if c is None:
        d.mocap_pos[kb] = (0, 0, -2)
    elif c[0] == "held":   # in the jaws: at the tool centre point of the posed arm
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        mujoco.mj_kinematics(m, d)
        d.mocap_pos[kb] = d.site_xpos[sid] + np.array([0.0, 0.0, 0.004])
        d.mocap_quat[kb] = A.quat_from_mat(d.site_xmat[sid].reshape(3, 3))
    else:
        d.mocap_pos[kb] = c[1]
        d.mocap_quat[kb] = A.quat_from_rotvec([0, 0, float(c[2]) if len(c) > 2 else 0.0])
    mujoco.mj_kinematics(m, d)


MASK_OCCLUDERS = False   # fit.py flags: the arm-only twin mask is never occluded by twin objects
CUP_W = 0.0
