"""Scene description: jar (data_so101/intheworld__so101_pick_and_place_2, LeRobot v3.0, so101_follower, normalised units).

Black SO-101 (white jaws) on a light-wood table; a white jar with a black inside (moved between episodes) and a
3x3 magic cube. Task "Grab the magic cube, and put it into the jar". Cameras (both static: 0 px phase-correlation
shift between the 40 episode medians, static_check.py):
  'top'  = observation.images.back  (overhead, looking down, the base at the bottom edge),
  'side' = observation.images.top   (low, from the front-left at table height; the base at the right edge).
World frame: the arm's normalised frame (arm.py): base bottom at z = 0 = table top, pan axis = world z, pan 0 reaches +y.
Joint units: LeRobot NORMALISED (-100..100 of the calibrated travel; meta/stats.json action max elbow_flex = +100.0 and
wrist_flex = +100.0 exactly, state wrist_flex max = +100.0), so q = deg2rad(scale * state + offset) per joint
(FIT_SCALES), gripper (0..100 of its travel): q = GRIP_LO + grip_off + grip_scale * g / 100 * (GRIP_HI - GRIP_LO).
Camera inits: seed.py (render search against the twin-independent DINO+SAM masks _d1c), refined by fit.py.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

import arm as A

HERE = pathlib.Path(__file__).resolve().parent
REAL_SET = "so101_jar_real"
BLACK = (0.05, 0.05, 0.055, 1)
WHITE = (0.80, 0.80, 0.78, 1)
ARM = A.ArmSpec(calib="new", model="so101", wrist_mount=False, base_pos=(0.0, 0.0, 0.0), base_yaw_deg=0.0,
                colors={"body": BLACK, "jaw": WHITE, "motor": (0.035, 0.035, 0.035, 1)})


def _seed(name, default):
    p = HERE / "out" / f"seed_{name}.json"
    if p.exists():
        j = json.loads(p.read_text())
        return dict(mount="world", pos=tuple(j["pos"]), quat=tuple(j["quat"]), fovy=float(j["fovy"]))
    return default


CAMERAS = {
    "top": _seed("top", dict(mount="world", pos=(0.0, 0.25, 0.60),
                             quat=tuple(A.look_quat((0.0, 0.25, 0.60), (0.0, 0.25, 0.0), up=(0, 1, 0))), fovy=50.0)),
    "side": _seed("side", dict(mount="world", pos=(-0.40, 0.20, 0.12),
                               quat=tuple(A.look_quat((-0.40, 0.20, 0.12), (0.0, 0.18, 0.08))), fovy=50.0)),
}
GROUP_RULE = {"top": "single pose (static camera: 0 px shift between episode medians)",
              "side": "single pose (static camera: 0 px shift between episode medians)"}
OFFSETS_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]
OFFSET_SIGMA = [8.0, 8.0, 8.0, 8.0, 12.0]
FIT_SCALES = True
SCALES_INIT = [1.1, 1.1, 1.1, 1.1, 1.0]
SCALES_PRIOR = [1.0, 1.0, 1.0, 1.0, 1.0]   # r2+: centred on degrees-like units (jseed: 0.91-1.04)
SCALE_SIGMA = [0.15, 0.15, 0.15, 0.15, 0.08]   # roll: state nearly constant (-3 +- 4 units), scale pinned near 1
JOINT_PARAMS = ("off", "grip", "gripoff")
ROLL_CANDIDATES = None

# ---------------------------------------------------------------------------------------------------- objects (RGB only)
# The metric never uses them (arm-only masks). Jar: white cylinder with a dark inside and a light rim, per episode;
# magic cube: 3x3 sticker box, per frame (table -> jaws -> jar). Placements come from objects.py (out/objects.json:
# twin-independent DINO+SAM object masks ray-cast through the FITTED cameras); table / wall / door planes from layout
# constants below (fitted by eye against the real frames through the fitted cameras, RGB only).
_OBJ_PATH = HERE / "out" / "objects.json"
OBJ = json.loads(_OBJ_PATH.read_text()) if _OBJ_PATH.exists() else None
JAR_R, JAR_H = 0.062, 0.115
CUBE_H = 0.057
LAYOUT = dict(wall=None, door=None)   # filled from objects.json when present
if OBJ is not None:
    JAR_R, JAR_H = OBJ.get("jar_r", JAR_R), OBJ.get("jar_h", JAR_H)
    LAYOUT = OBJ.get("layout", LAYOUT)


def extras(root, asset, wb) -> None:
    import xml.etree.ElementTree as ET

    ET.SubElement(asset, "texture", name="wood", type="2d", builtin="gradient", rgb1="0.66 0.58 0.50",
                  rgb2="0.60 0.52 0.44", width="64", height="64")
    ET.SubElement(asset, "material", name="table", rgba="0.66 0.57 0.48 1", specular="0.05", shininess="0.1")
    ET.SubElement(asset, "material", name="wall", rgba="0.86 0.88 0.87 1", specular="0.0", shininess="0.0")
    ET.SubElement(asset, "material", name="door", rgba="0.55 0.47 0.40 1", specular="0.0", shininess="0.0")
    ET.SubElement(asset, "material", name="jarbody", rgba="0.90 0.89 0.85 1", specular="0.3", shininess="0.4")
    ET.SubElement(asset, "material", name="jarin", rgba="0.06 0.06 0.07 1", specular="0.1", shininess="0.2")
    ET.SubElement(asset, "material", name="jarrim", rgba="0.80 0.70 0.62 1", specular="0.3", shininess="0.4")
    for nm, rgba in (("c_r", "0.85 0.20 0.18 1"), ("c_g", "0.20 0.65 0.30 1"), ("c_b", "0.20 0.35 0.80 1"),
                     ("c_y", "0.92 0.85 0.30 1"), ("c_o", "0.95 0.55 0.25 1"), ("c_w", "0.92 0.92 0.90 1"),
                     ("c_k", "0.08 0.08 0.08 1")):
        ET.SubElement(asset, "material", name=nm, rgba=rgba, specular="0.2", shininess="0.3")
    ET.SubElement(wb, "light", name="key", pos="0.3 -0.2 1.4", dir="-0.2 0.15 -1", diffuse="0.55 0.55 0.55",
                  specular="0.1 0.1 0.1", castshadow="true")
    ET.SubElement(wb, "geom", name="table", type="box", size="0.9 0.9 0.01", pos="0 0.25 -0.01", material="table",
                  contype="0", conaffinity="0", group="1")
    lay = LAYOUT or {}
    if lay.get("wall"):
        w = lay["wall"]   # dict(pos, quat, size): vertical plane behind the work area (side view backdrop)
        ET.SubElement(wb, "geom", name="wall", type="box", size=_fmt3(w["size"]), pos=_fmt3(w["pos"]),
                      quat=_fmt3(w["quat"]), material="wall", contype="0", conaffinity="0", group="1")
    if lay.get("back"):
        w = lay["back"]
        ET.SubElement(wb, "geom", name="back", type="box", size=_fmt3(w["size"]), pos=_fmt3(w["pos"]),
                      quat=_fmt3(w["quat"]), material="door", contype="0", conaffinity="0", group="1")
    if lay.get("door"):
        w = lay["door"]
        ET.SubElement(wb, "geom", name="door", type="box", size=_fmt3(w["size"]), pos=_fmt3(w["pos"]),
                      quat=_fmt3(w["quat"]), material="door", contype="0", conaffinity="0", group="1")
    jb = ET.SubElement(wb, "body", name="jar", pos="0.15 0.35 0", mocap="true")
    ET.SubElement(jb, "geom", name="jar_body", type="cylinder", size=f"{JAR_R:.4f} {JAR_H / 2:.4f}",
                  pos=f"0 0 {JAR_H / 2:.4f}", material="jarbody", contype="0", conaffinity="0", group="1")
    ET.SubElement(jb, "geom", name="jar_rim", type="cylinder", size=f"{JAR_R * 0.97:.4f} 0.0012",
                  pos=f"0 0 {JAR_H + 0.0006:.4f}", material="jarrim", contype="0", conaffinity="0", group="1")
    ET.SubElement(jb, "geom", name="jar_in", type="cylinder", size=f"{JAR_R * 0.86:.4f} 0.0012",
                  pos=f"0 0 {JAR_H + 0.0016:.4f}", material="jarin", contype="0", conaffinity="0", group="1")
    cb = ET.SubElement(wb, "body", name="cube", pos="0 0.25 -1", mocap="true")
    h = CUBE_H / 2
    ET.SubElement(cb, "geom", name="cube_core", type="box", size=f"{h:.4f} {h:.4f} {h:.4f}", material="c_k",
                  contype="0", conaffinity="0", group="1")
    cols = ["c_r", "c_g", "c_b", "c_y", "c_o", "c_w"]
    k = 0
    s = CUBE_H / 3
    for ax in range(3):
        for sgn in (-1, 1):
            for a_ in range(3):
                for b_ in range(3):
                    p = [0.0, 0.0, 0.0]
                    p[ax] = sgn * (h + 0.0004)
                    o = [i for i in range(3) if i != ax]
                    p[o[0]] = (a_ - 1) * s
                    p[o[1]] = (b_ - 1) * s
                    sz = [0.44 * s, 0.44 * s, 0.44 * s]
                    sz[ax] = 0.0005
                    ET.SubElement(cb, "geom", type="box", size=_fmt3(sz), pos=_fmt3(p),
                                  material=cols[(k * 7 + a_ * 3 + b_ * 5) % 6], contype="0", conaffinity="0",
                                  group="1")
            k += 1


def _fmt3(v):
    return " ".join(f"{float(x):.5g}" for x in v)


def frame_setup(rig, fs, i) -> None:
    """place the jar (per episode) and the cube (per frame) from out/objects.json; hidden when unknown."""
    import mujoco

    m, d = rig.arm.m, rig.arm.d
    jb = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jar")]
    cb = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")]
    if OBJ is None:
        d.mocap_pos[jb] = (0, 0, -2)
        d.mocap_pos[cb] = (0, 0, -2)
        return
    e = str(int(fs.labels["episode"][i]))
    jxy = OBJ["jar_xy"].get(e)
    d.mocap_pos[jb] = (jxy[0], jxy[1], 0.0) if jxy is not None else (0, 0, -2)
    c = OBJ["cube"].get(str(int(i)))
    if c is None:
        d.mocap_pos[cb] = (0, 0, -2)
    elif c[0] == "held":   # in the jaws: at the tool centre point of the posed arm
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        mujoco.mj_kinematics(m, d)
        d.mocap_pos[cb] = d.site_xpos[sid] + np.array([0.0, 0.0, 0.004])
        d.mocap_quat[cb] = A.quat_from_mat(d.site_xmat[sid].reshape(3, 3))
    else:
        d.mocap_pos[cb] = c[1]
        d.mocap_quat[cb] = A.quat_from_rotvec([0, 0, float(c[2]) if len(c) > 2 else 0.0])
    mujoco.mj_kinematics(m, d)

MASK_OCCLUDERS = False   # fit.py (ballcup copy) flags: the arm-only twin mask is never occluded by twin objects
CUP_W = 0.0
