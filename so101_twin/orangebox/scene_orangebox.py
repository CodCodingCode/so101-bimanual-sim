"""Scene description: orangebox (data_so101/AbdullahRasul__so101_blue_black_pick_place, LeRobot v3.0, so101_follower).

Orange SO-101 on a white table, a black open box (moved between episodes), a yellow-green block. Cameras: "top" =
observation.images.top (static overhead), "side" = observation.images.side (static, across the table; it moved by
~(4, 5) px between episodes 0-9 and 10-48, so it gets one pose per group). The dataset's "front" camera is on the
wrist and is not used.

World frame: the arm's normalised frame (arm.py): base bottom at z = 0 = table top, pan axis = z axis, pan 0 reaches +y.
Joint units: LeRobot NORMALISED (-100..100 of the calibrated travel; the dataset's action stats hit exactly -100.0 on
shoulder_lift and +100.0 on elbow_flex), so q = deg2rad(scale * state + offset) per joint (FIT_SCALES), gripper
(0..100 of its travel): q = GRIP_LO + grip_off + grip_scale * g / 100 * (GRIP_HI - GRIP_LO).
Camera inits: seed_search.py (twin-free search against orange hue masks), refined by fit.py.
"""
from __future__ import annotations

import json
import math
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import arm as A

HERE = pathlib.Path(__file__).resolve().parent
REAL_SET = "so101_orangebox_real"
# The real arm's wrist camera sits in a compact printed frame at the jaw root (checked on close-ups), not on the official
# side bracket of so101_new_calib_camera.xml (which sticks out ~6 cm): a 38 x 38 x 12 mm board box on the gripper body
# with a fitted pose replaces the official bracket + module (arm.py cam_board).
ARM = A.ArmSpec(calib="new", model="so101", wrist_mount=False, base_pos=(0.0, 0.0, 0.0), base_yaw_deg=0.0,
                cam_board=(0.019, 0.019, 0.006), cam_board_pose=(0.0, 0.042, -0.04, 1.0, 0.0, 0.0, 0.0),
                colors={"body": (0.95, 0.40, 0.06, 1), "jaw": (0.95, 0.40, 0.06, 1), "motor": (0.06, 0.06, 0.06, 1),
                        "cam_mount": (0.95, 0.40, 0.06, 1), "camera": (0.05, 0.05, 0.05, 1)})


def _seed(name, fallback):
    p = HERE / "out" / f"seed_{name}.json"
    if p.exists():
        j = json.loads(p.read_text())
        return dict(pos=tuple(j["pos"]), quat=tuple(j["quat"]), fovy=float(j["fovy"]))
    return fallback


_TOP = _seed("top", dict(pos=(-0.05, 0.28, 0.46), quat=tuple(A.look_quat((-0.05, 0.28, 0.46), (0, 0.28, 0), up=(0, -1, 0))), fovy=45.0))
_SIDE = _seed("side_e10-48", dict(pos=(0.4, 0.2, 0.3), quat=tuple(A.look_quat((0.4, 0.2, 0.3), (0, 0.2, 0))), fovy=55.0))
CAMERAS = {
    "top": dict(mount="world", pos=_TOP["pos"], quat=_TOP["quat"], fovy=_TOP["fovy"]),
    "side": dict(mount="world", pos=_SIDE["pos"], quat=_SIDE["quat"], fovy=_SIDE["fovy"],
                 group=lambda ep: "e00-09" if ep <= 9 else "e10-48"),
}
GROUP_RULE = {"top": "single pose (static: 0 px shift between episode medians)",
              "side": "episodes 0-9 -> 'e00-09', 10-48 -> 'e10-48' (phase correlation of the static background: "
                      "(4, 4..6) px shift between the two groups, 0 px within each)"}
OFFSETS_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]
OFFSET_SIGMA = [6.0, 6.0, 6.0, 6.0, 20.0]
FIT_SCALES = True
SCALES_INIT = [1.0, 1.0, 1.0, 1.0, 1.0]
SCALES_PRIOR = [1.05, 1.05, 1.05, 1.05, 1.0]
SCALE_SIGMA = [0.15, 0.15, 0.15, 0.15, 0.05]   # roll: state ~constant (-55 +- 1.6), so its scale is pinned near 1
JOINT_PARAMS = ("off", "grip", "gripoff", "board", "bsize", "comp")
# wrist roll: the state is ~constant (-55 +- 1.6 units); the offset was settled by the twin-free joint seed (CMA from
# offsets 0 and -30 both converge to about -36..-39 deg, i.e. a roll angle of about -91..-94 deg), so no candidate grid.
ROLL_CANDIDATES = None
BOARD_SIGMA = (0.03, 3.0)   # camera-board offset prior (m, rad): its pose comes from a global search (board_search.py)

# ------------------------------------------------------------------ objects (RGB render only)
BOX_H = 0.085             # box height (m), outer half sizes from the top-view detection
BOX_WALL = 0.004
CUBE_HALF = (0.018, 0.030, 0.018)
_OBJ = None


def _ray(P, u, v, z0):
    """world point where the top camera's ray through full-res pixel (u, v) meets the plane z = z0."""
    cp = P["cams"]["top"]
    g = next(iter(cp["groups"].values()))
    L = cp["lens"]
    f = 0.5 * 480 / math.tan(math.radians(cp["fovy"]) / 2)
    xd = (u + 0.5 - 320 - L["cx"] * 640) / f
    yd = (v + 0.5 - 240 - L["cy"] * 480) / f
    xu, yu = xd, yd
    for _ in range(10):
        r2 = xu * xu + yu * yu
        s = 1 + L["k1"] * r2 + L["k2"] * r2 * r2
        xu, yu = xd / s, yd / s
    R = A.mat_from_quat(np.asarray(g["quat"]) / np.linalg.norm(g["quat"]))
    d = R @ np.array([xu, -yu, -1.0])
    o = np.asarray(g["pos"], float)
    t = (z0 - o[2]) / d[2]
    return o + t * d


def objects(P):
    """per-episode box pose and per-frame cube positions (world) from out/objects_px.json and the fitted top camera."""
    global _OBJ
    if _OBJ is not None and _OBJ[0] is P:
        return _OBJ[1]
    px = json.loads((HERE / "out" / "objects_px.json").read_text())
    box = {}
    for e, b in px["box"].items():
        c = _ray(P, *b["center"], BOX_H * 0.6)
        cor = np.array([_ray(P, x, y, BOX_H) for x, y in b["corners"]])
        ex = cor[1] - cor[0]
        yaw = math.atan2(ex[1], ex[0])
        half = (0.5 * np.linalg.norm(cor[1] - cor[0]), 0.5 * np.linalg.norm(cor[2] - cor[1]))
        box[int(e)] = dict(pos=c[:2].tolist(), yaw=yaw, half=half)
    cube = {}
    for i, q in px["cube"].items():
        cube[int(i)] = dict(table=_ray(P, *q["center"], CUBE_HALF[2]).tolist(), px=q["px"],
                            angle=math.radians(q["angle"]))
    _OBJ = (P, dict(box=box, cube=cube))
    return _OBJ[1]


def extras(root: ET.Element, asset: ET.Element, wb: ET.Element) -> None:
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="flat", rgb1="0.75 0.75 0.75",
                  rgb2="0.75 0.75 0.75", width="8", height="8")
    ET.SubElement(asset, "material", name="table", rgba="0.86 0.87 0.88 1", specular="0.05", shininess="0.1")
    ET.SubElement(asset, "material", name="floor", rgba="0.62 0.55 0.48 1", specular="0.0")
    ET.SubElement(asset, "material", name="boxmat", rgba="0.06 0.06 0.065 1", specular="0.3", shininess="0.4")
    ET.SubElement(asset, "material", name="cubemat", rgba="0.80 0.85 0.15 1", specular="0.2", shininess="0.3")
    ET.SubElement(wb, "light", name="key", pos="0.1 0.3 1.5", dir="0 0 -1", directional="false",
                  diffuse="0.45 0.45 0.45", specular="0.1 0.1 0.1", castshadow="true", cutoff="80")
    ET.SubElement(wb, "light", name="fill", pos="0.8 -0.3 0.9", dir="-0.6 0.3 -0.7", diffuse="0.18 0.18 0.18",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="table_top", type="box", size="0.45 0.40 0.01", pos="0.0 0.25 -0.01",
                  material="table", contype="0", conaffinity="0")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.75", material="floor")
    b = ET.SubElement(wb, "body", name="box", pos="0.2 0.25 0", mocap="true")
    hx, hy, h, w = 0.08, 0.08, BOX_H, BOX_WALL
    for k, (p, s) in enumerate([((0, 0, w / 2), (hx, hy, w / 2)), ((hx - w / 2, 0, h / 2), (w / 2, hy, h / 2)),
                                ((-hx + w / 2, 0, h / 2), (w / 2, hy, h / 2)), ((0, hy - w / 2, h / 2), (hx, w / 2, h / 2)),
                                ((0, -hy + w / 2, h / 2), (hx, w / 2, h / 2))]):
        ET.SubElement(b, "geom", name=f"box{k}", type="box", size=" ".join(map(str, s)), pos=" ".join(map(str, p)),
                      material="boxmat", contype="0", conaffinity="0")
    c = ET.SubElement(wb, "body", name="cube", pos="0 0.25 0.02", mocap="true")
    ET.SubElement(c, "geom", name="cube", type="box", size=" ".join(map(str, CUBE_HALF)), material="cubemat",
                  contype="0", conaffinity="0")


def frame_setup(rig, fs, i: int) -> None:
    """box at its episode's detected pose (scaled to the detected size); cube: in the jaws while the gripper is closed
    in grasp/release, else at its top-view detection (table, or the box floor inside the box), else the episode's
    first detection (table) / the box centre (release)."""
    m, d = rig.arm.m, rig.arm.d
    O = objects(rig.p)
    ep = int(fs.labels["episode"][i])
    phase = str(fs.labels["phase"][i])
    bx = O["box"].get(ep)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
    if bx is not None:
        d.mocap_pos[m.body_mocapid[bid]] = (bx["pos"][0], bx["pos"][1], 0.0)
        d.mocap_quat[m.body_mocapid[bid]] = A.quat_from_rotvec([0, 0, bx["yaw"]])
        sx, sy = bx["half"][0] / 0.08, bx["half"][1] / 0.08
        for k in range(5):   # scale the walls to the detected outer size (model built for 0.08 half sizes)
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"box{k}")
            if not hasattr(rig, "_box0"):
                rig._box0 = {}
            if k not in rig._box0:
                rig._box0[k] = (m.geom_pos[gid].copy(), m.geom_size[gid].copy())
            p0, s0 = rig._box0[k]
            m.geom_pos[gid] = (p0[0] * sx, p0[1] * sy, p0[2])
            m.geom_size[gid] = (s0[0] * (sx if k in (0, 3, 4) else 1), s0[1] * (sy if k in (0, 1, 2) else 1), s0[2])
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    grip = float(fs.state[i][5])
    closed = grip < 20 and phase in ("grasp", "release")
    if closed:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        pos = d.site_xpos[sid].copy()
        quat = A.quat_from_mat(d.xmat[gb].reshape(3, 3))
    else:
        c = O["cube"].get(i)
        if c is not None and c["px"] > 600:
            pos = np.asarray(c["table"])
            if bx is not None and abs(pos[0] - bx["pos"][0]) < bx["half"][0] and abs(pos[1] - bx["pos"][1]) < bx["half"][1]:
                pos = _ray(rig.p, *json.loads((HERE / "out" / "objects_px.json").read_text())["cube"][str(i)]["center"],
                           BOX_WALL + CUBE_HALF[2])
            quat = A.quat_from_rotvec([0, 0, c["angle"]])
        else:
            first = [j for j in np.nonzero(np.asarray(fs.labels["episode"]) == ep)[0] if j in O["cube"]]
            if phase == "release" and bx is not None:
                pos = np.array([bx["pos"][0], bx["pos"][1], BOX_WALL + CUBE_HALF[2]])
            elif first:
                pos = np.asarray(O["cube"][int(first[0])]["table"])
            else:
                pos = np.array([0.1, 0.25, CUBE_HALF[2]])
            quat = np.array([1.0, 0, 0, 0])
    d.mocap_pos[m.body_mocapid[cid]] = pos
    d.mocap_quat[m.body_mocapid[cid]] = quat
    mujoco.mj_forward(m, d)
