"""Scene description: ballcup (data_so101/ReubenLim__so101_ball_in_cup, LeRobot v2.1, old 'main_*' calibration).

World frame (this scene's own; there is no builder twin): table top z = 0, the SO-101 pan axis is the world z axis
through the origin, the arm reaches +y at shoulder_pan = 0. 'top' = observation.images.overhead (fixed, above the
table looking down, base at the bottom edge); 'side' = observation.images.front (fixed, in front of the arm near
table height, looking back at it; the cardboard TV box is the backdrop).
Camera inits = cam_search.py results (seeded from background-subtraction masks, no SAM), see out/cam_search_*.json.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

import arm as A

HERE = pathlib.Path(__file__).resolve().parent
REAL_SET = "so101_ballcup_real"
BLACK = (0.06, 0.06, 0.065, 1)
ARM = A.ArmSpec(calib="old", model="so101", wrist_mount=False, base_pos=(0.0, 0.0, 0.0), base_yaw_deg=0.0,
                colors={"body": BLACK, "jaw": BLACK, "motor": (0.04, 0.04, 0.04, 1)})


def _cam(name, default):
    p = HERE / "out" / f"cam_search_{name}.json"
    if p.exists():
        j = json.loads(p.read_text())
        return dict(mount="world", pos=tuple(j["pos"]), quat=tuple(j["quat"]), fovy=float(j["fovy"]))
    return default


CAMERAS = {
    "top": _cam("top", dict(mount="world", pos=(0.0, 0.2, 0.6), quat=tuple(A.look_quat((0.0, 0.2, 0.6), (0, 0.22, 0))),
                            fovy=50.0)),
    "side": _cam("side", dict(mount="world", pos=(0.0, 0.55, 0.15),
                              quat=tuple(A.look_quat((0.0, 0.55, 0.15), (0, 0.05, 0.12))), fovy=50.0)),
}
GROUP_RULE = {"top": "single pose (static camera)", "side": "single pose (static camera)"}
OFFSETS_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]
OFFSET_SIGMA = [4.0, 4.0, 4.0, 4.0, 8.0]
JOINT_PARAMS = ("off", "grip")

# ---------------------------------------------------------------------------------------------------- objects
# Objects are placed from the twin-independent DINO+SAM object masks (out/obj_pixels.json, obj_pixels.py) through the
# CURRENT fitted cameras (update_objects, called by fit.Rig.set_params): cup = frustum (12 stacked cylinder slices so
# its shape can change at runtime; params P["cup"] = r_top, r_bot, h, zc), per episode at the top-view cup-mask
# centroid ray hitting z = zc * h; ball = sphere, per frame triangulated from the two views' ball centroids (rays
# within 4 cm), else top view onto z = R (on the table) or the gripper tool point if that is within 6 cm (held),
# else hidden. With CUP_W > 0 the fit also scores the twin cup against the SAM cup masks (a depth cue for both
# cameras: the arm alone leaves camera distance vs fovy weakly determined).
_PIX_PATH = HERE / "out" / "obj_pixels.json"
PIX = json.loads(_PIX_PATH.read_text()) if _PIX_PATH.exists() else None
MASK_OCCLUDERS = False          # fit.py --occ: cup / ball hide the arm pixels behind them in the twin's arm mask
CUP_W = 0.0                     # fit.py --cup-w: weight of the cup IoU term in the fit objective
CUP_SLICES = 12
CUP_INIT = dict(r_top=0.046, r_bot=0.033, h=0.105, zc=0.85)
BALL_R = 0.030           # objects: fitted from the ball-mask areas (front 2.9, top 3.1 cm) and the front IoU scan
CUP_RGBA = "0.86 0.80 0.68 1"
BALL_RGBA = "0.55 0.75 0.95 1"


def occluders(root, asset, wb) -> None:
    """cup (12 cylinder slices + inside disc) and ball, mocap bodies; shape set at runtime by apply_cup_shape."""
    import xml.etree.ElementTree as ET

    ET.SubElement(asset, "material", name="cupmat", rgba=CUP_RGBA, specular="0.1", shininess="0.2")
    ET.SubElement(asset, "material", name="cupin", rgba="0.70 0.64 0.52 1", specular="0.0", shininess="0.1")
    ET.SubElement(asset, "material", name="ballmat", rgba=BALL_RGBA, specular="0.3", shininess="0.5")
    b = ET.SubElement(wb, "body", name="cup", pos="0 0.3 0", mocap="true")
    for k in range(CUP_SLICES):
        ET.SubElement(b, "geom", name=f"cup_s{k}", type="cylinder", size="0.04 0.005", pos=f"0 0 {0.01 * k + 0.005}",
                      material="cupmat", contype="0", conaffinity="0", group="2")
    ET.SubElement(b, "geom", name="cup_inside", type="cylinder", size="0.04 0.0005", pos="0 0 0.1",
                  material="cupin", contype="0", conaffinity="0", group="2")
    bb = ET.SubElement(wb, "body", name="ball", pos="0 0.3 -1", mocap="true")
    ET.SubElement(bb, "geom", name="ball", type="sphere", size=f"{BALL_R:.4f}", material="ballmat",
                  contype="0", conaffinity="0", group="2")


def cup_geom_ids(m):
    import mujoco

    return [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"cup_s{k}") for k in range(CUP_SLICES)] + \
        [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_inside")]


def apply_cup_shape(m, cup) -> None:
    ids = cup_geom_ids(m)
    rt, rb, h = float(cup["r_top"]), float(cup["r_bot"]), float(cup["h"])
    for k in range(CUP_SLICES):
        f = (k + 0.5) / CUP_SLICES
        m.geom_size[ids[k]] = [rb + (rt - rb) * f, 0.5 * h / CUP_SLICES, 0.0]
        m.geom_pos[ids[k]] = [0.0, 0.0, h * f]
    m.geom_size[ids[-1]] = [0.93 * rt, 0.0005, 0.0]
    m.geom_pos[ids[-1]] = [0.0, 0.0, h + 0.0006]


def update_objects(rig, P) -> None:
    """per-episode cup xy and per-frame ball position for the current cameras -> rig.obj"""
    from geom import cam_ray, closest, hit_z

    cup = P.get("cup", CUP_INIT)
    apply_cup_shape(rig.arm.m, cup)
    eps = sorted(int(e) for e in PIX["episodes"])
    zc = float(cup["zc"]) * float(cup["h"])
    cups = {}
    for e in eps:
        px = PIX["episodes"][str(e)]["cup_top_px"]
        if px is None:
            continue
        o, d = cam_ray(P, "top", *px)
        cups[e] = hit_z(o, d, zc)[:2]
    balls = {}
    for i, fr in PIX["frames"].items():
        bt, bs = fr["ball_top"], fr["ball_side"]
        pos = None
        if bt["px"] is not None and bt["area"] > 80 and bs["px"] is not None and bs["area"] > 80:
            p, dist = closest(*cam_ray(P, "top", *bt["px"]), *cam_ray(P, "side", *bs["px"]))
            if dist < 0.04 and -0.02 < p[2] < 0.5:
                pos = p
        if pos is None and bt["px"] is not None and bt["area"] > 80:
            pos = ("top", hit_z(*cam_ray(P, "top", *bt["px"]), BALL_R))
        balls[int(i)] = pos
    rig.obj = dict(cup=cups, ball=balls, cup_r=float(cup["r_top"]))


def place_objects(rig, i: int) -> None:
    """cup at its episode position, ball at its frame position (hidden below the table if not seen)."""
    import mujoco

    m, d = rig.arm.m, rig.arm.d
    ob = getattr(rig, "obj", None)
    if ob is None:
        return
    e = _episode_of(i)
    xy = ob["cup"].get(e)
    if xy is None and ob["cup"]:
        xy = np.median(np.stack(list(ob["cup"].values())), 0)
    _set_mocap(m, d, "cup", [xy[0], xy[1], 0.0])
    b = ob["ball"].get(int(i))
    if isinstance(b, tuple):   # top view only: in the cup, held (tool point close), or on the table
        tcp = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")]
        p = b[1]
        if xy is not None and np.linalg.norm(p[:2] - np.asarray(xy)) < ob.get("cup_r", 0.045):
            b = np.array([xy[0], xy[1], BALL_R + 0.01])    # seen from above only, inside the rim: in the cup
        elif np.linalg.norm(p[:2] - tcp[:2]) < 0.06 and tcp[2] > 2 * BALL_R:
            b = tcp.copy()
        else:
            b = p
    if b is not None:
        b = np.array([b[0], b[1], max(float(b[2]), BALL_R)])
    _set_mocap(m, d, "ball", b if b is not None else [0.0, 0.3, -1.0])
    mujoco.mj_forward(m, d)


def extras(root, asset, wb) -> None:
    """full scene for the RGB render: table, dark mat around the base, cardboard backdrop, lights, cup, ball."""
    import xml.etree.ElementTree as ET

    ET.SubElement(asset, "material", name="table", rgba="0.86 0.88 0.90 1", specular="0.05", shininess="0.1")
    ET.SubElement(asset, "material", name="mat", rgba="0.07 0.13 0.12 1", specular="0.2", shininess="0.4")
    ET.SubElement(asset, "material", name="cardboard", rgba="0.72 0.62 0.48 1", specular="0.0", shininess="0.0")
    ET.SubElement(asset, "material", name="floor", rgba="0.45 0.42 0.38 1")
    ET.SubElement(wb, "light", name="key", pos="0.1 0.35 1.6", dir="0 -0.1 -1", directional="false",
                  diffuse="0.45 0.45 0.45", specular="0.1 0.1 0.1", castshadow="true", cutoff="90")
    ET.SubElement(wb, "light", name="fill", pos="0.0 1.2 0.6", dir="0 -0.8 -0.4", diffuse="0.2 0.2 0.2",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.75", material="floor")
    ET.SubElement(wb, "geom", name="table_top", type="box", size=" ".join(map(str, TABLE_HALF)),
                  pos=f"{TABLE_CENTER[0]} {TABLE_CENTER[1]} {-TABLE_HALF[2]}", material="table")
    import math

    # mat: its front edge passes through MAT_EDGE_MID with direction MAT_YAW (from the top image), extends backwards
    c, s_ = math.cos(math.radians(MAT_YAW)), math.sin(math.radians(MAT_YAW))
    n = (-s_, c) if (-s_ * 0 + c * 1) > 0 else (s_, -c)          # edge normal pointing to +y (away from the base side)
    mc = (MAT_EDGE_MID[0] - n[0] * MAT_HALF[1], MAT_EDGE_MID[1] - n[1] * MAT_HALF[1])
    ET.SubElement(wb, "geom", name="mat", type="box", size=" ".join(map(str, MAT_HALF)),
                  pos=f"{mc[0]} {mc[1]} 0.0005", euler=f"0 0 {math.radians(MAT_YAW):.5f}", material="mat")
    ET.SubElement(wb, "geom", name="backdrop", type="box", size=" ".join(map(str, BACK_HALF)),
                  pos=" ".join(map(str, BACK_CENTER)), euler=f"0 0 {math.radians(BACK_YAW):.5f}", material="cardboard")
    occluders(root, asset, wb)


TABLE_HALF = (1.5, 1.5, 0.02)
TABLE_CENTER = (0.0, 0.3)
MAT_HALF = (1.0, 0.3, 0.0005)
MAT_CENTER = (0.0, -0.05)
MAT_EDGE_MID = (0.0, 0.07)
MAT_YAW = 0.0
BACK_HALF = (1.5, 0.01, 0.6)
BACK_CENTER = (0.0, -0.23, 0.5)
BACK_YAW = 0.0
_LAY = HERE / "out" / "layout.json"   # layout.py: mat / backdrop from the fitted cameras (RGB only)
if _LAY.exists():
    _l = json.loads(_LAY.read_text())
    MAT_HALF, BACK_CENTER = tuple(_l["MAT_HALF"]), tuple(_l["BACK_CENTER"])
    MAT_EDGE_MID, MAT_YAW, BACK_YAW = tuple(_l["mat_edge_mid"]), float(_l["mat_yaw_deg"]), float(_l["back_yaw_deg"])


def _set_mocap(m, d, body, pos):
    import mujoco

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    d.mocap_pos[m.body_mocapid[bid]] = pos


_EP = None


def _episode_of(i: int) -> int:
    global _EP
    if _EP is None:
        import sys

        sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))
        from sets import Store

        _EP = np.asarray(Store().frames(REAL_SET).labels["episode"]).astype(int)
    return int(_EP[i])


def frame_setup(rig, fs, i: int) -> None:
    place_objects(rig, i)


OBJ_TRIM = 0.1        # objective = mean IoU of the best 90 % of fit frames (a few masks are broken)
