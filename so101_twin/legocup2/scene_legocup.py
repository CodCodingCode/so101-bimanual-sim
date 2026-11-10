"""Scene description: legocup (data_so101/Rayenghali__so101_follower_main_put_the_red_lego_block_in_the_black_cup_eps100_fps30_480x640),
official SO-101 new_calib + wrist-camera mount (so101_new_calib_camera.xml), black print.

World frame = the builder twin's (so101_twin/legocup/build_scene.py, used read-only): table top z = 0, arm pan axis at
(0, 0.0452), arm reaching +y at pan 0. 'top' = fixed overhead camera in front of the robot looking back and down;
'side' = the WRIST camera (dataset observation.images.secondary_0), mounted on the gripper body (mount 'gripper').
Camera inits = the builder's fitted cameras ported with so101_twin/common/port_scene.py legocup --calib new.
Objects: the builder's per-episode cup / brick placement (so101_twin/legocup/out/objects.json), re-ray-cast through the
refitted top camera when PLACEMENT_FROM_PIXELS is set (see place_objects() below).
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import arm as A

TWIN = pathlib.Path(__file__).resolve().parents[1]
_SRC = TWIN / "legocup" / "build_scene.py"
_spec = importlib.util.spec_from_file_location("legocup_build_scene_ro", _SRC)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
OBJECTS = json.loads((TWIN / "legocup" / "out" / "objects.json").read_text())

REAL_SET = "so101_legocup_real"
BLACK = (0.07, 0.07, 0.075, 1)
ARM = A.ArmSpec(calib="new", wrist_mount=True, base_pos=(0.0, 0.0452, 0.0), base_yaw_deg=0.0,
                colors={"body": BLACK, "jaw": BLACK, "motor": (0.05, 0.05, 0.05, 1), "cam_mount": BLACK,
                        "camera": (0.10, 0.10, 0.10, 1)})

CAMERAS = {
    "top": dict(mount="world", pos=(-0.0421, 0.466, 0.4687), quat=(-0.091007, 0.002164, 0.186426, 0.978243), fovy=35.04),
    # wrist camera: in the gripper body frame; it sits at the camera module, so the module/bracket (geom group 3)
    # are hidden from it
    "side": dict(mount="gripper", pos=(-0.00969, 0.07703, -0.0116), quat=(-0.052739, -0.015526, 0.220388, 0.973862),
                 fovy=59.86, hide_groups=(3,), pad=0.6),
}
GROUP_RULE = {"top": "single pose", "side": "single pose (rigid on the gripper body)"}
# Joint map: official new_calib q = deg2rad(state) (arm.py), plus these offsets (deg). Wrist roll -90: checked visually
# (out/check_roll2.jpg): only with roll -90 does the wrist-camera module face the top camera, centred on the gripper
# with the jaws below it, as in every real top frame (0 / 180 put the module out to the side, +90 behind the gripper).
OFFSETS_INIT = [0.0, 0.0, 0.0, 0.0, -90.0]
OFFSET_SIGMA = [5.0, 5.0, 5.0, 5.0, 8.0]
JOINT_PARAMS = ("off", "grip", "gripoff")
# gripper: q = -0.1745 + grip_off + grip_scale * gripper/100 * 1.919 rad. The wrist-camera jaw silhouettes want
# grip_scale ~2.1 (median IoU 0.54 at 1.0, 0.78 at 2.13 = the builder's fitted jaw gain with the same formula)
GRIP_SCALE_INIT = 2.1
# The dataset's arm joints are LeRobot NORMALISED units (-100..100 of each joint's calibrated travel, use_degrees=False):
# the action stats hit exactly -100.0 (shoulder_lift) and 100.0 (wrist_flex). So q_deg = scale * state + offset with
# a per-joint scale (deg per unit = travel / 200) fitted around 1 with a loose prior (sd 0.25).
FIT_SCALES = True
SCALE_SIGMA = [0.25, 0.25, 0.25, 0.25, 0.25]
OBJ_TRIM = 0.1        # objective = mean IoU of the best 90 % of fit frames (a few SAM masks are broken)


def extras(root: ET.Element, asset: ET.Element, wb: ET.Element) -> None:
    for name, kw in B.MATERIALS.items():
        if not name.startswith("arm_"):
            ET.SubElement(asset, "material", name=name, **kw)
    ET.SubElement(wb, "light", name="sun", pos="0.2 0.5 1.5", dir="-0.1 -0.2 -1", directional="false",
                  diffuse="0.45 0.45 0.45", specular="0.1 0.1 0.1", castshadow="true", cutoff="90")
    ET.SubElement(wb, "light", name="fill", pos="-0.8 0.6 1.0", dir="0.6 -0.3 -0.8", diffuse="0.15 0.15 0.15",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.74", material="floor")
    ET.SubElement(wb, "geom", name="table_top", type="box", size=" ".join(map(str, B.TABLE_HALF)),
                  pos=f"{B.TABLE_CENTER[0]} {B.TABLE_CENTER[1]} {-B.TABLE_HALF[2]}", material="table_dark")
    ET.SubElement(wb, "geom", name="mat", type="box", size=" ".join(map(str, B.MAT_HALF)),
                  pos=f"{B.MAT_CENTER[0]} {B.MAT_CENTER[1]} {B.MAT_HALF[2]}",
                  euler=f"0 0 {math.radians(B.MAT_YAW_DEG):.5f}", material="mat_grey")
    ET.SubElement(wb, "geom", name="wall", type="box", size="2.0 0.01 1.2", pos="0 -0.45 0.46", material="wall")
    cup = ET.SubElement(wb, "body", name="cup", pos=f"{B.CUP['xy'][0]} {B.CUP['xy'][1]} 0", mocap="true")
    B.cup_geoms(cup)
    blk = ET.SubElement(wb, "body", name="block", pos=f"{B.BLOCK['xy'][0]} {B.BLOCK['xy'][1]} {B.BLOCK['half'][2]}",
                        mocap="true")
    B.block_geoms(blk)


# per-episode object positions: {episode: (cup_xy, block_xy)}; place_objects() may replace them (re-ray-cast)
PLACEMENT = {int(k): (np.array(v["cup"], float), np.array(v["block"], float)) for k, v in OBJECTS.items()}


def _set_mocap(m, d, body, pos, quat=None):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    k = m.body_mocapid[bid]
    d.mocap_pos[k] = pos
    if quat is not None:
        d.mocap_quat[k] = quat


FIXED_TIP = np.array([-0.0112, -0.0003, -0.0986])    # fixed-jaw tip centroid, gripper body frame (official mesh)
MOVING_TIP = np.array([-0.0089, -0.0771, 0.0189])    # moving-jaw tip centroid, moving_jaw_so101_v1 body frame


def frame_setup(rig, fs, i: int) -> None:
    """Cup at the episode's position; brick on the table (approach), between the jaws (grasp / transfer_open:
    midway between the jaw tips, oriented with the gripper) or in the cup (release), as the builder's replay.py."""
    m, d = rig.arm.m, rig.arm.d
    ep = int(fs.labels["episode"][i])
    phase = str(fs.labels["phase"][i])
    cup_xy, blk_xy = PLACEMENT.get(ep, (np.array(B.CUP["xy"], float), np.array(B.BLOCK["xy"], float)))
    hz = B.BLOCK["half"][2]
    _set_mocap(m, d, "cup", np.array([cup_xy[0], cup_xy[1], 0.0]))
    mujoco.mj_forward(m, d)
    if phase in ("grasp", "transfer_open"):
        # brick centred between the two jaw tips (fixed-jaw tip on the gripper body, moving-jaw tip on its body; both
        # measured from the official meshes), 1 cm back from the tips towards the palm, oriented with the gripper
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        mj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        Rg = d.xmat[bid].reshape(3, 3)
        fixed_tip = d.xpos[bid] + Rg @ FIXED_TIP
        moving_tip = d.xpos[mj] + d.xmat[mj].reshape(3, 3) @ MOVING_TIP
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.xmat[bid])
        _set_mocap(m, d, "block", 0.5 * (fixed_tip + moving_tip) + Rg @ np.array([0.0, 0.0, 0.010]), q)
    elif phase == "release":
        _set_mocap(m, d, "block", np.array([cup_xy[0], cup_xy[1], 0.006 + hz]), np.array([1.0, 0, 0, 0]))
    else:
        _set_mocap(m, d, "block", np.array([blk_xy[0], blk_xy[1], hz]), np.array([1.0, 0, 0, 0]))
    mujoco.mj_forward(m, d)


Z_BLOCK = B.BLOCK["half"][2]          # the builder's ray-cast planes (place_objects.py)
Z_CUP = 0.5 * B.CUP["height"]


def ray_cast_objects(rig, cam: str = "top") -> dict:
    """Re-place every episode's cup / brick by casting the builder's detected pixel centroids (objects.json cup_px /
    block_px, real top frame) through the REFITTED top camera (pose, fovy, lens) onto the builder's planes. Updates
    PLACEMENT in place and returns {episode: dict(cup=.., block=.., moved_cm=..)}."""
    a = rig.arm
    rig.pose(cam, "all", np.zeros(6))
    cid = a.cam_id[cam]
    pos = a.d.cam_xpos[cid].copy()
    R = a.d.cam_xmat[cid].reshape(3, 3)
    L = rig.lens[cam]
    w, h = 640, 480
    f = 0.5 * h / math.tan(math.radians(float(a.m.cam_fovy[cid])) / 2)

    def hit(u, v, z):
        xd = (u + 0.5 - w / 2 - L.cx * w) / f
        yd = (v + 0.5 - h / 2 - L.cy * h) / f
        xu, yu = xd, yd
        for _ in range(20):
            r2 = xu * xu + yu * yu
            s = 1 + L.k1 * r2 + L.k2 * r2 * r2
            xu, yu = xd / s, yd / s
        d = R @ np.array([xu, -yu, -1.0])
        t = (z - pos[2]) / d[2]
        return (pos + t * d)[:2]

    out = {}
    for k, v in OBJECTS.items():
        ep = int(k)
        old_cup, old_blk = PLACEMENT[ep]
        cup = hit(*v["cup_px"], Z_CUP) if "cup_px" in v and not v.get("cup_missing") else old_cup
        blk = hit(*v["block_px"], Z_BLOCK) if "block_px" in v and not v.get("block_missing") else old_blk
        PLACEMENT[ep] = (np.asarray(cup, float), np.asarray(blk, float))
        out[ep] = dict(cup=[round(float(x), 4) for x in cup], block=[round(float(x), 4) for x in blk],
                       moved_cm=[round(100 * float(np.linalg.norm(cup - old_cup)), 2),
                                 round(100 * float(np.linalg.norm(blk - old_blk)), 2)])
    return out
