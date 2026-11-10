"""Scene description: stack (data_so101/ichbinblau__so101_stack2cubes_dataset), official SO-101, NEW calibration.

Written by the stack2 session (so101_twin/stack2/README.md). Same contract as scenes/lemon.py; used by common/fit.py
(and the stack2 copies of the toolkit, which import this file).

World frame = the builder's (so101_twin/stack/build_scene.py, read-only): table top z = 0, the arm's pan axis at
(0, 0.0452), reaching +y at pan = 0. Objects (paper disc, wooden cube, green block), table, cloth / panel backdrops and
the cube timeline (on the disc / in the jaws / on the block) are the builder's, unchanged; arm model, joint map and
cameras are new.

Arm: TheRobotStudio SO-101 so101_new_calib.xml (LeRobot so101_follower, range-middle calibration) with the official
wrist-camera mount (the real arm carries a wrist camera on the gripper), white body / black servos.
Joint map: q = deg2rad(scale * state + offset); the initial offsets / scales are NOT zero / one: the nominal identity
map puts the held cube 3-7 cm too low (below the table at the grasp), so they are seeded by bundle adjustment on the
held wooden cube at full rate (so101_twin/stack2/bundle2.py, even episodes only) and refined by the silhouette fit.

Cameras: "top" moves between episodes (phase correlation of the static table / disc / block background between
episode medians, never the arm: episodes 0-6 at (0, 0) px, 39-99 at (0, -3), 105-126 at (+2, -1), 131-198 at
(+45, +1)), so one pose per session, shared fovy / distortion; every session contains even (fit) episodes.
"side" is static (<= 1 px drift): one pose. Initial poses from out/seed_cams.json of the stack2 session (PnP / bundle
on the held cube), falling back to the builder's cameras.
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

TWIN = pathlib.Path(__file__).resolve().parents[2]
_SRC = TWIN / "stack" / "build_scene.py"
_spec = importlib.util.spec_from_file_location("stack_build_scene_ro", _SRC)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

REAL_SET = "so101_stack_real"
WHITE = (0.93, 0.93, 0.92, 1)
ARM = A.ArmSpec(calib="new", model="so101", wrist_mount=True, base_pos=(0.0, 0.0452, 0.0), base_yaw_deg=0.0,
                colors={"body": WHITE, "jaw": WHITE, "motor": (0.09, 0.09, 0.09, 1), "cam_mount": WHITE,
                        "camera": (0.08, 0.08, 0.08, 1)})


def top_group(ep: int) -> str:
    if ep <= 6:
        return "g0_e000-006"
    if ep <= 99:
        return "g1_e039-099"
    if ep <= 126:
        return "g2_e105-126"
    return "g3_e131-198"


GROUP_RULE = {"top": "episodes 0-6 -> g0, 39-99 -> g1, 105-126 -> g2, 131-198 -> g3 (sessions from phase correlation "
                     "of the static background between episode medians, not from the arm; each has even episodes)",
              "side": "single pose"}

# builder's cameras (weak), ported to the SO-101 frame by port_scene.py: fallback init only
CAMERAS = {
    "top": dict(mount="world", pos=(-0.3511, 0.2303, 0.522), quat=(0.851453, 0.088988, -0.174574, -0.486448),
                fovy=49.56, group=top_group),
    "side": dict(mount="world", pos=(0.1347, 0.5335, 0.1614), quat=(0.267161, 0.18721, 0.548046, 0.77021), fovy=33.39),
}
OFFSETS_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]
SCALES_INIT = [1.0, 1.0, 1.0, 1.0, 1.0]
SEED_JSON = TWIN / "stack2" / "out" / "seed_cams.json"
if SEED_JSON.exists():
    _s = json.loads(SEED_JSON.read_text())
    for _c, _v in ((k, v) for k, v in _s.get("cams", {}).items() if k in CAMERAS):
        CAMERAS[_c].update(pos=tuple(_v["pos"]), quat=tuple(_v["quat"]), fovy=float(_v["fovy"]))
        if _v.get("groups"):
            CAMERAS[_c]["init_groups"] = {g: dict(pos=tuple(p["pos"]), quat=tuple(p["quat"])) for g, p in _v["groups"].items()}
    _j = _s.get("joint", {})
    OFFSETS_INIT = [float(x) for x in _j.get("offsets", OFFSETS_INIT)]
    SCALES_INIT = [float(x) for x in _j.get("scales", SCALES_INIT)]

OFFSET_SIGMA = [5.0, 6.0, 6.0, 6.0, 10.0]
FIT_SCALES = True
FIT_LAG = True
ROLL_CANDIDATES = [0.0, 90.0, 180.0, -90.0]

# ------------------------------------------------------------------------------------------------ scene for RGB
HOLD = (5.0, 18.0)          # builder: gripper % range = jaws closed on the cube


def extras(root: ET.Element, asset: ET.Element, wb: ET.Element) -> None:
    for name, kw in B.MATERIALS.items():
        if not name.startswith(("arm_", "jaw_")):
            ET.SubElement(asset, "material", name=name, **kw)
    for name, o in B.OBJECTS.items():
        ET.SubElement(asset, "material", name=f"mat_{name}", rgba=o["rgba"], specular="0.2", shininess="0.3")
    ET.SubElement(wb, "light", name="sun", pos="-0.2 0.1 1.5", dir="0.25 0.15 -1", directional="false",
                  diffuse="0.45 0.45 0.45", specular="0.1 0.1 0.1", castshadow="true", cutoff="90")
    ET.SubElement(wb, "light", name="fill", pos="0.8 -0.4 1.0", dir="-0.6 0.4 -0.8", diffuse="0.15 0.15 0.15",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.74", material="floor")
    ET.SubElement(wb, "geom", name="table_top", type="box", size=" ".join(map(str, B.TABLE_HALF)),
                  pos=f"{B.TABLE_CENTER[0]} {B.TABLE_CENTER[1]} {-B.TABLE_HALF[2]}", material="table")
    for name, sheet in (("cloth", B.CLOTH), ("panel", B.PANEL)):
        ET.SubElement(wb, "geom", name=name, type="box", size=f"0.01 {sheet['half_w']} {sheet['height'] / 2}",
                      pos=f"{sheet['center'][0]} {sheet['center'][1]} {sheet['height'] / 2}",
                      euler=f"0 0 {math.radians(sheet['yaw_deg']):.4f}", material=name, group="4")   # side view only
    D = B.DISC
    ET.SubElement(wb, "geom", name="disc", type="cylinder", size=f"{D['r']} {D['h'] / 2}",
                  pos=f"{D['xy'][0]} {D['xy'][1]} {D['h'] / 2}", material="disc", contype="0", conaffinity="0")
    for name, o in B.OBJECTS.items():
        x, y = o["xy"]
        b = ET.SubElement(wb, "body", name=name, pos=f"{x} {y} {o['size'][2]}", mocap="true")
        ET.SubElement(b, "geom", name=name, type=o["type"], size=" ".join(map(str, o["size"])),
                      material=f"mat_{name}", contype="0", conaffinity="0")


GEOMGROUPS = {"top": {4: False}, "side": {4: True}}
TABLE_Z = 0.0               # table plane height in the arm frame (the stack2 render may set it from the fitted arm)   # builder: backdrop sheets only in the side view (stack2 render)

_WHERE = {}


def cube_where(fs) -> np.ndarray:
    """builder's rule (so101_twin/stack/replay.py): 'disc' | 'jaws' | 'block' from each episode's gripper timeline."""
    key = id(fs)
    if key in _WHERE:
        return _WHERE[key]
    g, ep, t = fs.state[:, 5], np.asarray(fs.labels["episode"]), np.asarray(fs.labels["t"])
    where = np.full(fs.n, "disc", dtype="<U5")
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        idx = idx[np.argsort(t[idx])]
        hold = (g[idx] >= HOLD[0]) & (g[idx] <= HOLD[1])
        if hold.any():
            first, last = np.argmax(hold), len(idx) - 1 - np.argmax(hold[::-1])
        else:
            op = g[idx] > HOLD[1]
            first, last = np.argmax(op) + 1, len(idx) - 1 - np.argmax(op[::-1]) - 1
        for k, i in enumerate(idx):
            where[i] = "disc" if k < first else ("jaws" if k <= last else "block")
    _WHERE[key] = where
    return where


def frame_setup(rig, fs, i: int) -> None:
    m, d = rig.arm.m, rig.arm.d
    where = str(cube_where(fs)[i])
    h = B.OBJECTS["cube"]["size"][2]
    quat = np.array([1.0, 0, 0, 0])
    if where == "jaws":
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        pos = d.site_xpos[sid].copy()
        mujoco.mju_mat2Quat(quat, d.site_xmat[sid].copy())
    elif where == "block":
        gx, gy = B.OBJECTS["green"]["xy"]
        pos = np.array([gx, gy, TABLE_Z + 2 * B.OBJECTS["green"]["size"][2] + h])
    else:
        x, y = B.OBJECTS["cube"]["xy"]
        pos = np.array([x, y, TABLE_Z + h])
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    d.mocap_pos[m.body_mocapid[bid]] = pos
    d.mocap_quat[m.body_mocapid[bid]] = quat
    mujoco.mj_forward(m, d)
