"""Single SO-101 arm "bincube" scene: the digital twin of data_so101/MCeut__so101_lab_openBin_putCube.

    .venv/bin/python so101_twin/bincube/build_scene.py        # writes so101_bincube.xml next to this file

World frame: table top surface at z = 0, the arm base origin at (0, 0, 0), the arm reaches along +y when the
pan joint is at zero. Despite the real set's camera names, the real "top" camera (observation.images.front) is a
WRIST camera mounted on the gripper (both jaws are visible at the bottom of every frame); the real "side" camera
(observation.images.left) is a static view from in front / left of the robot looking back at the arm.
In the sim, "top" is a camera attached to the Fixed_Jaw body (parameters in that body's frame) and "side" is a
world camera. fit_cameras.py writes the fitted values into CAMS here (see README.md).

Arm model: mujoco_menagerie trs_so_arm100 meshes from ../assets/so_arm100 (Apache-2.0); joint names
Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw. The real arm is all black (printed parts + motors).
"""
from __future__ import annotations

import copy
import math
import pathlib
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
ARM_XML = HERE.parent / "assets" / "so_arm100" / "so_arm100.xml"
MESHDIR = (HERE.parent / "assets" / "so_arm100").resolve()
OUT_XML = HERE / "so101_bincube.xml"

RES_W, RES_H = 640, 480

# ---- cameras (fit_cameras.py overwrites these lines; yaw/pitch/roll deg, pitch < 0 looks down) -------------------
# "top": wrist camera, pos / angles in the Fixed_Jaw body frame (jaw tip along -y of that frame).
# "side": static camera in the world frame.
CAMS = {
    "top": dict(pos=(0.0073, 0.0098, -0.0607), yaw=-106.60, pitch=16.21, roll=173.40, fovy=45.01, body="Fixed_Jaw"),
    "side": dict(pos=(-0.1763, 0.5181, 0.3834), yaw=-66.26, pitch=-42.59, roll=-8.17, fovy=36.28),
}

# ---- scene layout (metres, table surface z = 0) ---------------------------------------------------------------
BASE_POS = (0.0, 0.0, 0.0)
BASE_QUAT = "0 0 0 1"          # Menagerie arm reaches along -y at identity; 180 deg about z -> reaches +y
BOARD_HALF = (0.30, 0.45, 0.009)   # light wood board the robot and the bin sit on
BOARD_CENTER = (0.05, 0.30)
BENCH_HALF = (1.2, 1.2, 0.01)      # black optical breadboard under the board (seen beyond the board's edges)
WALL_Y = -0.30                     # brown wall / box behind the robot as seen from the side camera
# Objects on the table surface. Bin = open grey box (4 walls + floor); lid = flat dark box hinged on the bin's
# +y (camera-side) top edge: closed lies on top of the bin, open is flipped over onto the table in front of it.
# The cube is moved during replay (table -> jaws -> bin), per-episode start positions in replay.CUBE_XY.
BIN = dict(xy=(-0.013, 0.299), yaw=0.0, half=(0.047, 0.037, 0.060), wall=0.004, rgba="0.45 0.47 0.48 1")
LID = dict(half=(0.052, 0.042, 0.007), rgba="0.12 0.12 0.12 1")
CUBE = dict(xy=(0.0, 0.163), half=0.020, rgba="0.95 0.82 0.10 1")

MATERIALS = {
    "arm_black": dict(rgba="0.07 0.07 0.07 1", specular="0.2", shininess="0.3"),
    "wood": dict(rgba="0.86 0.80 0.68 1", specular="0.05", shininess="0.1"),
    "bench": dict(rgba="0.08 0.08 0.09 1", specular="0.1", shininess="0.2"),
    "wall": dict(rgba="0.55 0.42 0.30 1", specular="0.0"),
    "floor": dict(rgba="0.25 0.25 0.25 1"),
    "bin": dict(rgba=BIN["rgba"], specular="0.2", shininess="0.3"),
    "bin_in": dict(rgba="0.22 0.23 0.24 1", specular="0.1", shininess="0.1"),
    "lid": dict(rgba=LID["rgba"], specular="0.3", shininess="0.4"),
    "cube": dict(rgba=CUBE["rgba"], specular="0.2", shininess="0.3"),
}


def cam_xyaxes(yaw: float, pitch: float, roll: float) -> tuple[list[float], list[float]]:
    """Camera right/up axes for a view direction given by yaw (about z), pitch (elevation) and roll, in degrees."""
    import numpy as np

    y, p, r = map(math.radians, (yaw, pitch, roll))
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    # roll about the forward axis
    right_r = math.cos(r) * right + math.sin(r) * up
    up_r = -math.sin(r) * right + math.cos(r) * up
    return right_r.tolist(), up_r.tolist()


def cam_attrs(c: dict) -> dict:
    right, up = cam_xyaxes(c["yaw"], c["pitch"], c["roll"])
    return dict(pos=" ".join(f"{v:.5f}" for v in c["pos"]),
                xyaxes=" ".join(f"{v:.6f}" for v in right + up),
                fovy=f"{c['fovy']:.3f}", resolution=f"{RES_W} {RES_H}")


def bin_geoms(parent: ET.Element) -> None:
    """Grey open bin: four thin walls + floor, centred on BIN['xy'], rotated BIN['yaw'] about z."""
    hx, hy, hz = BIN["half"]
    w = BIN["wall"]
    for i, (px, py, sx, sy) in enumerate([(hx - w, 0, w, hy), (-(hx - w), 0, w, hy), (0, hy - w, hx, w), (0, -(hy - w), hx, w)]):
        ET.SubElement(parent, "geom", name=f"bin_wall_{i}", type="box", size=f"{sx} {sy} {hz}", pos=f"{px} {py} {hz}",
                      material="bin", contype="0", conaffinity="0")
    ET.SubElement(parent, "geom", name="bin_floor", type="box", size=f"{hx} {hy} 0.003", pos=f"0 0 0.003",
                  material="bin_in", contype="0", conaffinity="0")


def build() -> str:
    arm = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", model="so101_bincube")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(MESHDIR), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(RES_W), offheight=str(RES_H))
    ET.SubElement(vis, "headlight", ambient="0.40 0.40 0.40", diffuse="0.35 0.35 0.35", specular="0.05 0.05 0.05")
    ET.SubElement(vis, "quality", shadowsize="4096")

    root.append(copy.deepcopy(arm.find("default")))
    asset = ET.SubElement(root, "asset")
    for a in arm.find("asset"):
        asset.append(copy.deepcopy(a))
    for name, kw in MATERIALS.items():
        ET.SubElement(asset, "material", name=name, **kw)

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", name="sun", pos="0.2 0.5 1.8", dir="-0.1 -0.2 -1", directional="false",
                  diffuse="0.45 0.45 0.45", specular="0.1 0.1 0.1", castshadow="true", cutoff="90")
    ET.SubElement(wb, "light", name="fill", pos="0.9 0.9 1.0", dir="-0.6 -0.5 -0.7", diffuse="0.18 0.18 0.18",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.80", material="floor")
    ET.SubElement(wb, "geom", name="bench", type="box", size=" ".join(map(str, BENCH_HALF)),
                  pos=f"0 0.3 {-2 * BOARD_HALF[2] - BENCH_HALF[2]}", material="bench")
    ET.SubElement(wb, "geom", name="board", type="box", size=" ".join(map(str, BOARD_HALF)),
                  pos=f"{BOARD_CENTER[0]} {BOARD_CENTER[1]} {-BOARD_HALF[2]}", material="wood")
    ET.SubElement(wb, "geom", name="wall", type="box", size="2.0 0.01 1.0", pos=f"0 {WALL_Y} 0.6", material="wall")
    for name, c in CAMS.items():
        if "body" not in c:
            ET.SubElement(wb, "camera", name=name, **cam_attrs(c))

    # the arm, all black; wrist camera on the Fixed_Jaw body
    base = copy.deepcopy(arm.find("worldbody").find("body"))
    base.set("pos", " ".join(map(str, BASE_POS)))
    base.set("quat", BASE_QUAT)
    for g in base.iter("geom"):
        if g.get("class", "") in ("visual", "motor_visual"):
            g.set("material", "arm_black")
    jaw = next(b for b in base.iter("body") if b.get("name") == "Fixed_Jaw")
    ET.SubElement(jaw, "site", name="tcp", pos="-0.012 -0.080 0", size="0.004", group="3", rgba="1 0 0 0")
    for name, c in CAMS.items():
        if "body" in c:
            host = next(b for b in base.iter("body") if b.get("name") == c["body"])
            ET.SubElement(host, "camera", name=name, **cam_attrs(c))
    wb.append(base)

    # bin (static), lid and cube (mocap bodies so replay.py can move them without physics)
    bx, by = BIN["xy"]
    b = ET.SubElement(wb, "body", name="bin", pos=f"{bx} {by} 0", euler=f"0 0 {BIN['yaw']:.4f}")
    bin_geoms(b)
    lh = LID["half"]
    b = ET.SubElement(wb, "body", name="lid", pos=f"{bx} {by} {2 * BIN['half'][2] + lh[2]}", mocap="true")
    ET.SubElement(b, "geom", name="lid", type="box", size=" ".join(map(str, lh)), material="lid",
                  contype="0", conaffinity="0")
    ET.SubElement(b, "geom", name="lid_handle", type="box", size=f"{lh[0] * 0.5} 0.006 0.004",
                  pos=f"0 {lh[1] * 0.7} {lh[2] + 0.004}", material="lid", contype="0", conaffinity="0")
    cx, cy = CUBE["xy"]
    b = ET.SubElement(wb, "body", name="cube", pos=f"{cx} {cy} {CUBE['half']}", mocap="true")
    ET.SubElement(b, "geom", name="cube", type="box", size=" ".join([str(CUBE["half"])] * 3), material="cube",
                  contype="0", conaffinity="0")

    act = ET.SubElement(root, "actuator")
    for a in arm.find("actuator"):
        act.append(copy.deepcopy(a))
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="rest", qpos="0 -3.3 3.1 1.1 0 0", ctrl="0 -3.3 3.1 1.1 0 0")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def write() -> pathlib.Path:
    OUT_XML.write_text(build() + "\n")
    return OUT_XML


if __name__ == "__main__":
    p = write()
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(p))
    print(f"wrote {p}: nq={m.nq} nu={m.nu} ncam={m.ncam} nbody={m.nbody} nmocap={m.nmocap}")
    print("cameras:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)])
