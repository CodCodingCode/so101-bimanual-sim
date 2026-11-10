"""Single SO-101 arm "drawer" scene: the digital twin of data_so101/0mrrabbit0__so101_place_block_in_drawer.

    .venv/bin/python so101_twin/drawer/build_scene.py        # writes so101_drawer.xml next to this file

World frame: table (optical breadboard) top surface at z = 0, the arm base origin at (0, 0, 0), the arm
reaches along +y when the pan joint is at zero. The real "side" camera (dataset observation.images.head)
is an overhead camera looking straight down: image up = +y, image right = +x. The real "top" camera
(observation.images.front) is NOT static: it is the SO-101 wrist camera mounted on the gripper (the white
jaws fill the bottom of every frame), so in the twin it is a camera inside the Fixed_Jaw body and its
parameters are expressed in that body's frame. Camera parameters (pos, yaw/pitch/roll in degrees, fovy) are
at the top; fit_cameras.py writes the fitted values into CAMS here (see README.md).

Arm model: mujoco_menagerie trs_so_arm100 meshes from ../assets/so_arm100 (Apache-2.0); joint names
Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw. The real arm is printed green with an orange section
at the wrist end of the forearm, black motors, and a white gripper / wrist-camera mount.
"""
from __future__ import annotations

import copy
import math
import pathlib
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
ARM_XML = HERE.parent / "assets" / "so_arm100" / "so_arm100.xml"
MESHDIR = (HERE.parent / "assets" / "so_arm100").resolve()
OUT_XML = HERE / "so101_drawer.xml"
TEX_PNG = HERE / "breadboard.png"

RES_W, RES_H = 640, 480

# ---- cameras (initial guesses here; fit_cameras.py overwrites them; yaw/pitch/roll deg, pitch < 0 looks down) --
# "side": world frame. "top": the wrist camera, pos / angles in the Fixed_Jaw body frame (fingers along -y of
# that frame, jaw opening along x, jaw axis z).
CAMS = {
    "top": dict(pos=(0.0421, 0.0182, 0.0624), yaw=-111.57, pitch=-16.52, roll=23.15, fovy=41.72),
    "side": dict(pos=(0.0898, 0.0291, 0.3874), yaw=92.35, pitch=-78.47, roll=3.23, fovy=70.08),
}
CAM_BODY = {"top": "Fixed_Jaw", "side": None}

# ---- scene layout (metres, table surface z = 0; object (x, y) from place_objects.py with the fitted "side" camera) --
BASE_POS = (0.0, 0.0, 0.0)
BASE_QUAT = "0 0 0 1"          # Menagerie arm reaches along -y at identity; 180 deg about z -> reaches +y
TABLE_HALF = (0.60, 0.50, 0.02)
TABLE_CENTER = (0.10, 0.20)
HOLE_PITCH = 0.025             # optical breadboard grid
# Drawer unit: grey box with a light blue-grey top, in front of the base. The drawer (a white tray) slides out
# toward the robot (-y) by DRAWER_TRAVEL when open (replay.py moves it from the real frames' drawer indicator).
DRAWER_UNIT = dict(center=(-0.010, 0.335), size=(0.113, 0.130, 0.055))     # full outer size x, y, z (place_objects.py; height from the front-face pixel height)
DRAWER_TRAY = dict(size=(0.096, 0.110, 0.036), wall=0.003)                 # inner tray (full size), wall thickness
DRAWER_TRAVEL = 0.070
# Blocks (half sizes). The purple one is the "left block" that gets moved (mocap body, see replay.py).
OBJECTS = {
    "purple": dict(xy=(0.032, 0.184), size=(0.011, 0.015, 0.011), rgba="0.52 0.33 0.85 1"),
    "cyan": dict(xy=(0.108, 0.191), size=(0.010, 0.014, 0.013), rgba="0.10 0.70 0.85 1"),
}
# Background clutter: a laptop to the robot's left (dark base, light screen), a keyboard on the right.
LAPTOP = dict(base_center=(-0.29, 0.15, 0.012), base_half=(0.11, 0.15, 0.012),
              screen_center=(-0.33, 0.15, 0.11), screen_half=(0.005, 0.14, 0.10), tilt_deg=-15)
KEYBOARD = dict(center=(0.42, 0.15, 0.008), half=(0.05, 0.07, 0.008))

MATERIALS = {
    "arm_green": dict(rgba="0.47 0.62 0.40 1", specular="0.15", shininess="0.25"),
    "arm_orange": dict(rgba="0.95 0.45 0.10 1", specular="0.15", shininess="0.25"),
    "arm_black": dict(rgba="0.10 0.10 0.10 1"),
    "jaw_white": dict(rgba="0.92 0.92 0.90 1", specular="0.2", shininess="0.3"),
    "drawer_grey": dict(rgba="0.80 0.80 0.80 1", specular="0.1", shininess="0.2"),
    "drawer_top": dict(rgba="0.62 0.72 0.80 1", specular="0.2", shininess="0.3"),
    "tray_white": dict(rgba="0.93 0.93 0.92 1", specular="0.1", shininess="0.2"),
    "tray_floor": dict(rgba="0.70 0.70 0.70 1", specular="0.1", shininess="0.2"),
    "laptop_dark": dict(rgba="0.12 0.12 0.13 1"),
    "screen": dict(rgba="0.85 0.86 0.88 1", specular="0.3", shininess="0.5"),
    "keyboard": dict(rgba="0.18 0.18 0.19 1"),
    "wall": dict(rgba="0.80 0.80 0.78 1", specular="0.0"),
    "floor": dict(rgba="0.25 0.24 0.22 1"),
}
JAW_MESHES = {"Fixed_Jaw", "Moving_Jaw"}
# orange section at the wrist end of the forearm (visual box in the Lower_Arm frame, which extends along +z)
FOREARM_ORANGE = dict(pos="0 0.004 0.112", size="0.017 0.021 0.023")


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


def write_breadboard_texture(path: pathlib.Path = TEX_PNG, px: int = 64) -> pathlib.Path:
    """One tile = one hole pitch: dark board with a black hole in the middle."""
    import numpy as np
    from PIL import Image

    yy, xx = np.mgrid[0:px, 0:px]
    r = np.hypot(xx - px / 2 + 0.5, yy - px / 2 + 0.5) / px * HOLE_PITCH
    img = np.full((px, px, 3), (34, 34, 36), np.uint8)
    img[r < 0.0032] = (6, 6, 7)
    Image.fromarray(img).save(path)
    return path


def drawer_geoms(parent: ET.Element, wb: ET.Element) -> None:
    """Grey drawer unit (hollow at the front) with a blue-grey top; the tray itself is a separate mocap body."""
    cx, cy = DRAWER_UNIT["center"]
    sx, sy, sz = DRAWER_UNIT["size"]
    tx, ty, tz = DRAWER_TRAY["size"]
    gap = 0.004
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    kw = dict(material="drawer_grey", contype="0", conaffinity="0")
    top_t = 0.006
    # top plate (blue-grey) and a thin raised rim around it
    ET.SubElement(parent, "geom", name="drawer_top", type="box", size=f"{hx} {hy} {top_t / 2}",
                  pos=f"{cx} {cy} {sz - top_t / 2}", material="drawer_top", contype="0", conaffinity="0")
    for i, (ox, oy, ex, ey) in enumerate([(0, hy - 0.003, hx, 0.003), (0, -hy + 0.003, hx, 0.003),
                                          (hx - 0.003, 0, 0.003, hy), (-hx + 0.003, 0, 0.003, hy)]):
        ET.SubElement(parent, "geom", name=f"drawer_rim_{i}", type="box", size=f"{ex} {ey} 0.002",
                      pos=f"{cx + ox} {cy + oy} {sz + 0.002}", material="tray_white", contype="0", conaffinity="0")
    # side walls, back wall, bottom plate
    wall = (sx - tx - 2 * gap) / 2
    ET.SubElement(parent, "geom", name="drawer_side_l", type="box", size=f"{wall / 2} {hy} {hz}",
                  pos=f"{cx - hx + wall / 2} {cy} {hz}", **kw)
    ET.SubElement(parent, "geom", name="drawer_side_r", type="box", size=f"{wall / 2} {hy} {hz}",
                  pos=f"{cx + hx - wall / 2} {cy} {hz}", **kw)
    ET.SubElement(parent, "geom", name="drawer_back", type="box", size=f"{hx} 0.004 {hz}",
                  pos=f"{cx} {cy + hy - 0.004} {hz}", **kw)
    ET.SubElement(parent, "geom", name="drawer_bottom", type="box", size=f"{hx} {hy} 0.004",
                  pos=f"{cx} {cy} 0.004", **kw)
    # the sliding tray: mocap body at its closed position (front flush with the unit's front face)
    tray = ET.SubElement(wb, "body", name="tray", pos=f"{cx} {cy} 0", mocap="true")   # mocap: must be a child of world
    w = DRAWER_TRAY["wall"]
    z0 = 0.010
    ET.SubElement(tray, "geom", name="tray_floor", type="box", size=f"{tx / 2} {ty / 2} 0.002",
                  pos=f"0 {-hy + ty / 2} {z0}", material="tray_floor", contype="0", conaffinity="0")
    ET.SubElement(tray, "geom", name="tray_front", type="box", size=f"{tx / 2 + gap} {w} {tz / 2 + 0.004}",
                  pos=f"0 {-hy + w} {z0 + tz / 2}", material="tray_white", contype="0", conaffinity="0")
    ET.SubElement(tray, "geom", name="tray_back", type="box", size=f"{tx / 2} {w} {tz / 2}",
                  pos=f"0 {-hy + ty - w} {z0 + tz / 2}", material="tray_white", contype="0", conaffinity="0")
    for s, nm in ((-1, "l"), (1, "r")):
        ET.SubElement(tray, "geom", name=f"tray_side_{nm}", type="box", size=f"{w} {ty / 2} {tz / 2}",
                      pos=f"{s * (tx / 2 - w)} {-hy + ty / 2} {z0 + tz / 2}", material="tray_white",
                      contype="0", conaffinity="0")


def build() -> str:
    arm = ET.parse(ARM_XML).getroot()
    write_breadboard_texture()
    root = ET.Element("mujoco", model="so101_drawer")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(MESHDIR), texturedir=str(HERE), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(RES_W), offheight=str(RES_H))
    ET.SubElement(vis, "headlight", ambient="0.35 0.35 0.35", diffuse="0.35 0.35 0.35", specular="0.05 0.05 0.05")
    ET.SubElement(vis, "quality", shadowsize="4096")

    root.append(copy.deepcopy(arm.find("default")))
    asset = ET.SubElement(root, "asset")
    for a in arm.find("asset"):
        asset.append(copy.deepcopy(a))
    for name, kw in MATERIALS.items():
        ET.SubElement(asset, "material", name=name, **kw)
    ET.SubElement(asset, "texture", name="breadboard", type="2d", file=TEX_PNG.name)
    ET.SubElement(asset, "material", name="breadboard", texture="breadboard", texuniform="true",
                  texrepeat=f"{1 / HOLE_PITCH:.4f} {1 / HOLE_PITCH:.4f}", specular="0.35", shininess="0.4",
                  reflectance="0.0")
    for name, o in OBJECTS.items():
        ET.SubElement(asset, "material", name=f"mat_{name}", rgba=o["rgba"], specular="0.2", shininess="0.3")

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", name="sun", pos="0.2 0.3 1.8", dir="-0.05 -0.05 -1", directional="false",
                  diffuse="0.45 0.45 0.45", specular="0.15 0.15 0.15", castshadow="true", cutoff="90")
    ET.SubElement(wb, "light", name="fill", pos="0.9 -0.5 1.0", dir="-0.6 0.5 -0.7", diffuse="0.15 0.15 0.15",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.74", material="floor")
    ET.SubElement(wb, "geom", name="table_top", type="box", size=" ".join(map(str, TABLE_HALF)),
                  pos=f"{TABLE_CENTER[0]} {TABLE_CENTER[1]} {-TABLE_HALF[2]}", material="breadboard")
    ET.SubElement(wb, "geom", name="wall", type="box", size="2.0 0.01 1.2", pos="0 1.2 0.46", material="wall")
    # laptop (left) and keyboard (right)
    bc, bh = LAPTOP["base_center"], LAPTOP["base_half"]
    ET.SubElement(wb, "geom", name="laptop_base", type="box", size=" ".join(map(str, bh)),
                  pos=" ".join(map(str, bc)), material="laptop_dark", contype="0", conaffinity="0")
    sc, sh = LAPTOP["screen_center"], LAPTOP["screen_half"]
    ET.SubElement(wb, "geom", name="laptop_screen", type="box", size=" ".join(map(str, sh)),
                  pos=" ".join(map(str, sc)), euler=f"0 {math.radians(LAPTOP['tilt_deg']):.4f} 0",
                  material="screen", contype="0", conaffinity="0")
    ET.SubElement(wb, "geom", name="keyboard", type="box", size=" ".join(map(str, KEYBOARD["half"])),
                  pos=" ".join(map(str, KEYBOARD["center"])), material="keyboard", contype="0", conaffinity="0")
    for name, c in CAMS.items():
        if CAM_BODY[name] is None:
            ET.SubElement(wb, "camera", name=name, **cam_attrs(c))

    # the arm, recoloured
    base = copy.deepcopy(arm.find("worldbody").find("body"))
    base.set("pos", " ".join(map(str, BASE_POS)))
    base.set("quat", BASE_QUAT)
    for g in base.iter("geom"):
        cls = g.get("class", "")
        if cls == "visual":
            g.set("material", "jaw_white" if g.get("mesh") in JAW_MESHES else "arm_green")
        elif cls == "motor_visual":
            g.set("material", "arm_black")
    lower = next(b for b in base.iter("body") if b.get("name") == "Lower_Arm")
    ET.SubElement(lower, "geom", name="forearm_orange", type="box", pos=FOREARM_ORANGE["pos"],
                  size=FOREARM_ORANGE["size"], material="arm_orange", contype="0", conaffinity="0", group="2")
    jaw = next(b for b in base.iter("body") if b.get("name") == "Fixed_Jaw")
    ET.SubElement(jaw, "site", name="tcp", pos="-0.012 -0.080 0", size="0.004", group="3", rgba="1 0 0 0")
    for name, c in CAMS.items():
        if CAM_BODY[name] == "Fixed_Jaw":
            ET.SubElement(jaw, "camera", name=name, **cam_attrs(c))
    wb.append(base)

    # objects (mocap bodies so replay.py can move the purple block without physics)
    for name, o in OBJECTS.items():
        x, y = o["xy"]
        b = ET.SubElement(wb, "body", name=name, pos=f"{x} {y} {o['size'][2]}", mocap="true")
        ET.SubElement(b, "geom", name=name, type="box", size=" ".join(map(str, o["size"])),
                      material=f"mat_{name}", contype="0", conaffinity="0")
    drawer = ET.SubElement(wb, "body", name="drawer", pos="0 0 0")
    drawer_geoms(drawer, wb)

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
