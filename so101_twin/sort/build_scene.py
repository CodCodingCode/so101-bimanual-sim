"""Single SO-101 arm "sort" scene: the digital twin of data_so101/hubnemo__so101_sort_cubes.

    .venv/bin/python so101_twin/sort/build_scene.py        # writes so101_sort.xml next to this file

World frame: table top surface at z = 0, the arm base origin at (0, 0, 0), the arm reaches along +y when the
pan joint is at zero (same convention as ../lemon). The real "top" camera sits behind and above the arm base and
looks along +y and down; it was moved between two recording sessions, so there are two static cameras
"top" (session A, episodes 4..55) and "top_b" (session B, episodes 66..101). The real "side" camera is a
wrist camera on the fixed jaw: modelled as camera "side" attached to the Fixed_Jaw body (pos / yaw / pitch / roll
in the jaw frame, where the finger tips point along -y). fit_cameras.py writes the fitted values into CAMS.

Arm model: mujoco_menagerie trs_so_arm100 meshes from ../assets/so_arm100 (Apache-2.0); joint names
Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw. Arm body white (reads lilac under the real room light),
motors black, wrist / jaws light blue like the real arm.
"""
from __future__ import annotations

import copy
import math
import pathlib
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
ARM_XML = HERE.parent / "assets" / "so_arm100" / "so_arm100.xml"
MESHDIR = (HERE.parent / "assets" / "so_arm100").resolve()
OUT_XML = HERE / "so101_sort.xml"

RES_W, RES_H = 640, 480

# ---- cameras (initial guesses here; fit_cameras.py overwrites them; yaw/pitch/roll deg, pitch < 0 looks down) --
# "top"/"top_b": world frame.  "side": Fixed_Jaw body frame (fingers along -y, jaw opens along x).
CAMS = {
    "top": dict(pos=(-0.0297, -0.0256, 0.3029), yaw=83.84, pitch=-27.29, roll=-3.10, fovy=57.02),
    "top_b": dict(pos=(-0.0164, -0.0249, 0.3297), yaw=85.51, pitch=-36.72, roll=-0.28, fovy=50.47),
    "side": dict(pos=(-0.0100, 0.0000, 0.0450), yaw=-90.00, pitch=-10.00, roll=180.00, fovy=60.00),
}
# session -> which static camera renders the real "top" image; episodes by LeRobot episode index
SESSIONS = {"top": range(0, 60), "top_b": range(60, 200)}


def top_cam_for_episode(ep: int) -> str:
    for cam, eps in SESSIONS.items():
        if int(ep) in eps:
            return cam
    raise KeyError(ep)


# ---- scene layout (metres, table surface z = 0) ---------------------------------------------------------------
BASE_POS = (0.0, 0.0, 0.0)
BASE_QUAT = "0 0 0 1"          # Menagerie arm reaches along -y at identity; 180 deg about z -> reaches +y
TABLE_HALF = (0.80, 0.55, 0.02)
TABLE_CENTER = (0.0, 0.30)
# Cubes (purple foam, ~4 cm) and the cardboard box. Cube layout varies per episode: replay.py reads per-episode
# positions from out/cube_layouts.json (place_objects.py); these are the defaults (episode 4, frame 0).
CUBE_HALF = 0.021
N_CUBES = 4
CUBES_DEFAULT = [(-0.03, 0.30), (0.05, 0.33), (0.02, 0.38), (0.09, 0.27)]
CUBE_RGBA = "0.33 0.25 0.70 1"
BOX = dict(xy=(0.40, 0.50), half=(0.085, 0.065, 0.075), rgba="0.72 0.50 0.28 1", wall=0.006)
# static background props (approximate): PC tower, laptop, dark room behind the table
PROPS = {
    "pc_tower": dict(pos=(-0.50, 0.95, 0.20), half=(0.09, 0.22, 0.20), rgba="0.08 0.08 0.09 1"),
    "laptop_base": dict(pos=(0.22, 0.90, 0.008), half=(0.16, 0.11, 0.008), rgba="0.70 0.70 0.72 1"),
    "laptop_lid": dict(pos=(0.22, 1.00, 0.11), half=(0.16, 0.004, 0.11), rgba="0.72 0.72 0.74 1"),
}
WALL_Y = 1.40                  # dark back wall

MATERIALS = {
    "arm_white": dict(rgba="0.80 0.80 0.90 1", specular="0.2", shininess="0.3"),
    "arm_black": dict(rgba="0.10 0.10 0.10 1"),
    "jaw_blue": dict(rgba="0.20 0.60 0.85 1", specular="0.2", shininess="0.3"),
    "table": dict(rgba="0.86 0.85 0.90 1", specular="0.1", shininess="0.2"),
    "wall": dict(rgba="0.12 0.08 0.08 1", specular="0.0"),
    "floor": dict(rgba="0.25 0.20 0.20 1"),
    "cube": dict(rgba=CUBE_RGBA, specular="0.05", shininess="0.1"),
    "cardboard": dict(rgba=BOX["rgba"], specular="0.05", shininess="0.1"),
}
BLUE_MESHES = {"Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw"}


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


def box_geoms(parent: ET.Element) -> None:
    """Open-top cardboard box: four walls + floor."""
    x0, y0 = BOX["xy"]
    hx, hy, hz = BOX["half"]
    w = BOX["wall"]
    ET.SubElement(parent, "geom", name="box_floor", type="box", size=f"{hx} {hy} {w / 2}", pos=f"{x0} {y0} {w / 2}",
                  material="cardboard", contype="0", conaffinity="0")
    for i, (dx, dy, sx, sy) in enumerate([(hx, 0, w / 2, hy), (-hx, 0, w / 2, hy), (0, hy, hx, w / 2), (0, -hy, hx, w / 2)]):
        ET.SubElement(parent, "geom", name=f"box_wall_{i}", type="box", size=f"{sx} {sy} {hz}",
                      pos=f"{x0 + dx:.4f} {y0 + dy:.4f} {hz}", material="cardboard", contype="0", conaffinity="0")


def build() -> str:
    arm = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", model="so101_sort")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(MESHDIR), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(RES_W), offheight=str(RES_H))
    ET.SubElement(vis, "headlight", ambient="0.38 0.38 0.40", diffuse="0.30 0.30 0.30", specular="0.05 0.05 0.05")
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "map", znear="0.0005")   # fraction of the model extent; the wrist camera is ~3 cm from the fingers

    root.append(copy.deepcopy(arm.find("default")))
    asset = ET.SubElement(root, "asset")
    for a in arm.find("asset"):
        asset.append(copy.deepcopy(a))
    for name, kw in MATERIALS.items():
        ET.SubElement(asset, "material", name=name, **kw)
    for name, p in PROPS.items():
        ET.SubElement(asset, "material", name=f"mat_{name}", rgba=p["rgba"], specular="0.1", shininess="0.2")

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", name="sun", pos="-0.2 0.3 1.8", dir="0.1 0.1 -1", directional="false",
                  diffuse="0.45 0.43 0.47", specular="0.1 0.1 0.1", castshadow="true", cutoff="100")
    ET.SubElement(wb, "light", name="fill", pos="0.8 -0.6 1.0", dir="-0.6 0.5 -0.7", diffuse="0.15 0.14 0.16",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.74", material="floor")
    ET.SubElement(wb, "geom", name="table_top", type="box", size=" ".join(map(str, TABLE_HALF)),
                  pos=f"{TABLE_CENTER[0]} {TABLE_CENTER[1]} {-TABLE_HALF[2]}", material="table")
    ET.SubElement(wb, "geom", name="wall", type="box", size="3.0 0.01 1.5", pos=f"0 {WALL_Y} 0.76", material="wall")
    for name, p in PROPS.items():
        ET.SubElement(wb, "geom", name=name, type="box", size=" ".join(map(str, p["half"])),
                      pos=" ".join(map(str, p["pos"])), material=f"mat_{name}", contype="0", conaffinity="0")
    for name in ("top", "top_b"):
        ET.SubElement(wb, "camera", name=name, **cam_attrs(CAMS[name]))

    # the arm, recoloured
    base = copy.deepcopy(arm.find("worldbody").find("body"))
    base.set("pos", " ".join(map(str, BASE_POS)))
    base.set("quat", BASE_QUAT)
    for g in base.iter("geom"):
        cls = g.get("class", "")
        if cls == "visual":
            g.set("material", "jaw_blue" if g.get("mesh") in BLUE_MESHES else "arm_white")
        elif cls == "motor_visual":
            g.set("material", "arm_black")
    jaw = next(b for b in base.iter("body") if b.get("name") == "Fixed_Jaw")
    ET.SubElement(jaw, "site", name="tcp", pos="-0.012 -0.080 0", size="0.004", group="3", rgba="1 0 0 0")
    ET.SubElement(jaw, "camera", name="side", **cam_attrs(CAMS["side"]))
    wb.append(base)

    # cubes (mocap bodies so replay.py can place them per episode / carry the target without physics)
    for i in range(N_CUBES):
        x, y = CUBES_DEFAULT[i]
        b = ET.SubElement(wb, "body", name=f"cube{i}", pos=f"{x} {y} {CUBE_HALF}", mocap="true")
        ET.SubElement(b, "geom", name=f"cube{i}", type="box", size=f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}",
                      material="cube", contype="0", conaffinity="0")
    box = ET.SubElement(wb, "body", name="box", pos="0 0 0")
    box_geoms(box)

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
