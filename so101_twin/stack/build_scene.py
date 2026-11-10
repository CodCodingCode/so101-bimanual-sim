"""Single SO-101 arm "stack" scene: the digital twin of data_so101/ichbinblau__so101_stack2cubes_dataset.

    .venv/bin/python so101_twin/stack/build_scene.py        # writes so101_stack.xml next to this file

World frame: table top surface at z = 0, the arm base origin at (0, 0, 0), the arm reaches along +y when the
pan joint is at zero. In the real "top" view (a low overhead camera, ~0.45 m up, tilted: the base is in view
right of centre and the folded arm projects toward the top edge) image-left is +y and image-up is +x. The real
"side" camera is a static table-level camera at (+x, +y) beyond the green block looking back diagonally toward
the base / the blue cloth backdrop, which hangs behind the work area on the (-x, -y) side.
Camera parameters (pos, yaw/pitch/roll in degrees, fovy) are at the top; fit_cameras.py writes the fitted
values into CAMS here (see README.md for the fitted numbers and IoU).

Arm model: mujoco_menagerie trs_so_arm100 meshes from ../assets/so_arm100 (Apache-2.0); joint names
Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw. Arm body white, motors black, jaws white like the real arm.
"""
from __future__ import annotations

import copy
import math
import pathlib
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
ARM_XML = HERE.parent / "assets" / "so_arm100" / "so_arm100.xml"
MESHDIR = (HERE.parent / "assets" / "so_arm100").resolve()
OUT_XML = HERE / "so101_stack.xml"

RES_W, RES_H = 640, 480

# ---- cameras (initial guesses here; fit_cameras.py overwrites them; yaw/pitch/roll deg, pitch < 0 looks down) --
CAMS = {
    "top": dict(pos=(-0.3511, 0.2303, 0.5220), yaw=-2.73, pitch=-67.40, roll=33.25, fovy=49.56),
    "side": dict(pos=(0.1347, 0.5335, 0.1614), yaw=-127.99, pitch=-19.22, roll=-0.27, fovy=33.39),
}

# ---- scene layout (metres, table surface z = 0) ---------------------------------------------------------------
BASE_POS = (0.0, 0.0, 0.0)
BASE_QUAT = "0 0 0 1"          # Menagerie arm reaches along -y at identity; 180 deg about z -> reaches +y
TABLE_HALF = (0.60, 0.60, 0.02)
TABLE_CENTER = (-0.05, 0.10)
# blue cloth backdrop: a vertical sheet facing the side camera, 0.6 m in front of it; white panel to its right
CLOTH = dict(center=(-0.31, -0.04), yaw_deg=31.5, half_w=0.8, height=1.2)
PANEL = dict(center=(-0.41, 0.31), yaw_deg=121.5, half_w=0.35, height=1.2)
# Objects, (x, y) on the table surface; full sizes (box edge lengths). Positions estimated from real frame 0 by
# casting the fitted cameras' rays onto the table plane (place_objects.py). The cube is moved during replay.
DISC = dict(xy=(-0.106, 0.212), r=0.042, h=0.003, rgba="0.97 0.97 0.97 1")      # white paper disc = cube start spot
OBJECTS = {
    "cube": dict(xy=(-0.151, 0.196), size=(0.017, 0.017, 0.017), rgba="0.85 0.62 0.36 1", type="box"),
    "green": dict(xy=(-0.082, 0.318), size=(0.026, 0.026, 0.026), rgba="0.12 0.42 0.22 1", type="box"),
}

MATERIALS = {
    "arm_white": dict(rgba="0.93 0.93 0.92 1", specular="0.15", shininess="0.2"),
    "arm_black": dict(rgba="0.10 0.10 0.10 1"),
    "jaw_white": dict(rgba="0.93 0.93 0.92 1", specular="0.15", shininess="0.2"),
    "table": dict(rgba="0.84 0.82 0.80 1", specular="0.05", shininess="0.1"),
    "cloth": dict(rgba="0.42 0.60 0.78 1", specular="0.0"),
    "panel": dict(rgba="0.95 0.95 0.94 1", specular="0.0"),
    "disc": dict(rgba=DISC["rgba"], specular="0.0"),
    "floor": dict(rgba="0.35 0.33 0.30 1"),
}
JAW_MESHES = {"Fixed_Jaw", "Moving_Jaw"}


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


def build() -> str:
    arm = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", model="so101_stack")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(MESHDIR), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(RES_W), offheight=str(RES_H))
    ET.SubElement(vis, "headlight", ambient="0.36 0.36 0.36", diffuse="0.30 0.30 0.30", specular="0.05 0.05 0.05")
    ET.SubElement(vis, "quality", shadowsize="4096")

    root.append(copy.deepcopy(arm.find("default")))
    asset = ET.SubElement(root, "asset")
    for a in arm.find("asset"):
        asset.append(copy.deepcopy(a))
    for name, kw in MATERIALS.items():
        ET.SubElement(asset, "material", name=name, **kw)
    for name, o in OBJECTS.items():
        ET.SubElement(asset, "material", name=f"mat_{name}", rgba=o["rgba"], specular="0.2", shininess="0.3")

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", name="sun", pos="-0.2 0.1 1.5", dir="0.25 0.15 -1", directional="false",
                  diffuse="0.45 0.45 0.45", specular="0.1 0.1 0.1", castshadow="true", cutoff="90")
    ET.SubElement(wb, "light", name="fill", pos="0.8 -0.4 1.0", dir="-0.6 0.4 -0.8", diffuse="0.15 0.15 0.15",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.74", material="floor")
    ET.SubElement(wb, "geom", name="table_top", type="box", size=" ".join(map(str, TABLE_HALF)),
                  pos=f"{TABLE_CENTER[0]} {TABLE_CENTER[1]} {-TABLE_HALF[2]}", material="table")
    for name, sheet in (("cloth", CLOTH), ("panel", PANEL)):
        ET.SubElement(wb, "geom", name=name, type="box", size=f"0.01 {sheet['half_w']} {sheet['height'] / 2}",
                      pos=f"{sheet['center'][0]} {sheet['center'][1]} {sheet['height'] / 2}",
                      euler=f"0 0 {math.radians(sheet['yaw_deg']):.4f}", material=name, group="4")   # group 4: hidden in "top"
    ET.SubElement(wb, "geom", name="disc", type="cylinder", size=f"{DISC['r']} {DISC['h'] / 2}",
                  pos=f"{DISC['xy'][0]} {DISC['xy'][1]} {DISC['h'] / 2}", material="disc", contype="0", conaffinity="0")
    for name, c in CAMS.items():
        ET.SubElement(wb, "camera", name=name, **cam_attrs(c))

    # the arm, recoloured
    base = copy.deepcopy(arm.find("worldbody").find("body"))
    base.set("pos", " ".join(map(str, BASE_POS)))
    base.set("quat", BASE_QUAT)
    for g in base.iter("geom"):
        cls = g.get("class", "")
        if cls == "visual":
            g.set("material", "jaw_white" if g.get("mesh") in JAW_MESHES else "arm_white")
        elif cls == "motor_visual":
            g.set("material", "arm_black")
    jaw = next(b for b in base.iter("body") if b.get("name") == "Fixed_Jaw")
    ET.SubElement(jaw, "site", name="tcp", pos="-0.012 -0.080 0", size="0.004", group="3", rgba="1 0 0 0")
    wb.append(base)

    # objects (mocap bodies so replay.py can move the cube without physics)
    for name, o in OBJECTS.items():
        x, y = o["xy"]
        b = ET.SubElement(wb, "body", name=name, pos=f"{x} {y} {o['size'][2]}", mocap="true")
        ET.SubElement(b, "geom", name=name, type=o["type"], size=" ".join(map(str, o["size"])),
                      material=f"mat_{name}", contype="0", conaffinity="0")

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
