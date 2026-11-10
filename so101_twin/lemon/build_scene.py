"""Single SO-101 arm "lemon" scene: the digital twin of data_so101/Beegbrain__pick_lemon_and_drop_in_bowl.

    .venv/bin/python so101_twin/lemon/build_scene.py        # writes so101_lemon.xml next to this file

World frame: table top surface at z = 0, the arm base origin at (0, 0, 0), the arm reaches along +y when the
pan joint is at zero. The real "top" camera is a low view from the robot's right (+x) looking across the table
toward -x; the real "side" camera is an overhead view from in front of the robot (+y) looking back toward -y
and down. Camera parameters (pos, yaw/pitch/roll in degrees, fovy) are at the top; fit_cameras.py writes the
fitted values into CAMS here (see README.md for the fitted numbers and IoU).

Arm model: mujoco_menagerie trs_so_arm100 meshes from ../assets/so_arm100 (Apache-2.0); joint names
Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw. Arm body is orange, motors black, jaws blue like the real arm.
"""
from __future__ import annotations

import copy
import math
import pathlib
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
ARM_XML = HERE.parent / "assets" / "so_arm100" / "so_arm100.xml"
MESHDIR = (HERE.parent / "assets" / "so_arm100").resolve()
OUT_XML = HERE / "so101_lemon.xml"

RES_W, RES_H = 640, 480

# ---- cameras (initial guesses here; fit_cameras.py overwrites them; yaw/pitch/roll deg, pitch < 0 looks down) --
CAMS = {
    "top": dict(pos=(0.4494, 0.2623, 0.1447), yaw=184.28, pitch=-3.68, roll=-1.24, fovy=42.83),
    "side": dict(pos=(-0.0314, 0.4912, 0.3675), yaw=-88.68, pitch=-57.97, roll=3.22, fovy=46.55),
}

# ---- scene layout (metres, table surface z = 0) ---------------------------------------------------------------
BASE_POS = (0.0, 0.0, 0.0)
BASE_QUAT = "0 0 0 1"          # Menagerie arm reaches along -y at identity; 180 deg about z -> reaches +y
TABLE_HALF = (0.55, 0.45, 0.02)
TABLE_CENTER = (0.0, 0.25)
WALL_X = -0.62                 # plain wall behind the scene as seen from the top camera (looking -x)
# Objects, (x, y) on the table surface; sizes are radii. Positions estimated from the first real frames by casting
# the fitted cameras' rays onto the table plane (place_objects.py). The lemon is moved during replay (see replay.py).
OBJECTS = {
    "lemon": dict(xy=(-0.006, 0.280), size=(0.028, 0.023, 0.023), rgba="0.98 0.85 0.12 1", type="ellipsoid"),
    "apple": dict(xy=(0.120, 0.214), size=(0.035, 0.035, 0.035), rgba="0.78 0.10 0.10 1", type="sphere"),
    "strawberry": dict(xy=(0.101, 0.345), size=(0.024, 0.024, 0.028), rgba="0.80 0.08 0.10 1", type="ellipsoid"),
    "peach": dict(xy=(-0.042, 0.411), size=(0.032, 0.032, 0.030), rgba="0.95 0.58 0.12 1", type="ellipsoid"),
}
BOWL = dict(xy=(-0.203, 0.270), r_top=0.088, r_bot=0.050, height=0.055, rgba="0.80 0.10 0.08 1", n=40)

MATERIALS = {
    "arm_orange": dict(rgba="0.93 0.42 0.10 1", specular="0.2", shininess="0.3"),
    "arm_black": dict(rgba="0.10 0.10 0.10 1"),
    "jaw_blue": dict(rgba="0.12 0.42 0.75 1", specular="0.2", shininess="0.3"),
    "wood": dict(rgba="0.88 0.79 0.64 1", specular="0.05", shininess="0.1"),
    "wall": dict(rgba="0.92 0.91 0.88 1", specular="0.0"),
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


def bowl_geoms(parent: ET.Element) -> None:
    """Red bowl = a flared ring of thin boxes + a floor disc."""
    x0, y0 = BOWL["xy"]
    n, h = BOWL["n"], BOWL["height"]
    r_t, r_b = BOWL["r_top"], BOWL["r_bot"]
    r_mid = 0.5 * (r_t + r_b)
    tilt = math.atan2(r_t - r_b, h)                     # outward flare
    seg_len = math.sqrt(h ** 2 + (r_t - r_b) ** 2)
    half_w = math.pi * r_t / n * 1.08
    for i in range(n):
        a = 2 * math.pi * i / n
        cx, cy, cz = x0 + r_mid * math.cos(a), y0 + r_mid * math.sin(a), h / 2
        # box z-axis along the wall: first tilt about y by -tilt (lean outward), then yaw about z by a
        ET.SubElement(parent, "geom", name=f"bowl_wall_{i}", type="box",
                      size=f"{0.004} {half_w} {seg_len / 2}", pos=f"{cx:.5f} {cy:.5f} {cz:.5f}",
                      euler=f"0 {-tilt:.5f} {a:.5f}", material="bowl", contype="0", conaffinity="0")
    ET.SubElement(parent, "geom", name="bowl_floor", type="cylinder", size=f"{r_b} 0.002",
                  pos=f"{x0} {y0} 0.002", material="bowl", contype="0", conaffinity="0")


def build() -> str:
    arm = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", model="so101_lemon")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(MESHDIR), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(RES_W), offheight=str(RES_H))
    ET.SubElement(vis, "headlight", ambient="0.32 0.32 0.32", diffuse="0.30 0.30 0.30", specular="0.05 0.05 0.05")
    ET.SubElement(vis, "quality", shadowsize="4096")

    root.append(copy.deepcopy(arm.find("default")))
    asset = ET.SubElement(root, "asset")
    for a in arm.find("asset"):
        asset.append(copy.deepcopy(a))
    for name, kw in MATERIALS.items():
        ET.SubElement(asset, "material", name=name, **kw)
    ET.SubElement(asset, "material", name="bowl", rgba=BOWL["rgba"], specular="0.4", shininess="0.5")
    for name, o in OBJECTS.items():
        ET.SubElement(asset, "material", name=f"mat_{name}", rgba=o["rgba"], specular="0.3", shininess="0.4")

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", name="sun", pos="0.3 0.6 1.6", dir="-0.2 -0.3 -1", directional="false",
                  diffuse="0.42 0.42 0.42", specular="0.1 0.1 0.1", castshadow="true", cutoff="90")
    ET.SubElement(wb, "light", name="fill", pos="0.8 -0.4 1.0", dir="-0.6 0.4 -0.8", diffuse="0.15 0.15 0.15",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.74", material="floor")
    ET.SubElement(wb, "geom", name="table_top", type="box", size=" ".join(map(str, TABLE_HALF)),
                  pos=f"{TABLE_CENTER[0]} {TABLE_CENTER[1]} {-TABLE_HALF[2]}", material="wood")
    ET.SubElement(wb, "geom", name="wall", type="box", size="0.01 2.0 1.2", pos=f"{WALL_X} 0.2 0.46", material="wall")
    for name, c in CAMS.items():
        ET.SubElement(wb, "camera", name=name, **cam_attrs(c))

    # the arm, recoloured
    base = copy.deepcopy(arm.find("worldbody").find("body"))
    base.set("pos", " ".join(map(str, BASE_POS)))
    base.set("quat", BASE_QUAT)
    for g in base.iter("geom"):
        cls = g.get("class", "")
        if cls == "visual":
            g.set("material", "jaw_blue" if g.get("mesh") in JAW_MESHES else "arm_orange")
        elif cls == "motor_visual":
            g.set("material", "arm_black")
    jaw = next(b for b in base.iter("body") if b.get("name") == "Fixed_Jaw")
    ET.SubElement(jaw, "site", name="tcp", pos="-0.012 -0.080 0", size="0.004", group="3", rgba="1 0 0 0")
    wb.append(base)

    # objects (mocap bodies so replay.py can move the lemon without physics)
    for name, o in OBJECTS.items():
        x, y = o["xy"]
        b = ET.SubElement(wb, "body", name=name, pos=f"{x} {y} {o['size'][2]}", mocap="true")
        ET.SubElement(b, "geom", name=name, type=o["type"], size=" ".join(map(str, o["size"])),
                      material=f"mat_{name}", contype="0", conaffinity="0")
    bowl = ET.SubElement(wb, "body", name="bowl", pos="0 0 0")
    bowl_geoms(bowl)

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
