"""Single SO-101 arm "ringpole" scene: the digital twin of data_so101/nbirukov__so101_pick_stack_ring_pole.

    .venv/bin/python so101_twin/ringpole/build_scene.py        # writes so101_ringpole.xml next to this file

World frame: table top surface at z = 0, the arm base origin at (0, 0, 0), the arm reaches along +x when the
pan joint is at zero (Menagerie base rotated +90 deg about z). The real "top" camera (dataset `observation.images.up`)
is a static overhead view; the real "side" camera (`observation.images.sholder`) is mounted on the arm's shoulder
and rotates with the pan joint, so in the twin it is a child of a `pan_frame` body under `Rotation_Pitch`
(x = arm forward, z = up at the pan axis). CAMS["top"] is in the world frame, CAMS["side"] in the pan frame;
fit_cameras.py writes the fitted values here (see README.md).

Objects: a white vertical pole (thin cylinder) on a purple disc base and a coloured ring (torus approximated by a
flat cylinder with a hole = a ring of thin boxes) - both mocap bodies moved per frame by replay.py (per-episode
layout from place_objects.py). Arm body white, motors black like the real white SO-101.
"""
from __future__ import annotations

import copy
import math
import pathlib
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
ARM_XML = HERE.parent / "assets" / "so_arm100" / "so_arm100.xml"
MESHDIR = (HERE.parent / "assets" / "so_arm100").resolve()
OUT_XML = HERE / "so101_ringpole.xml"

RES_W, RES_H = 640, 480

# ---- cameras (initial guesses here; fit_cameras.py overwrites them; yaw/pitch/roll deg, pitch < 0 looks down) --
# "top": world frame. "side": pan frame (origin = pan axis at the Rotation_Pitch body, x = arm forward, z = up).
CAMS = {
    "top": dict(pos=(0.1787, -0.1573, 0.7271), yaw=69.06, pitch=-79.70, roll=22.84, fovy=34.07),
    "side": dict(pos=(0.0507, -0.0966, 0.0579), yaw=21.54, pitch=-5.66, roll=-3.67, fovy=21.22),
}

# ---- scene layout (metres, table surface z = 0) ---------------------------------------------------------------
BASE_POS = (0.0, 0.0, 0.0)
BASE_QUAT = "0.7071068 0 0 0.7071068"   # Menagerie arm reaches along -y at identity; +90 deg about z -> reaches +x
TABLE_HALF = (0.60, 0.50, 0.02)
TABLE_CENTER = (0.25, 0.0)
WALL_X = 0.75                           # bright orange backdrop (curtain) behind the pole as seen from the shoulder cam
# Default object layout (episode 0 / frame 0); replay.py overrides per episode from place_objects.py output.
POLE = dict(xy=(0.335, 0.03), r=0.011, h=0.19, base_r=0.040, base_h=0.012, rgba="0.95 0.93 0.90 1",
            base_rgba="0.62 0.52 0.82 1")
RING = dict(xy=(0.32, 0.14), r_out=0.032, r_in=0.012, h=0.022, n=24)
RING_COLORS = {"orange": "0.98 0.45 0.08 1", "pink": "0.95 0.35 0.55 1", "yellow": "0.98 0.85 0.15 1"}

MATERIALS = {
    "arm_white": dict(rgba="0.92 0.92 0.90 1", specular="0.2", shininess="0.3"),
    "arm_black": dict(rgba="0.10 0.10 0.10 1"),
    "wood": dict(rgba="0.50 0.36 0.27 1", specular="0.15", shininess="0.3"),
    "wood_edge": dict(rgba="0.55 0.38 0.22 1", specular="0.05", shininess="0.1"),
    "wall": dict(rgba="1.0 0.55 0.12 1", specular="0.0", emission="0.6"),
    "wall_far": dict(rgba="0.85 0.75 0.60 1", specular="0.0"),
    "floor": dict(rgba="0.20 0.16 0.14 1"),
    "pole": dict(rgba=POLE["rgba"], specular="0.3", shininess="0.4"),
    "pole_base": dict(rgba=POLE["base_rgba"], specular="0.3", shininess="0.4"),
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


def ring_geoms(parent: ET.Element, name: str = "ring") -> None:
    """Flat torus approximation: n thin boxes around a circle (leaves the hole open)."""
    n, h = RING["n"], RING["h"]
    r_o, r_i = RING["r_out"], RING["r_in"]
    r_mid = 0.5 * (r_o + r_i)
    radial = 0.5 * (r_o - r_i)
    half_w = math.pi * r_o / n * 1.1
    for i in range(n):
        a = 2 * math.pi * i / n
        ET.SubElement(parent, "geom", name=f"{name}_{i}", type="box", size=f"{radial:.5f} {half_w:.5f} {h / 2:.5f}",
                      pos=f"{r_mid * math.cos(a):.5f} {r_mid * math.sin(a):.5f} 0",
                      euler=f"0 0 {a:.5f}", material="ring", contype="0", conaffinity="0")


def build(ring_color: str = "orange") -> str:
    arm = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", model="so101_ringpole")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(MESHDIR), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(RES_W), offheight=str(RES_H))
    ET.SubElement(vis, "headlight", ambient="0.36 0.33 0.30", diffuse="0.32 0.30 0.28", specular="0.05 0.05 0.05")
    ET.SubElement(vis, "quality", shadowsize="4096")

    root.append(copy.deepcopy(arm.find("default")))
    asset = ET.SubElement(root, "asset")
    for a in arm.find("asset"):
        asset.append(copy.deepcopy(a))
    for name, kw in MATERIALS.items():
        ET.SubElement(asset, "material", name=name, **kw)
    ET.SubElement(asset, "material", name="ring", rgba=RING_COLORS[ring_color], specular="0.3", shininess="0.5")

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", name="sun", pos="0.2 0.3 1.5", dir="0.05 -0.2 -1", directional="false",
                  diffuse="0.45 0.42 0.38", specular="0.1 0.1 0.1", castshadow="true", cutoff="90")
    ET.SubElement(wb, "light", name="warm", pos="0.9 0.0 0.5", dir="-1 0 -0.3", diffuse="0.35 0.22 0.10",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.74", material="floor")
    ET.SubElement(wb, "geom", name="table_top", type="box", size=" ".join(map(str, TABLE_HALF)),
                  pos=f"{TABLE_CENTER[0]} {TABLE_CENTER[1]} {-TABLE_HALF[2]}", material="wood")
    # lighter wood strip at the near (-y) table edge and a beige wall strip beyond the far (+y) edge (top view)
    ET.SubElement(wb, "geom", name="edge_near", type="box", size=f"{TABLE_HALF[0]} 0.03 0.015",
                  pos=f"{TABLE_CENTER[0]} {-TABLE_HALF[1] - 0.03} -0.015", material="wood_edge")
    ET.SubElement(wb, "geom", name="wall_far", type="box", size=f"{TABLE_HALF[0] + 0.5} 0.01 0.6",
                  pos=f"{TABLE_CENTER[0]} {TABLE_HALF[1] + 0.02} 0.0", material="wall_far")
    ET.SubElement(wb, "geom", name="wall", type="box", size="0.01 2.0 1.2", pos=f"{WALL_X} 0.0 0.46", material="wall")
    ET.SubElement(wb, "camera", name="top", **cam_attrs(CAMS["top"]))

    # the arm, recoloured
    base = copy.deepcopy(arm.find("worldbody").find("body"))
    base.set("pos", " ".join(map(str, BASE_POS)))
    base.set("quat", BASE_QUAT)
    for g in base.iter("geom"):
        cls = g.get("class", "")
        if cls == "visual":
            g.set("material", "arm_white")
        elif cls == "motor_visual":
            g.set("material", "arm_black")
    jaw = next(b for b in base.iter("body") if b.get("name") == "Fixed_Jaw")
    ET.SubElement(jaw, "site", name="tcp", pos="-0.012 -0.080 0", size="0.004", group="3", rgba="1 0 0 0")
    # pan frame under Rotation_Pitch: x = arm forward, y = left, z = up (Rotation_Pitch body: z fwd, x left, y up)
    rp = next(b for b in base.iter("body") if b.get("name") == "Rotation_Pitch")
    pf = ET.SubElement(rp, "body", name="pan_frame", pos="0 0 0", quat="0.5 -0.5 -0.5 -0.5")
    ET.SubElement(pf, "camera", name="side", **cam_attrs(CAMS["side"]))
    wb.append(base)

    # objects (mocap bodies so replay.py can move them per frame without physics)
    px, py = POLE["xy"]
    pole = ET.SubElement(wb, "body", name="pole", pos=f"{px} {py} 0", mocap="true")
    ET.SubElement(pole, "geom", name="pole", type="cylinder", size=f"{POLE['r']} {POLE['h'] / 2}",
                  pos=f"0 0 {POLE['h'] / 2}", material="pole", contype="0", conaffinity="0")
    ET.SubElement(pole, "geom", name="pole_base", type="cylinder", size=f"{POLE['base_r']} {POLE['base_h'] / 2}",
                  pos=f"0 0 {POLE['base_h'] / 2}", material="pole_base", contype="0", conaffinity="0")
    rx, ry = RING["xy"]
    ring = ET.SubElement(wb, "body", name="ring", pos=f"{rx} {ry} {RING['h'] / 2}", mocap="true")
    ring_geoms(ring)

    act = ET.SubElement(root, "actuator")
    for a in arm.find("actuator"):
        act.append(copy.deepcopy(a))
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="rest", qpos="0 -3.3 3.1 1.1 0 0", ctrl="0 -3.3 3.1 1.1 0 0")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def write(ring_color: str = "orange") -> pathlib.Path:
    OUT_XML.write_text(build(ring_color) + "\n")
    return OUT_XML


if __name__ == "__main__":
    p = write()
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(p))
    print(f"wrote {p}: nq={m.nq} nu={m.nu} ncam={m.ncam} nbody={m.nbody} nmocap={m.nmocap}")
    print("cameras:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)])
