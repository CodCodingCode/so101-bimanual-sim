"""Single SO-101 arm "legocup" scene: the digital twin of
data_so101/Rayenghali__so101_follower_main_put_the_red_lego_block_in_the_black_cup_eps100_fps30_480x640.

    .venv/bin/python so101_twin/legocup/build_scene.py        # writes so101_legocup.xml next to this file

World frame: table top surface at z = 0, the arm base origin at (0, 0, 0), the arm reaches along +y when the pan
joint is at zero. The real "top" camera (dataset `observation.images.main`) is an overhead view from in front of
the robot (+y) looking back toward -y and steeply down; the base is just above the top edge of the image. The real
"side" camera (`observation.images.secondary_0`) is NOT static: it is a wrist camera mounted on the gripper (the
two jaw prongs hang from the top edge of every frame), so here it is a camera child of the `Fixed_Jaw` body and its
pose is given in that body's frame. Camera parameters (pos, yaw/pitch/roll deg, fovy) are at the top;
fit_cameras.py --apply writes the fitted values into CAMS (see README.md for the numbers and IoU).

Arm model: mujoco_menagerie trs_so_arm100 meshes from ../assets/so_arm100 (Apache-2.0); joint names
Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw. The real arm is printed black: every link, motor and jaw is
black here. A dark square plate on the fixed jaw approximates the wrist-camera mount seen from the top camera.
"""
from __future__ import annotations

import copy
import json
import math
import pathlib
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
ARM_XML = HERE.parent / "assets" / "so_arm100" / "so_arm100.xml"
MESHDIR = (HERE.parent / "assets" / "so_arm100").resolve()
OUT_XML = HERE / "so101_legocup.xml"
OBJECTS_JSON = HERE / "out" / "objects.json"     # per-episode cup / block positions (place_objects.py)

RES_W, RES_H = 640, 480

# ---- cameras (initial guesses here; fit_cameras.py --apply overwrites them; yaw/pitch/roll deg, pitch < 0 looks
# down). "top" is in the world frame; "side" is in the Fixed_Jaw body frame (jaw tips are along -y of that frame).
CAMS = {
    "top": dict(pos=(-0.0421, 0.4660, 0.4687), yaw=-85.35, pitch=-68.51, roll=5.98, fovy=35.04),
    "side": dict(pos=(0.0058, -0.0125, 0.0776), yaw=-92.08, pitch=-25.45, roll=183.88, fovy=59.86),
}
CAM_PARENT = {"top": "world", "side": "Fixed_Jaw"}

# ---- scene layout (metres, table surface z = 0) ---------------------------------------------------------------
BASE_POS = (0.0, 0.0, 0.0)
BASE_QUAT = "0 0 0 1"          # Menagerie arm reaches along -y at identity; 180 deg about z -> reaches +y
TABLE_HALF = (0.70, 0.60, 0.02)
TABLE_CENTER = (0.0, 0.30)
# grey rubber mat (place_objects.py --mat prints its corners from the fitted top camera)
MAT_CENTER = (-0.035, 0.251)      # from place_objects.py --mat (ray-cast mat corners of real frame 0)
MAT_HALF = (0.164, 0.157, 0.001)
MAT_YAW_DEG = 10.3
# Objects: default positions (episode 0); replay.py moves them per episode from out/objects.json.
CUP = dict(xy=(-0.08, 0.17), r=0.040, height=0.080, n=32, rgba="0.03 0.03 0.03 1", handle=True, handle_dir=-1)  # handle on -x (image right)
BLOCK = dict(xy=(0.02, 0.33), half=(0.016, 0.016, 0.0096), yaw_deg=0.0, rgba="0.90 0.10 0.08 1", studs=True)
CAM_MOUNT = None   # (x, y, z) centre + half sizes in the Fixed_Jaw frame, e.g. dict(pos=(0, 0.03, -0.05), half=(0.022, 0.002, 0.022))

MATERIALS = {
    "arm_black": dict(rgba="0.07 0.07 0.075 1", specular="0.25", shininess="0.35"),
    "mat_grey": dict(rgba="0.56 0.58 0.57 1", specular="0.08", shininess="0.2"),
    "table_dark": dict(rgba="0.19 0.19 0.18 1", specular="0.05", shininess="0.1"),
    "wall": dict(rgba="0.25 0.25 0.24 1", specular="0.0"),
    "floor": dict(rgba="0.15 0.15 0.14 1"),
    "cup": dict(rgba=CUP["rgba"], specular="0.5", shininess="0.6"),
    "block": dict(rgba=BLOCK["rgba"], specular="0.3", shininess="0.4"),
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


def cup_geoms(parent: ET.Element) -> None:
    """Black mug = a straight ring of thin boxes + a floor disc + a handle (half ring of boxes) on +x."""
    n, h, r = CUP["n"], CUP["height"], CUP["r"]
    half_w = math.pi * r / n * 1.08
    for i in range(n):
        a = 2 * math.pi * i / n
        ET.SubElement(parent, "geom", name=f"cup_wall_{i}", type="box",
                      size=f"0.003 {half_w:.5f} {h / 2:.5f}", pos=f"{r * math.cos(a):.5f} {r * math.sin(a):.5f} {h / 2:.5f}",
                      euler=f"0 0 {a:.5f}", material="cup", contype="0", conaffinity="0")
    ET.SubElement(parent, "geom", name="cup_floor", type="cylinder", size=f"{r - 0.002} 0.003", pos="0 0 0.003",
                  material="cup", contype="0", conaffinity="0")
    if CUP.get("handle"):
        rh, nh = 0.022, 7     # handle: half ring in the x-z plane, attached at x = r
        for i in range(nh):
            a = -math.pi / 2 + math.pi * i / (nh - 1)
            cx, cz = CUP.get("handle_dir", 1) * (r + 0.004 + rh * math.cos(a)), h * 0.5 + rh * math.sin(a)
            ET.SubElement(parent, "geom", name=f"cup_handle_{i}", type="capsule", size="0.0045 0.009",
                          pos=f"{cx:.5f} 0 {cz:.5f}", euler=f"0 {a + math.pi / 2:.5f} 0", material="cup",
                          contype="0", conaffinity="0")


def block_geoms(parent: ET.Element) -> None:
    hx, hy, hz = BLOCK["half"]
    ET.SubElement(parent, "geom", name="block", type="box", size=f"{hx} {hy} {hz}", pos="0 0 0",
                  material="block", contype="0", conaffinity="0")
    if BLOCK.get("studs"):
        k = 0
        for i in range(4):
            for j in range(4):
                x, y = (i - 1.5) * 0.008, (j - 1.5) * 0.008
                ET.SubElement(parent, "geom", name=f"stud_{k}", type="cylinder", size="0.0024 0.0009",
                              pos=f"{x:.4f} {y:.4f} {hz + 0.0009:.4f}", material="block", contype="0", conaffinity="0")
                k += 1


def build() -> str:
    arm = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", model="so101_legocup")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(MESHDIR), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(RES_W), offheight=str(RES_H))
    ET.SubElement(vis, "headlight", ambient="0.34 0.34 0.34", diffuse="0.30 0.30 0.30", specular="0.05 0.05 0.05")
    ET.SubElement(vis, "quality", shadowsize="4096")

    root.append(copy.deepcopy(arm.find("default")))
    asset = ET.SubElement(root, "asset")
    for a in arm.find("asset"):
        asset.append(copy.deepcopy(a))
    for name, kw in MATERIALS.items():
        ET.SubElement(asset, "material", name=name, **kw)

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", name="sun", pos="0.2 0.5 1.5", dir="-0.1 -0.2 -1", directional="false",
                  diffuse="0.45 0.45 0.45", specular="0.1 0.1 0.1", castshadow="true", cutoff="90")
    ET.SubElement(wb, "light", name="fill", pos="-0.8 0.6 1.0", dir="0.6 -0.3 -0.8", diffuse="0.15 0.15 0.15",
                  specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.74", material="floor")
    ET.SubElement(wb, "geom", name="table_top", type="box", size=" ".join(map(str, TABLE_HALF)),
                  pos=f"{TABLE_CENTER[0]} {TABLE_CENTER[1]} {-TABLE_HALF[2]}", material="table_dark")
    ET.SubElement(wb, "geom", name="mat", type="box", size=" ".join(map(str, MAT_HALF)),
                  pos=f"{MAT_CENTER[0]} {MAT_CENTER[1]} {MAT_HALF[2]}", euler=f"0 0 {math.radians(MAT_YAW_DEG):.5f}",
                  material="mat_grey")
    ET.SubElement(wb, "geom", name="wall", type="box", size="2.0 0.01 1.2", pos="0 -0.45 0.46", material="wall")
    for name, c in CAMS.items():
        if CAM_PARENT[name] == "world":
            ET.SubElement(wb, "camera", name=name, **cam_attrs(c))

    # the arm, all black
    base = copy.deepcopy(arm.find("worldbody").find("body"))
    base.set("pos", " ".join(map(str, BASE_POS)))
    base.set("quat", BASE_QUAT)
    for g in base.iter("geom"):
        if g.get("class", "") in ("visual", "motor_visual"):
            g.set("material", "arm_black")
    jaw = next(b for b in base.iter("body") if b.get("name") == "Fixed_Jaw")
    ET.SubElement(jaw, "site", name="tcp", pos="-0.012 -0.080 0", size="0.004", group="3", rgba="1 0 0 0")
    if CAM_MOUNT is not None:
        ET.SubElement(jaw, "geom", name="cam_mount", type="box", size=" ".join(map(str, CAM_MOUNT["half"])),
                      pos=" ".join(map(str, CAM_MOUNT["pos"])), material="arm_black", contype="0", conaffinity="0",
                      group="2")
    for name, c in CAMS.items():
        if CAM_PARENT[name] == "Fixed_Jaw":
            ET.SubElement(jaw, "camera", name=name, **cam_attrs(c))
    wb.append(base)

    # objects as mocap bodies so replay.py can move them without physics
    cup = ET.SubElement(wb, "body", name="cup", pos=f"{CUP['xy'][0]} {CUP['xy'][1]} 0", mocap="true")
    cup_geoms(cup)
    blk = ET.SubElement(wb, "body", name="block", pos=f"{BLOCK['xy'][0]} {BLOCK['xy'][1]} {BLOCK['half'][2]}",
                        euler=f"0 0 {math.radians(BLOCK['yaw_deg']):.5f}", mocap="true")
    block_geoms(blk)

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
