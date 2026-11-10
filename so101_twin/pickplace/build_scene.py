"""Single SO-101 pick-place twin scene: one Menagerie SO-100 arm on a white table, a transparent box with an
orange rim to the arm's left (-x), a small pink lego brick in front of the arm (-y), two fixed cameras "top"
and "side" matching the real so101_pickplace_real set (dataset cameras "up" and "side").

    .venv/bin/python so101_twin/pickplace/build_scene.py        # writes so101_pickplace.xml next to this file

World frame: arm base plate at the origin on the table top (z=0). The Menagerie arm reaches along -y at
qpos=0, which is toward the "top" camera (the real "up" camera is a front-elevated view, the gripper points
at it in the rest pose). The "side" camera sits on the box side (-x) and looks across the table along +x.

Everything matched to the real bench is a parameter below. CAM_TOP / CAM_SIDE hold the values found by
fit_cameras.py (silhouette-IoU fit); fit_cameras.py also writes cameras.json, which (when present) overrides
them so the fit never has to edit this file.
"""
from __future__ import annotations

import copy
import json
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ARM_XML = HERE / "assets" / "so_arm100" / "so_arm100.xml"
OUT_XML = HERE / "so101_pickplace.xml"
CAMERAS_JSON = HERE / "cameras.json"

W, H = 640, 480
RES = f"{W} {H}"

# ---- bench parameters -----------------------------------------------------------------------------------
# Cameras: pos (m), lookat (m), roll (deg, about the viewing axis), fovy (deg). Initial guesses by eye; the
# fitted values from fit_cameras.py are written to cameras.json and copied back here when the fit is accepted.
CAM_TOP = dict(pos=[-0.0272, -0.4157, 0.5580], lookat=[0.0104, -0.0828, 0.0407], roll=-3.43, fovy=42.03)   # fitted, IoU 0.70
CAM_SIDE = dict(pos=[-0.3859, -0.2566, 0.1461], lookat=[-0.0070, -0.0677, 0.0245], roll=5.51, fovy=46.33)  # fitted, IoU 0.53

TABLE_HALF = (0.45, 0.35)      # table top half extents (m)
TABLE_CENTER_Y = -0.25         # the arm is clamped near the far edge: far edge at TABLE_CENTER_Y + TABLE_HALF[1] = +0.10
TABLE_THICK = 0.02
# Transparent box with orange bottom rim, to the arm's left (-x). Dimensions are guesses from the frames.
BOX_CENTER = (-0.15, 0.02)     # (x, y) of the box centre on the table
BOX_SIZE = (0.13, 0.11, 0.06)  # full outer size (x, y, z)
BOX_WALL = 0.002
RIM = 0.006                    # orange rim: square section side (m), runs along the bottom outer edge
# Pink 2x4 lego brick (31.8 x 15.8 x 9.6 mm + studs). Position as in the first frames (in front of the arm).
BRICK_POS = (-0.075, -0.23)
BRICK_SIZE = (0.0159, 0.0079, 0.0055)
BRICK_YAW_DEG = 90.0

# The real SO-101 base/shoulder housing is bulkier than the Menagerie SO-100 base mesh; an extra dark box on the
# Base body (centre xyz, half-size xyz, in the Base frame) approximates it for the silhouettes. None disables it.
BASE_BLOCK = None

# Arm look: the real SO-101 is printed black; the fixed jaw is white (teal tip), the moving jaw black.
ARM_RGBA = "0.07 0.07 0.075 1"
JAW_RGBA = "0.92 0.92 0.9 1"
# ------------------------------------------------------------------------------------------------------------


def look_quat(pos, lookat, roll_deg=0.0) -> np.ndarray:
    """MuJoCo camera quaternion (wxyz) for a camera at pos looking at lookat, world z up, rolled about view axis."""
    pos, lookat = np.asarray(pos, float), np.asarray(lookat, float)
    f = lookat - pos
    f /= np.linalg.norm(f)                     # camera looks along -z of its frame
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c, s = np.cos(np.deg2rad(roll_deg)), np.sin(np.deg2rad(roll_deg))
    r, u = c * r + s * u, -s * r + c * u
    R = np.stack([r, u, -f], axis=1)           # columns = camera x, y, z in world
    return mat2quat(R)


def mat2quat(R: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def camera_params() -> dict:
    cams = {"top": dict(CAM_TOP), "side": dict(CAM_SIDE)}
    if CAMERAS_JSON.exists():
        for k, v in json.loads(CAMERAS_JSON.read_text()).items():
            if k in cams:
                cams[k].update({kk: v[kk] for kk in ("pos", "lookat", "roll", "fovy") if kk in v})
    return cams


def cam_attrs(c: dict) -> dict:
    q = look_quat(c["pos"], c["lookat"], c.get("roll", 0.0))
    return dict(pos=" ".join(f"{x:.5f}" for x in c["pos"]), quat=" ".join(f"{x:.6f}" for x in q),
                fovy=f"{c['fovy']:.3f}", resolution=RES)


def build(cams: dict | None = None) -> str:
    cams = cams or camera_params()
    arm = ET.parse(ARM_XML).getroot()

    root = ET.Element("mujoco", model="so101_pickplace_twin")
    ET.SubElement(root, "compiler", angle="radian", meshdir="assets/so_arm100", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(W), offheight=str(H))
    ET.SubElement(vis, "headlight", ambient="0.42 0.42 0.42", diffuse="0.22 0.22 0.22", specular="0.03 0.03 0.03")
    ET.SubElement(vis, "quality", shadowsize="4096")

    root.append(copy.deepcopy(arm.find("default")))

    asset = ET.SubElement(root, "asset")
    for a in arm.find("asset"):
        a = copy.deepcopy(a)
        if a.tag == "material" and a.get("name") == "white":
            a.set("rgba", ARM_RGBA)          # the Menagerie "white" links are black on the real SO-101
        if a.tag == "material" and a.get("name") == "black":
            a.set("rgba", ARM_RGBA)
        asset.append(a)
    ET.SubElement(asset, "texture", type="skybox", name="sky", builtin="flat", rgb1="0.72 0.72 0.72",
                  rgb2="0.72 0.72 0.72", width="32", height="32")
    for name, kw in {
        "jaw_white": dict(rgba=JAW_RGBA, specular="0.2", shininess="0.3"),
        "worktop": dict(rgba="0.86 0.86 0.85 1", reflectance="0.0", specular="0.1", shininess="0.1"),
        "wall": dict(rgba="0.70 0.70 0.70 1"),
        "floor": dict(rgba="0.75 0.75 0.75 1"),
        "box_glass": dict(rgba="0.85 0.92 0.97 0.14", specular="0.4", shininess="0.5"),
        "box_rim": dict(rgba="0.95 0.55 0.10 1"),
        "brick": dict(rgba="0.98 0.30 0.42 1", specular="0.3", shininess="0.4"),
    }.items():
        ET.SubElement(asset, "material", name=name, **kw)

    wb = ET.SubElement(root, "worldbody")
    tx, ty = TABLE_HALF
    ET.SubElement(wb, "geom", name="table_top", type="box", size=f"{tx} {ty} {TABLE_THICK / 2}",
                  pos=f"0 {TABLE_CENTER_Y} {-TABLE_THICK / 2}", material="worktop")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.1", pos="0 0 -0.75", material="floor")
    ET.SubElement(wb, "geom", name="wall_back", type="box", size="2 0.01 1.2", pos="0 1.0 0.45", material="wall")
    ET.SubElement(wb, "geom", name="wall_right", type="box", size="0.01 2 1.2", pos="1.2 0 0.45", material="wall")
    # lights: soft, mostly from above and the front like the real room
    ET.SubElement(wb, "light", name="key", pos="0.3 -0.6 1.5", dir="-0.2 0.4 -1", diffuse="0.22 0.22 0.22",
                  specular="0.1 0.1 0.1", castshadow="true")
    ET.SubElement(wb, "light", name="fill", pos="-0.8 -0.3 1.0", dir="0.6 0.2 -0.8", diffuse="0.2 0.2 0.2",
                  specular="0 0 0", castshadow="false")

    for name, c in cams.items():
        ET.SubElement(wb, "camera", name=name, **cam_attrs(c))

    # arm: Menagerie body tree, base plate on the table at the origin
    base = copy.deepcopy(arm.find("worldbody").find("body"))
    base.set("pos", "0 0 0")
    for g in base.iter("geom"):
        if g.get("mesh") == "Fixed_Jaw":
            g.set("material", "jaw_white")
    if BASE_BLOCK is not None:
        c, hs = BASE_BLOCK
        blk = ET.Element("geom", name="base_block", type="box", pos=" ".join(map(str, c)), size=" ".join(map(str, hs)),
                         **{"class": "visual"})
        base.insert(0, blk)
    wb.append(base)

    # box: floor + four thin transparent walls + orange rim along the bottom edges
    bx, by = BOX_CENTER
    sx, sy, sz = BOX_SIZE
    box = ET.SubElement(wb, "body", name="box", pos=f"{bx} {by} 0")
    ET.SubElement(box, "geom", name="box_floor", type="box", size=f"{sx / 2} {sy / 2} {BOX_WALL}", pos=f"0 0 {BOX_WALL}",
                  material="box_glass", contype="0", conaffinity="0")
    for nm, pos, size in (
        ("box_nx", (-sx / 2 + BOX_WALL, 0, sz / 2), (BOX_WALL, sy / 2, sz / 2)),
        ("box_px", (sx / 2 - BOX_WALL, 0, sz / 2), (BOX_WALL, sy / 2, sz / 2)),
        ("box_ny", (0, -sy / 2 + BOX_WALL, sz / 2), (sx / 2, BOX_WALL, sz / 2)),
        ("box_py", (0, sy / 2 - BOX_WALL, sz / 2), (sx / 2, BOX_WALL, sz / 2)),
    ):
        ET.SubElement(box, "geom", name=nm, type="box", size=" ".join(map(str, size)), pos=" ".join(map(str, pos)),
                      material="box_glass", contype="0", conaffinity="0")
    r = RIM / 2
    for nm, pos, size in (
        ("rim_nx", (-sx / 2 + r, 0, r), (r, sy / 2, r)),
        ("rim_px", (sx / 2 - r, 0, r), (r, sy / 2, r)),
        ("rim_ny", (0, -sy / 2 + r, r), (sx / 2, r, r)),
        ("rim_py", (0, sy / 2 - r, r), (sx / 2, r, r)),
    ):
        ET.SubElement(box, "geom", name=nm, type="box", size=" ".join(map(str, size)), pos=" ".join(map(str, pos)),
                      material="box_rim", contype="0", conaffinity="0")

    # brick (static; its real position varies per episode and is not tracked)
    px, py = BRICK_POS
    hx, hy, hz = BRICK_SIZE
    brick = ET.SubElement(wb, "body", name="brick", pos=f"{px} {py} {hz}", euler=f"0 0 {np.deg2rad(BRICK_YAW_DEG)}")
    ET.SubElement(brick, "geom", name="brick", type="box", size=f"{hx} {hy} {hz}", material="brick", contype="0",
                  conaffinity="0")
    for i in range(4):          # studs
        for j in range(2):
            ET.SubElement(brick, "geom", name=f"stud_{i}{j}", type="cylinder", size="0.0024 0.0009",
                          pos=f"{(i - 1.5) * 0.008} {(j - 0.5) * 0.008} {hz + 0.0009}", material="brick",
                          contype="0", conaffinity="0")

    root.append(copy.deepcopy(arm.find("actuator")))
    root.append(copy.deepcopy(arm.find("contact")))
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="rest", qpos="0 -3.32 3.11 1.18 0 -0.174", ctrl="0 -3.32 3.11 1.18 0 -0.174")

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def load_model(cams: dict | None = None):
    """Compile the scene in memory (meshes resolved against this directory)."""
    import mujoco

    xml = build(cams).replace('meshdir="assets/so_arm100"', f'meshdir="{HERE / "assets" / "so_arm100"}"')
    return mujoco.MjModel.from_xml_string(xml)


if __name__ == "__main__":
    import mujoco

    OUT_XML.write_text(build() + "\n")
    m = mujoco.MjModel.from_xml_path(str(OUT_XML))
    print(f"wrote {OUT_XML}: nq={m.nq} nu={m.nu} ncam={m.ncam} ngeom={m.ngeom}")
    print("cameras:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)])
    print("joints:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)])
