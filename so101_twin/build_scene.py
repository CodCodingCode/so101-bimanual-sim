"""Build the bimanual SO-100/101 workcell MJCF: two Menagerie SO-100 arms in the YAM workcell.

Generates bimanual_so100_workcell.xml next to this file. Everything that will later be matched to the
real bench (arm base poses, camera poses and intrinsics, table height, object set) is a parameter at
the top so calibration (Phase 2 of SIM2REAL_DIGITAL_TWIN_PLAN.md) only edits numbers here.

    python so101_twin/build_scene.py            # write the XML
    python so101_twin/render_cams.py            # render top / left wrist / right wrist to out/

Arm model: mujoco_menagerie trs_so_arm100 (Apache-2.0, assets/so_arm100/LICENSE). The SO-101 differs
from the SO-100 in motors and small bracket changes, not in link lengths; replace the meshes when an
SO-101 MJCF is available and keep the body/joint names.
"""
from __future__ import annotations

import copy
import pathlib
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
ARM_XML = HERE / "assets" / "so_arm100" / "so_arm100.xml"
OUT_XML = HERE / "bimanual_so100_workcell.xml"

# ---- bench parameters (edit these during calibration) -------------------------------------------
TABLE_Z = 0.73            # top surface of the table (m); table box half-height 0.02 -> surface at 0.75
ARM_Y = -0.22             # arm bases sit along the near edge, facing +y (into the scene)
ARM_X = 0.20              # half the distance between the two bases
ARM_YAW_QUAT = "0 0 0 1"   # Menagerie arm reaches along -y at identity; 180 deg about z makes it reach +y
# Top camera: same mast as the YAM cell, looking down and forward at the work area.
CAM_TOP = dict(pos="0 -0.42 1.18", quat="0.92388 0.382683 0 0", fovy="58")
# Wrist camera, expressed in the Fixed_Jaw body frame. The SO-101 wrist mount puts the camera above the
# fixed jaw, looking along the jaw toward the fingertips. Tune on the render until the fingertips sit
# at the bottom of the frame like the real wrist view.
# Fixed_Jaw frame: fingers extend along -y (pads at y -0.057..-0.101), the moving jaw sits at -x, so the
# outer face of the fixed jaw (where the real mount goes) is +x. xyaxes = camera x then camera y; the view
# direction is -z. Here: x_cam = z_jaw, y_cam tilted 25 deg so the view looks along -y and down toward -x.
CAM_WRIST = dict(pos="0.08 0.01 0", xyaxes="0 0 1 0.94 -0.342 0", fovy="70")  # 20 deg tilt toward the fingers
RES = "640 480"
# Objects: placeholders until the real object set is chosen. Keep 2-3, model them exactly.
OBJECTS = """
    <body name="block_red" pos="-0.12 0.08 {z}">
      <freejoint/>
      <geom name="block_red" size="0.0175 0.0175 0.0175" type="box" material="prop_red" mass="0.03"/>
    </body>
    <body name="block_blue" pos="0.12 0.08 {z}">
      <freejoint/>
      <geom name="block_blue" size="0.0175 0.0175 0.0175" type="box" material="prop_blue" mass="0.03"/>
    </body>
    <body name="bin" pos="0 0.2 {bz}">
      <geom name="bin_nx" size="0.006 0.06 0.02" pos="-0.08 0 0.02" type="box" material="prop_bin"/>
      <geom name="bin_px" size="0.006 0.06 0.02" pos="0.08 0 0.02" type="box" material="prop_bin"/>
      <geom name="bin_ny" size="0.08 0.006 0.02" pos="0 -0.06 0.02" type="box" material="prop_bin"/>
      <geom name="bin_py" size="0.08 0.006 0.02" pos="0 0.06 0.02" type="box" material="prop_bin"/>
      <geom name="bin_floor" size="0.08 0.06 0.003" pos="0 0 0.003" type="box" material="prop_bin"/>
    </body>
"""

NAMED = ("body", "joint", "geom", "site", "camera", "light")


def prefix_tree(root: ET.Element, prefix: str) -> None:
    """Prefix every body/joint/geom/site name and every joint= reference under root (meshes and classes shared)."""
    for el in root.iter():
        if el.tag in NAMED and el.get("name"):
            el.set("name", f"{prefix}_{el.get('name')}")
        if el.get("joint"):
            el.set("joint", f"{prefix}_{el.get('joint')}")
        if el.get("body"):
            el.set("body", f"{prefix}_{el.get('body')}")


def arm_instance(arm_root: ET.Element, side: str, x: float) -> tuple[ET.Element, list[ET.Element]]:
    wb = arm_root.find("worldbody")
    base = copy.deepcopy(wb.find("body"))
    base.set("pos", f"{x} {ARM_Y} {TABLE_Z + 0.02}")
    base.set("quat", ARM_YAW_QUAT)
    # wrist camera + tool-centre-point site on the fixed jaw
    jaw = next(b for b in base.iter("body") if b.get("name") == "Fixed_Jaw")
    cam = ET.SubElement(jaw, "camera", name="cam_wrist", resolution=RES, **CAM_WRIST)
    ET.SubElement(jaw, "site", name="tcp", pos="0.0 -0.09 0", size="0.005", group="3", rgba="1 0.2 0.2 0.6")
    prefix_tree(base, side)
    acts = [copy.deepcopy(a) for a in arm_root.find("actuator")]
    for a in acts:
        a.set("name", f"{side}_{a.get('name')}")
        a.set("joint", f"{side}_{a.get('joint')}")
    return base, acts


def build() -> str:
    arm = ET.parse(ARM_XML).getroot()
    arm_default = arm.find("default")
    arm_asset = arm.find("asset")

    root = ET.Element("mujoco", model="bimanual_so100_workcell")
    ET.SubElement(root, "compiler", angle="radian", meshdir="assets/so_arm100", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1280", offheight="960")
    ET.SubElement(vis, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6", specular="0.1 0.1 0.1")

    root.append(copy.deepcopy(arm_default))

    asset = ET.SubElement(root, "asset")
    for m in arm_asset:
        asset.append(copy.deepcopy(m))
    ET.SubElement(asset, "texture", type="skybox", name="skybox", builtin="gradient", rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(asset, "texture", type="2d", name="groundplane", builtin="checker", mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
    for name, kw in {
        "groundplane": dict(texture="groundplane", texrepeat="5 5", texuniform="true", reflectance="0.2"),
        "table": dict(reflectance="0.05", rgba="0.55 0.47 0.38 1"),
        "worktop": dict(reflectance="0.02", rgba="0.9 0.9 0.88 1"),
        "wall": dict(rgba="0.93 0.93 0.92 1"),
        "mast": dict(rgba="0.25 0.25 0.28 1"),
        "prop_red": dict(rgba="0.78 0.2 0.18 1"),
        "prop_blue": dict(rgba="0.16 0.35 0.72 1"),
        "prop_bin": dict(rgba="0.3 0.3 0.34 1"),
    }.items():
        ET.SubElement(asset, "material", name=name, **kw)

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "geom", name="floor", size="0 0 0.05", type="plane", material="groundplane")
    ET.SubElement(wb, "geom", name="table_top", size="0.4 0.37 0.02", pos=f"0 0 {TABLE_Z}", type="box", material="worktop")
    for nm, (x, y) in {"nn": (-0.35, -0.32), "np": (-0.35, 0.32), "pn": (0.35, -0.32), "pp": (0.35, 0.32)}.items():
        ET.SubElement(wb, "geom", name=f"table_leg_{nm}", size="0.03 0.03 0.355", pos=f"{x} {y} 0.355", type="box", material="table")
    ET.SubElement(wb, "geom", name="wall_back", size="0.37 0.01 0.275", pos="0 0.34 1.025", type="box", material="wall")
    ET.SubElement(wb, "geom", name="cam_top_post", size="0.022 0.6", pos="0 -0.52 0.6", quat="0 1 0 0", type="capsule", contype="0", conaffinity="0", material="mast")
    ET.SubElement(wb, "geom", name="cam_top_boom", size="0.018 0.05", pos="0 -0.47 1.18", quat="0.707107 0.707107 0 0", type="capsule", contype="0", conaffinity="0", material="mast")
    ET.SubElement(wb, "camera", name="cam_top", resolution=RES, **CAM_TOP)
    for i, (x, y) in enumerate([(-0.9, -0.9), (0.9, -0.9), (-0.9, 0.9), (0.9, 0.9)]):
        ET.SubElement(wb, "light", name=f"fill_{i}", pos=f"{x} {y} 2.2", cutoff="75", exponent="8", diffuse="0.35 0.35 0.35", specular="0.05 0.05 0.05")

    acts_all = []
    for side, x in (("left", -ARM_X), ("right", ARM_X)):
        base, acts = arm_instance(arm, side, x)
        wb.append(base)
        acts_all += acts

    objs = ET.fromstring(f"<w>{OBJECTS.format(z=TABLE_Z + 0.02 + 0.0175, bz=TABLE_Z + 0.02)}</w>")
    for b in objs:
        wb.append(b)

    act = ET.SubElement(root, "actuator")
    for a in acts_all:
        act.append(a)

    # Home keyframe: both arms folded up and back, jaws open; objects at their declared poses.
    home_arm = "0 -1.57 1.57 1.2 0 0.5"
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="home",
                  qpos=f"{home_arm} {home_arm} -0.12 0.08 {TABLE_Z + 0.0375} 1 0 0 0 0.12 0.08 {TABLE_Z + 0.0375} 1 0 0 0",
                  ctrl=f"{home_arm} {home_arm}")

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


if __name__ == "__main__":
    xml = build()
    OUT_XML.write_text(xml + "\n")
    import mujoco  # noqa: E402  (import here so the file is buildable without mujoco)

    m = mujoco.MjModel.from_xml_path(str(OUT_XML))  # meshdir resolves relative to the XML file
    print(f"wrote {OUT_XML.name}: nq={m.nq} nu={m.nu} ncam={m.ncam} nbody={m.nbody}")
    print("cameras:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)])
    print("actuators:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])
