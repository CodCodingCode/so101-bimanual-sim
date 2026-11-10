"""legocup2 COPY of so101_twin/common/arm.py (do not edit the original; changes marked LEGOCUP2):
  - the wrist-camera module + its mount bracket are geom group 3 (Arm.hide_groups[cam] hides groups per camera, so a
    camera placed inside the module does not see the module itself);
  - gripper offset grip_off (rad) in addition to grip_scale.

Official SO-101 arm (TheRobotStudio SO-ARM100 Simulation/SO101) as a building block for every twin scene.

    from so101_twin.common import arm          # or sys.path.insert(0, ".../so101_twin/common"); import arm
    spec = arm.ArmSpec(calib="new", wrist_mount=True, colors={"body": (0.95, 0.95, 0.95, 1)})
    cams = [arm.CamSpec("top", mount="world", pos=(0.4, 0.3, 0.2), quat=arm.look_quat((0.4, 0.3, 0.2), (0, 0.2, 0))),
            arm.CamSpec("wrist", mount="gripper", pos=(0, 0.05, -0.03), quat=(1, 0, 0, 0))]
    a = arm.Arm(arm.build_mjcf(spec, cams))
    a.set_state(lerobot_state)                  # 6 floats, LeRobot SO-101 degrees + gripper 0..100
    rgb = a.render("top"); m = a.arm_mask("top")

Normalised arm frame (both calibrations): the base's bottom face is at z = 0, the shoulder-pan axis is the z axis
through the origin, and at shoulder_pan = 0 the arm reaches along +y. ArmSpec.base_pos / base_yaw_deg place that
frame in the world (the existing Menagerie-based twins put the pan axis at (0, 0.0452, 0) reaching +y).

Joint mapping (LeRobot state -> model qpos, radians), verified by rendering against the real frames (see README.md):
  calib="new"  (LeRobot so101_follower, range-middle calibration):  q = deg2rad(state[:5]);  same signs
  calib="old"  (LeRobot v2.1 'main_*' SO-100 calibration):          q = deg2rad(OLD_SIGNS * state[:5])
  gripper (both): q_grip = GRIP_LO + state[5] / 100 * (GRIP_HI - GRIP_LO),  GRIP = [-0.1745, 1.7453] rad
Small per-joint offsets (degrees, added after the sign) are a fit parameter (fit.py), default 0.
ArmSpec(model="so100") swaps in the Menagerie SO-100 meshes the builder twins used (same normalised frame; joint map
q = deg2rad(SIGNS_SO100 * state + NOMINAL_SO100 + offset)), for an apples-to-apples comparison of the two meshes.

Mounts for cameras: 'world' (worldbody), 'gripper' (the wrist-roll body that carries the fixed jaw and the wrist
camera mount), 'shoulder' (the shoulder-pan body: rotates with pan only). A camera's pos/quat are in that body's
frame (MuJoCo camera convention: looks along -z, +y up in the image).
"""
from __future__ import annotations

import copy
import dataclasses
import math
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets" / "so101"
MESHDIR = ASSETS / "assets"
XMLS = {"new": ASSETS / "so101_new_calib.xml", "old": ASSETS / "so101_old_calib.xml"}
CAMERA_XML = ASSETS / "so101_new_calib_camera.xml"

SO100_XML = HERE.parent / "assets" / "so_arm100" / "so_arm100.xml"   # Menagerie SO-100 (what the builders used)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_NAMES = {"so101": JOINTS, "so100": ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")}
GRIP_LO, GRIP_HI = -0.17453, 1.74533      # gripper / Jaw hinge range (rad) in both models
# LeRobot degrees -> model radians:  q = deg2rad(SIGN * state + NOMINAL + offset)   (verified, see the docstring)
SIGNS = {"new": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),          # official SO-101, so101_new_calib.xml
         "old": np.array([1.0, -1.0, 1.0, 1.0, -1.0])}        # official SO-101, so101_old_calib.xml
SIGNS_SO100 = {"new": np.array([-1.0, 1.0, 1.0, 1.0, 1.0]), "old": np.array([-1.0, -1.0, 1.0, 1.0, -1.0])}
NOMINAL_SO100 = {"new": np.array([0.0, -90.0, 90.0, 0.0, 0.0]), "old": np.zeros(5)}
GRIP_SIGN = {"new": 1.0, "old": 1.0}

# normalisation of each model file into the arm frame described above: rotate about z, then translate
_NORM = {("so101", "old"): dict(yaw=180.0, shift=(0.0208, -0.0231, -0.0300)),
         ("so101", "new"): dict(yaw=90.0, shift=(0.0, -0.0388, 0.0024)),
         ("so100", "old"): dict(yaw=180.0, shift=(0.0, -0.0452, 0.0)),
         ("so100", "new"): dict(yaw=180.0, shift=(0.0, -0.0452, 0.0))}
_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
MOUNT_BODY = {"gripper": "gripper", "shoulder": "shoulder", "world": None}
MOUNT_BODY_SO100 = {"gripper": "Fixed_Jaw", "shoulder": "Rotation_Pitch", "world": None}

PARTS = {   # mesh -> colour group
    "base_so101_v2": "body", "base_motor_holder_so101_v1": "body", "waveshare_mounting_plate_so101_v2": "body",
    "motor_holder_so101_base_v1": "body", "rotation_pitch_so101_v1": "body", "upper_arm_so101_v1": "body",
    "under_arm_so101_v1": "body", "motor_holder_so101_wrist_v1": "body", "wrist_roll_pitch_so101_v2": "body",
    "wrist_roll_follower_so101_v1": "jaw", "moving_jaw_so101_v1": "jaw",
    "sts3215_03a_v1": "motor", "sts3215_03a_no_horn_v1": "motor",
    "wrist_camera_mount_so101_v1": "cam_mount", "wrist_camera_so101_v1": "camera",
}
DEFAULT_COLORS = {"body": (1.0, 0.82, 0.12, 1), "jaw": None, "motor": (0.1, 0.1, 0.1, 1), "cam_mount": None,
                  "camera": (0.1, 0.1, 0.1, 1)}   # None -> same as body


@dataclasses.dataclass
class ArmSpec:
    calib: str = "new"                    # 'new' | 'old'  (LeRobot calibration convention of the dataset)
    model: str = "so101"                  # 'so101' official SO-101 meshes | 'so100' Menagerie SO-100 (comparison)
    wrist_mount: bool = False             # add the official wrist-camera mount + camera module (from the _camera xml)
    colors: dict = dataclasses.field(default_factory=dict)   # overrides of DEFAULT_COLORS: body/jaw/motor/cam_mount/camera
    base_pos: tuple = (0.0, 0.0, 0.0)     # world position of the normalised arm frame
    base_yaw_deg: float = 0.0             # rotation of the arm frame about world z (0: pan zero reaches +y)
    specular: float = 0.2
    shininess: float = 0.3


@dataclasses.dataclass
class CamSpec:
    name: str
    mount: str = "world"                  # 'world' | 'gripper' | 'shoulder'
    pos: tuple = (0.0, 0.0, 1.0)          # in the mount body's frame
    quat: tuple = (1.0, 0.0, 0.0, 0.0)    # (w, x, y, z) MuJoCo camera frame in the mount body's frame
    fovy: float = 45.0


# ---------------------------------------------------------------------------------------------------- math helpers
def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def mat_from_quat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def quat_from_rotvec(v):
    v = np.asarray(v, float)
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.array([1.0, 0, 0, 0])
    return np.concatenate([[math.cos(a / 2)], math.sin(a / 2) * v / a])


def look_quat(pos, target, roll_deg: float = 0.0, up=(0, 0, 1)):
    """MuJoCo camera quat (looks along -z) at pos looking at target, image-up close to `up`, then rolled."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(f, (0, 1, 0))
    right /= np.linalg.norm(right)
    u = np.cross(right, f)
    r = math.radians(roll_deg)
    right, u = math.cos(r) * right + math.sin(r) * u, -math.sin(r) * right + math.cos(r) * u
    return quat_from_mat(np.stack([right, u, -f], axis=1))


def ypr_quat(yaw: float, pitch: float, roll: float):
    """Camera quat from the yaw/pitch/roll convention used by every existing so101_twin/<scene>/build_scene.py."""
    y, p, r = map(math.radians, (yaw, pitch, roll))
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    right, up = math.cos(r) * right + math.sin(r) * up, -math.sin(r) * right + math.cos(r) * up
    return quat_from_mat(np.stack([right, up, np.cross(right, up)], axis=1))


def _fmt(v):
    return " ".join(f"{float(x):.7g}" for x in v)


# ---------------------------------------------------------------------------------------------------- MJCF
def lerobot_to_qpos(state, calib: str, offsets_deg=None, grip_scale: float = 1.0, model: str = "so101",
                    scales=None, grip_off: float = 0.0) -> np.ndarray:
    """(6,) LeRobot state [deg x5, gripper 0..100] -> (6,) model qpos (radians) in JOINTS order.

    scales: optional (5,) degrees per state unit (1 = the state is in degrees). Datasets recorded with LeRobot's
    normalised mode (-100..100 of the calibrated travel, use_degrees=False) need ~0.9-1.1 here; fit.py FIT_SCALES."""
    s = np.asarray(state, float)
    off = np.zeros(5) if offsets_deg is None else np.asarray(offsets_deg, float)[:5]
    sj = s[:5] if scales is None else s[:5] * np.asarray(scales, float)[:5]
    if model == "so100":
        q = np.deg2rad(SIGNS_SO100[calib] * sj + NOMINAL_SO100[calib] + off)
    else:
        q = np.deg2rad(SIGNS[calib] * sj + off)
    g = GRIP_LO + grip_off + GRIP_SIGN[calib] * grip_scale * np.clip(s[5], 0, 100) / 100.0 * (GRIP_HI - GRIP_LO)  # LEGOCUP2 grip_off
    return np.concatenate([q, [g]])


def _arm_body_so100(spec: ArmSpec):
    """Menagerie SO-100 (as used by the builder twins): same normalised frame, recoloured, collision geoms dropped."""
    src = ET.parse(SO100_XML).getroot()
    base = copy.deepcopy(src.find("worldbody").find("body"))
    meshes = [copy.deepcopy(m) for m in src.find("asset").findall("mesh")]
    default = copy.deepcopy(src.find("default"))
    for b in base.iter("body"):
        for g in list(b.findall("geom")):
            cls = g.get("class", "")
            if cls in ("collision", "finger_collision"):
                b.remove(g)
                continue
            part = "motor" if cls == "motor_visual" else ("jaw" if g.get("mesh") in ("Fixed_Jaw", "Moving_Jaw") else "body")
            g.set("material", "so101_" + part)
    for b in base.iter("body"):   # bodies whose mass came from the (dropped) collision geoms
        if b.find("inertial") is None:
            b.insert(0, ET.Element("inertial", pos="0 0 0.02", mass="0.2", diaginertia="1e-4 1e-4 1e-4"))
    jaw = next(b for b in base.iter("body") if b.get("name") == "Fixed_Jaw")
    ET.SubElement(jaw, "site", name="tcp", pos="-0.012 -0.080 0", size="0.004", group="5")
    meshes = [m for m in meshes if "Collision" not in m.get("file", "")]
    for m in meshes:
        m.set("file", str((SO100_XML.parent / m.get("file")).resolve()))
    return base, meshes, default


def _arm_body(spec: ArmSpec):
    """(normalised arm root body, mesh asset elements, default element or None) from the model xml."""
    if spec.model == "so100":
        base, meshes, default = _arm_body_so100(spec)
        return _wrap(spec, base), meshes, default
    src = ET.parse(XMLS[spec.calib]).getroot()
    base = copy.deepcopy(src.find("worldbody").find("body"))
    meshes = [copy.deepcopy(m) for m in src.find("asset").findall("mesh")]
    if spec.wrist_mount:
        cam_src = ET.parse(CAMERA_XML).getroot()
        mount = next(b for b in cam_src.iter("body") if b.get("name") == "wrist_camera_mount")
        grip = next(b for b in base.iter("body") if b.get("name") == "gripper")
        # the gripper body's local frame is identical in old/new calib (same child geom poses), so graft as is
        jaw = next(b for b in grip.findall("body") if b.get("name") == "moving_jaw_so101_v1")
        grip.insert(list(grip).index(jaw), copy.deepcopy(mount))
        have = {m.get("file") for m in meshes}
        meshes += [copy.deepcopy(m) for m in cam_src.find("asset").findall("mesh") if m.get("file") not in have]
    # drop collision geoms and sites (never rendered), recolour visual geoms
    for b in base.iter("body"):
        for g in list(b.findall("geom")):
            if g.get("class") == "collision":
                b.remove(g)
                continue
            g.set("material", "so101_" + PARTS.get(g.get("mesh"), "body"))
        for s in list(b.findall("site")):
            b.remove(s)
    grip = next(b for b in base.iter("body") if b.get("name") == "gripper")
    # tool centre point between the jaw tips = the official 'gripperframe' site (same pose in both calibrations)
    ET.SubElement(grip, "site", name="tcp", pos="-0.0079 -0.000218 -0.0981", size="0.004", group="5")
    for b in base.iter("body"):
        b.attrib.pop("childclass", None)
    for j in base.iter("joint"):
        j.attrib.pop("class", None)
    for g in base.iter("geom"):
        g.attrib.pop("class", None)
        g.set("type", "mesh")
        g.set("contype", "0")
        g.set("conaffinity", "0")
        # LEGOCUP2: wrist camera module + bracket in group 3 so the wrist camera can hide them (Arm.hide_groups)
        g.set("group", "3" if g.get("mesh") in ("wrist_camera_so101_v1", "wrist_camera_mount_so101_v1") else "2")
    for m in meshes:
        m.set("file", str((MESHDIR / m.get("file")).resolve()))
    return _wrap(spec, base), meshes, None


def _wrap(spec: ArmSpec, base: ET.Element) -> ET.Element:
    """normalisation wrapper: arm_root (scene placement) > arm_norm (model/calib-specific) > base"""
    nrm = _NORM[(spec.model, spec.calib)]
    root = ET.Element("body", name="arm_root", pos=_fmt(spec.base_pos),
                      quat=_fmt(quat_from_rotvec([0, 0, math.radians(spec.base_yaw_deg)])))
    yq = quat_from_rotvec([0, 0, math.radians(nrm["yaw"])])
    norm = ET.SubElement(root, "body", name="arm_norm", pos=_fmt(nrm["shift"]), quat=_fmt(yq))
    base.set("pos", "0 0 0")
    base.set("quat", "1 0 0 0")
    norm.append(base)
    return root


def build_mjcf(spec: ArmSpec, cams: list[CamSpec], extras=None, model_name: str = "so101_twin",
               offwidth: int = 1280, offheight: int = 960) -> str:
    """MJCF string: one SO-101 arm (+ cameras). `extras(root, asset, worldbody)` may add scene objects/lights."""
    arm_root, meshes, default = _arm_body(spec)
    root = ET.Element("mujoco", model=model_name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    if default is not None:
        root.append(default)
    ET.SubElement(root, "option", timestep="0.002")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(offwidth), offheight=str(offheight))
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "headlight", ambient="0.32 0.32 0.32", diffuse="0.30 0.30 0.30", specular="0.05 0.05 0.05")
    asset = ET.SubElement(root, "asset")
    for m in meshes:
        asset.append(m)
    colors = dict(DEFAULT_COLORS)
    colors.update(spec.colors or {})
    for part in ("body", "jaw", "motor", "cam_mount", "camera"):
        c = colors[part] if colors[part] is not None else colors["body"]
        ET.SubElement(asset, "material", name=f"so101_{part}", rgba=_fmt(c), specular=str(spec.specular),
                      shininess=str(spec.shininess))
    wb = ET.SubElement(root, "worldbody")
    wb.append(arm_root)
    by_name = {b.get("name"): b for b in arm_root.iter("body")}
    mounts = MOUNT_BODY_SO100 if spec.model == "so100" else MOUNT_BODY
    for c in cams:
        parent = wb if c.mount == "world" else by_name[mounts[c.mount]]
        ET.SubElement(parent, "camera", name=c.name, pos=_fmt(c.pos), quat=_fmt(c.quat), fovy=f"{c.fovy:.4f}")
    if extras is not None:
        extras(root, asset, wb)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------------------------------- runtime
class Arm:
    """Compiled twin: pose from LeRobot state, render RGB / arm mask from any named camera (with lens warp)."""

    def __init__(self, xml: str, calib: str = "new", model: str = "so101"):
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.calib = calib
        self.model = model
        self.offsets_deg = np.zeros(5)
        self.scales = np.ones(5)
        self.grip_scale = 1.0
        self.grip_off = 0.0                  # LEGOCUP2
        self.hide_groups: dict[str, tuple] = {}   # LEGOCUP2: camera name -> geom groups not rendered from it
        self._r: dict[tuple, mujoco.Renderer] = {}
        self.qadr = np.array([self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                              for j in JOINT_NAMES[model]])
        root = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arm_root")
        sub = set()
        for b in range(self.m.nbody):
            x = b
            while x > 0 and x != root:
                x = self.m.body_parentid[x]
            if x == root:
                sub.add(b)
        self.arm_bodies = np.array(sorted(sub))
        self.arm_geoms = np.array([g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in sub])
        self._arm_lut = np.zeros(self.m.ngeom + 1, bool)    # index = geom id + 1 (seg id -1 = background)
        self._arm_lut[self.arm_geoms + 1] = True
        self.cam_id = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_CAMERA, i): i for i in range(self.m.ncam)}
        mujoco.mj_forward(self.m, self.d)

    # -- state
    def set_state(self, state) -> np.ndarray:
        q = lerobot_to_qpos(state, self.calib, self.offsets_deg, self.grip_scale, self.model, self.scales, self.grip_off)
        self.d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, self.d)
        return q

    def set_camera(self, name: str, pos=None, quat=None, fovy=None) -> None:
        cid = self.cam_id[name]
        if pos is not None:
            self.m.cam_pos[cid] = pos
        if quat is not None:
            self.m.cam_quat[cid] = np.asarray(quat, float) / np.linalg.norm(quat)
        if fovy is not None:
            self.m.cam_fovy[cid] = fovy
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)

    def renderer(self, w: int, h: int) -> mujoco.Renderer:
        if (w, h) not in self._r:
            self._r[(w, h)] = mujoco.Renderer(self.m, h, w)
        return self._r[(w, h)]

    # -- pinhole renders
    def render_raw(self, cam: str, w: int, h: int, seg: bool = False) -> np.ndarray:
        r = self.renderer(w, h)
        if seg:
            r.enable_segmentation_rendering()
        opt = mujoco.MjvOption()          # LEGOCUP2: group 3 (wrist camera parts) visible unless hidden for this cam
        opt.geomgroup[3] = 1
        for g in self.hide_groups.get(cam, ()):
            opt.geomgroup[g] = 0
        r.update_scene(self.d, camera=self.cam_id[cam], scene_option=opt)
        out = r.render().copy()
        if seg:
            r.disable_segmentation_rendering()
        return out

    def render(self, cam: str, w: int = 640, h: int = 480, lens: "Lens | None" = None) -> np.ndarray:
        """uint8 HxWx3; with `lens`, rendered as a padded pinhole image and warped with the radial distortion."""
        if lens is None:
            return self.render_raw(cam, w, h)
        return lens.render(self, cam, w, h, seg=False)

    def arm_mask(self, cam: str, w: int = 640, h: int = 480, lens: "Lens | None" = None) -> np.ndarray:
        """bool HxW: pixels covered by any geom of the arm (segmentation render)."""
        if lens is None:
            seg = self.render_raw(cam, w, h, seg=True)
        else:
            seg = lens.render(self, cam, w, h, seg=True)
        return self._arm_lut[seg[..., 0] + 1] & (seg[..., 1] == _GEOM)

    def close(self):
        for r in self._r.values():
            r.close()
        self._r.clear()


class Lens:
    """Radial distortion (k1, k2) + principal-point shift (cx, cy as fractions of width/height) applied as a warp.

    The camera's fovy is the pinhole vertical FOV of the undistorted image. Rendering: a pinhole image with the same
    focal length but `pad` extra border is rendered, then for every output pixel (distorted coordinates) the
    undistorted source coordinate is found (fixed-point inversion of x_d = x_u (1 + k1 r^2 + k2 r^4)) and sampled
    (nearest for segmentation, bilinear for RGB).
    """

    def __init__(self, k1=0.0, k2=0.0, cx=0.0, cy=0.0, pad: float = 0.25):
        self.k1, self.k2, self.cx, self.cy, self.pad = float(k1), float(k2), float(cx), float(cy), pad
        self._maps = {}

    def as_dict(self):
        return dict(k1=self.k1, k2=self.k2, cx=self.cx, cy=self.cy)

    def _map(self, fovy: float, w: int, h: int):
        key = (round(fovy, 6), w, h)
        if key in self._maps:
            return self._maps[key]
        f = 0.5 * h / math.tan(math.radians(fovy) / 2)
        pw, ph = int(round(w * (1 + 2 * self.pad))) // 2 * 2, int(round(h * (1 + 2 * self.pad))) // 2 * 2
        fovy_r = 2 * math.degrees(math.atan(0.5 * ph / f))
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        xd = (u - w / 2 - self.cx * w) / f
        yd = (v - h / 2 - self.cy * h) / f
        xu, yu = xd.copy(), yd.copy()
        for _ in range(8):
            r2 = xu * xu + yu * yu
            s = 1 + self.k1 * r2 + self.k2 * r2 * r2
            xu, yu = xd / s, yd / s
        su = xu * f + pw / 2 - 0.5
        sv = yu * f + ph / 2 - 0.5
        self._maps[key] = (pw, ph, fovy_r, su.astype(np.float32), sv.astype(np.float32))
        if len(self._maps) > 64:
            self._maps.pop(next(iter(self._maps)))
        return self._maps[key]

    def render(self, arm: Arm, cam: str, w: int, h: int, seg: bool):
        cid = arm.cam_id[cam]
        fovy = float(arm.m.cam_fovy[cid])
        pw, ph, fovy_r, su, sv = self._map(fovy, w, h)
        arm.m.cam_fovy[cid] = fovy_r
        try:
            img = arm.render_raw(cam, pw, ph, seg=seg)
        finally:
            arm.m.cam_fovy[cid] = fovy
        return warp(img, su, sv, nearest=seg)


def warp(img: np.ndarray, su: np.ndarray, sv: np.ndarray, nearest: bool) -> np.ndarray:
    """Sample img (H'xW'xC) at float source coords (su, sv) (pixel centres at integers)."""
    H, W = img.shape[:2]
    if nearest:
        x = np.clip(np.rint(su).astype(np.int32), 0, W - 1)
        y = np.clip(np.rint(sv).astype(np.int32), 0, H - 1)
        out = img[y, x]
        bad = (su < -0.5) | (su > W - 0.5) | (sv < -0.5) | (sv > H - 0.5)
        if bad.any():
            out = out.copy()
            out[bad] = -1 if out.dtype.kind == "i" else 0
        return out
    x0 = np.clip(np.floor(su).astype(np.int32), 0, W - 2)
    y0 = np.clip(np.floor(sv).astype(np.int32), 0, H - 2)
    fx = np.clip(su - x0, 0, 1)[..., None]
    fy = np.clip(sv - y0, 0, 1)[..., None]
    im = img.astype(np.float32)
    top = im[y0, x0] * (1 - fx) + im[y0, x0 + 1] * fx
    bot = im[y0 + 1, x0] * (1 - fx) + im[y0 + 1, x0 + 1] * fx
    return np.clip(top * (1 - fy) + bot * fy + 0.5, 0, 255).astype(np.uint8)


def mount_pose_from_world(a: Arm, mount: str, cam_pos_world, cam_xmat_world) -> tuple[np.ndarray, np.ndarray]:
    """(pos, quat) of a camera in the `mount` body frame of the CURRENTLY POSED arm `a`, from its world pose.

    Use to port a mounted camera from an old (Menagerie) twin: pose the old Sim and this Arm with the same LeRobot
    state, read the old camera's world pose (d.cam_xpos[cid], d.cam_xmat[cid]) and convert it here.
    """
    if mount == "world":
        return np.asarray(cam_pos_world, float), quat_from_mat(np.asarray(cam_xmat_world).reshape(3, 3))
    bid = mujoco.mj_name2id(a.m, mujoco.mjtObj.mjOBJ_BODY,
                            (MOUNT_BODY_SO100 if a.model == "so100" else MOUNT_BODY)[mount])
    Rb = a.d.xmat[bid].reshape(3, 3)
    pos = Rb.T @ (np.asarray(cam_pos_world, float) - a.d.xpos[bid])
    R = Rb.T @ np.asarray(cam_xmat_world, float).reshape(3, 3)
    return pos, quat_from_mat(R)
