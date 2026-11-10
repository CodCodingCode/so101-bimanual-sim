"""Shared helpers for the stack twin: joint mapping real -> sim, scene loading, offscreen rendering, real-set access.

Joint convention (LeRobot v3.0 `so101_follower` degrees mode: zero = calibration "middle" pose, arm straight up;
gripper linear 0..100):
    sim_rad[j] = deg2rad(SIGNS[j] * SCALES[j] * real[j] + OFFSETS_DEG[j])  for the five revolute joints
(SCALES = degrees per state unit; 1.0 if the state is in degrees. LeRobot's default for so101_follower is the
normalised -100..100 range of the calibrated joint travel, which for the SO-101 is close to, but not exactly,
degrees; the fit refines SCALES for lift / elbow / wrist_flex.)
    Jaw        = JAW_RANGE[0] + gripper/100 * (JAW_RANGE[1] - JAW_RANGE[0])
Derived first guess (same as so101_twin/pickplace): Pitch = lift - 90, Elbow = elbow + 90, pan flipped; the
dataset rest pose (lift -98.5, elbow +98.7, wrist +44.7) lands on the Menagerie `rest` keyframe. The offsets are
refined by fit_cameras.py --offsets and stored in joints.json (overrides the defaults below when present).
See README.md for how SIGNS / OFFSETS_DEG were verified.
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
XML = HERE / "so101_stack.xml"
JOINTS_JSON = HERE / "joints.json"
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))

LEROBOT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SIM_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
SIGNS = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])   # pan -1: real +pan swings the gripper toward +x (image-up in the top view)
OFFSETS_DEG = np.array([0.0, -90.0, 90.0, 20.0, 0.0, 0.0])   # fit_cameras.py --offsets (deg, added before deg2rad)
SCALES = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])            # deg per state unit (fit_cameras.py --offsets)
JAW_RANGE = (-0.174, 1.75)
if JOINTS_JSON.exists():                                      # fitted values written by fit_cameras.py --offsets
    import json as _json

    _j = _json.loads(JOINTS_JSON.read_text())
    SIGNS[:5] = np.asarray(_j["signs"], float)[:5]
    OFFSETS_DEG[:5] = np.asarray(_j["offsets_deg"], float)[:5]
    SCALES[:5] = np.asarray(_j.get("scales", SCALES[:5]), float)[:5]
CAMS = ("top", "side")
W, H = 640, 480


def real_to_sim(q: np.ndarray) -> np.ndarray:
    """(6,) real [deg x5, gripper 0..100] -> (6,) sim qpos [rad] in SIM_ORDER."""
    q = np.asarray(q, dtype=float)
    rad = np.deg2rad(SIGNS[:5] * SCALES[:5] * q[:5] + OFFSETS_DEG[:5])
    jaw = JAW_RANGE[0] + np.clip(q[5], 0, 100) / 100.0 * (JAW_RANGE[1] - JAW_RANGE[0])
    return np.concatenate([rad, [jaw]])


def real_set():
    from sets import Store

    return Store().frames("so101_stack_real")


class Sim:
    def __init__(self, xml: pathlib.Path = XML, w: int = W, h: int = H):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.w, self.h = w, h
        self.cam_id = {c: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMS}
        self.arm_geoms = self._arm_geoms()
        self.obj_geom = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("disc", "green", "cube")}
        mujoco.mj_forward(self.m, self.d)

    def set_object(self, name: str, xy, size=None) -> None:
        """Move a static scene object (disc: geom_pos + radius; green / cube: mocap body + box half size)."""
        gid = self.obj_geom[name]
        if name == "disc":
            self.m.geom_pos[gid][:2] = xy
            if size is not None:
                self.m.geom_size[gid][0] = size
        else:
            bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            if size is not None:
                self.m.geom_size[gid][:] = size
            self.d.mocap_pos[self.m.body_mocapid[bid]] = (xy[0], xy[1], self.m.geom_size[gid][2])
        mujoco.mj_forward(self.m, self.d)

    def seg(self, cam: str) -> np.ndarray:
        """int HxW geom ids (-1 = background) from a segmentation render."""
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera=cam)
        seg = self.r.render()[..., 0].copy()
        self.r.disable_segmentation_rendering()
        return seg

    def _arm_geoms(self) -> np.ndarray:
        base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "Base")
        ids = []
        for g in range(self.m.ngeom):
            b = self.m.geom_bodyid[g]
            while b != 0 and b != base:
                b = self.m.body_parentid[b]
            if b == base and self.m.geom_group[g] == 2:   # visual meshes only
                ids.append(g)
        return np.array(ids)

    def set_joints(self, q_real: np.ndarray) -> None:
        self.d.qpos[:6] = real_to_sim(q_real)
        self.d.ctrl[:6] = self.d.qpos[:6]
        mujoco.mj_forward(self.m, self.d)

    def set_camera(self, name: str, pos, yaw: float, pitch: float, roll: float, fovy: float) -> None:
        from build_scene import cam_xyaxes

        cid = self.cam_id[name]
        right, up = cam_xyaxes(yaw, pitch, roll)
        fwd = np.cross(up, right)   # camera looks along -z, so z = right x up = -forward
        R = np.stack([right, up, -fwd], axis=1)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).ravel())
        self.m.cam_pos[cid] = pos
        self.m.cam_quat[cid] = q
        self.m.cam_fovy[cid] = fovy

    def render(self, cam: str) -> np.ndarray:
        """RGB render. The backdrop sheets (geom group 4) are only meant for the side view and are hidden in "top"."""
        opt = mujoco.MjvOption()
        opt.geomgroup[4] = 0 if cam == "top" else 1
        self.r.update_scene(self.d, camera=cam, scene_option=opt)
        return self.r.render().copy()

    def arm_mask(self, cam: str) -> np.ndarray:
        """bool HxW: pixels covered by the arm's visual geoms (segmentation render)."""
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera=cam)
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        return np.isin(seg[..., 0], self.arm_geoms)

    def close(self):
        self.r.close()


def hstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=1)


def vstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=0)
