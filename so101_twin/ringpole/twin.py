"""Shared helpers for the ringpole twin: joint mapping real -> sim, scene loading, offscreen rendering, real-set access.

Joint convention (LeRobot v3.0 `so101_follower` calibration, degrees relative to the calibration "middle" pose;
gripper 0..100 percent open):
    sim_rad[j] = deg2rad(SIGNS[j] * real_deg[j] + OFFSETS_DEG[j])  for the five revolute joints
    Jaw        = JAW_RANGE[0] + gripper/100 * (JAW_RANGE[1] - JAW_RANGE[0])
Started from the pickplace twin's fitted v3.0 mapping; see README.md for how SIGNS / OFFSETS_DEG were verified.

Cameras: "top" is a static world-frame camera; "side" is attached to the `pan_frame` body (a child of
Rotation_Pitch, x = arm forward, z = up at the pan axis), so its parameters are relative to that frame and the
camera rotates with the pan joint like the real shoulder camera.
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
XML = HERE / "so101_ringpole.xml"
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))

LEROBOT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SIM_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
SIGNS = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
OFFSETS_DEG = np.array([0.00, -86.60, 84.07, 8.77, 153.60, 0.00])   # fit_cameras.py --offsets (lift/elbow/wrist refined from the pickplace v3.0 fit)
JAW_RANGE = (-0.174, 1.75)
CAMS = ("top", "side")
W, H = 640, 480
REAL_SET = "so101_ringpole_real"


def real_to_sim(q: np.ndarray) -> np.ndarray:
    """(6,) real [deg x5, gripper 0..100] -> (6,) sim qpos [rad] in SIM_ORDER."""
    q = np.asarray(q, dtype=float)
    rad = np.deg2rad(SIGNS[:5] * q[:5] + OFFSETS_DEG[:5])
    jaw = JAW_RANGE[0] + np.clip(q[5], 0, 100) / 100.0 * (JAW_RANGE[1] - JAW_RANGE[0])
    return np.concatenate([rad, [jaw]])


def real_set():
    from sets import Store

    return Store().frames(REAL_SET)


class Sim:
    def __init__(self, xml: pathlib.Path = XML, w: int = W, h: int = H):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.w, self.h = w, h
        self.cam_id = {c: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMS}
        self.arm_geoms = self._arm_geoms(white_only=True)
        self.pole_geoms = np.array([mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "pole")])
        self.ring_geoms = np.array([g for g in range(self.m.ngeom)
                                    if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("ring_")])
        self.tcp = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        mujoco.mj_forward(self.m, self.d)

    def _arm_geoms(self, white_only: bool) -> np.ndarray:
        base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "Base")
        white = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_MATERIAL, "arm_white")
        ids = []
        for g in range(self.m.ngeom):
            b = self.m.geom_bodyid[g]
            while b != 0 and b != base:
                b = self.m.body_parentid[b]
            if b == base and self.m.geom_group[g] == 2:   # visual meshes only
                if not white_only or self.m.geom_matid[g] == white:
                    ids.append(g)
        return np.array(ids)

    def set_joints(self, q_real: np.ndarray) -> None:
        self.d.qpos[:6] = real_to_sim(q_real)
        self.d.ctrl[:6] = self.d.qpos[:6]
        mujoco.mj_forward(self.m, self.d)

    def set_mocap(self, body: str, pos, quat=None) -> None:
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body)
        mid = self.m.body_mocapid[bid]
        self.d.mocap_pos[mid] = pos
        if quat is not None:
            self.d.mocap_quat[mid] = quat

    def set_camera(self, name: str, pos, yaw: float, pitch: float, roll: float, fovy: float) -> None:
        """Camera pose in its parent body's frame (world for 'top', pan_frame for 'side')."""
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
        mujoco.mj_forward(self.m, self.d)

    def render(self, cam: str) -> np.ndarray:
        self.r.update_scene(self.d, camera=cam)
        return self.r.render().copy()

    def seg(self, cam: str) -> np.ndarray:
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera=cam)
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        return seg[..., 0]

    def arm_mask(self, cam: str, with_pole: bool = True) -> np.ndarray:
        """bool HxW: pixels covered by the arm's white visual geoms (+ the white pole): the 'white' silhouette."""
        ids = np.concatenate([self.arm_geoms, self.pole_geoms]) if with_pole else self.arm_geoms
        return np.isin(self.seg(cam), ids)

    def close(self):
        self.r.close()


def hstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=1)


def vstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=0)
