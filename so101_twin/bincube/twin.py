"""Shared helpers for the bincube twin: joint mapping real -> sim, scene loading, offscreen rendering, real-set access.

Joint convention (LeRobot v2.1 recorded with the new so101_follower calibration, degrees; gripper linear 0..100):
    sim_rad[j] = deg2rad(SIGNS[j] * real_deg[j] + OFFSETS_DEG[j])  for the five revolute joints
    Jaw        = JAW_RANGE[0] + gripper/100 * (JAW_RANGE[1] - JAW_RANGE[0])
See README.md for how SIGNS / OFFSETS_DEG were verified.
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
XML = HERE / "so101_bincube.xml"
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))

LEROBOT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SIM_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
SIGNS = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])                       # so101_follower calibration (as pickplace)
OFFSETS_DEG = np.array([0.0, -90.0, 90.0, 0.0, 105.0, 0.0])   # derived; roll offset from check_roll.py (wrist-view cube projection)
JAW_RANGE = (-0.174, 1.75)
CAMS = ("top", "side")
W, H = 640, 480


def real_to_sim(q: np.ndarray) -> np.ndarray:
    """(6,) real [deg x5, gripper 0..100] -> (6,) sim qpos [rad] in SIM_ORDER."""
    q = np.asarray(q, dtype=float)
    rad = np.deg2rad(SIGNS[:5] * q[:5] + OFFSETS_DEG[:5])
    jaw = JAW_RANGE[0] + np.clip(q[5], 0, 100) / 100.0 * (JAW_RANGE[1] - JAW_RANGE[0])
    return np.concatenate([rad, [jaw]])


def real_set():
    from sets import Store

    return Store().frames("so101_bincube_real")


class Sim:
    def __init__(self, xml: pathlib.Path = XML, w: int = W, h: int = H):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.w, self.h = w, h
        self.cam_id = {c: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMS}
        self.arm_geoms = self._arm_geoms()
        self.arm_geoms_nobase = self._arm_geoms(skip_base=True)
        mujoco.mj_forward(self.m, self.d)

    def _arm_geoms(self, skip_base: bool = False) -> np.ndarray:
        """Visual geoms of the arm; skip_base drops the static Base body's geoms (they are background-subtracted
        from the real side mask, so they must not count in the sim mask either)."""
        base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "Base")
        ids = []
        for g in range(self.m.ngeom):
            b0 = b = self.m.geom_bodyid[g]
            while b != 0 and b != base:
                b = self.m.body_parentid[b]
            if b == base and self.m.geom_group[g] == 2 and not (skip_base and b0 == base):   # visual meshes only
                ids.append(g)
        return np.array(ids)

    def set_joints(self, q_real: np.ndarray) -> None:
        self.d.qpos[:6] = real_to_sim(q_real)
        self.d.ctrl[:6] = self.d.qpos[:6]
        mujoco.mj_forward(self.m, self.d)

    def set_camera(self, name: str, pos, yaw: float, pitch: float, roll: float, fovy: float) -> None:
        """pos / yaw / pitch / roll are in the frame of the camera's parent body (world for "side", Fixed_Jaw for
        the wrist camera "top"), i.e. exactly the numbers in build_scene.CAMS."""
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

    def arm_mask(self, cam: str, skip_base: bool = False) -> np.ndarray:
        """bool HxW: pixels covered by the arm's visual geoms (segmentation render)."""
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera=cam)
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        return np.isin(seg[..., 0], self.arm_geoms_nobase if skip_base else self.arm_geoms)

    def set_mocap(self, body: str, pos, quat=None) -> None:
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body)
        mid = self.m.body_mocapid[bid]
        self.d.mocap_pos[mid] = pos
        if quat is not None:
            self.d.mocap_quat[mid] = quat

    def close(self):
        self.r.close()


def hstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=1)


def vstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=0)
