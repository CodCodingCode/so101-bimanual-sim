"""Shared helpers for the drawer twin: joint mapping real -> sim, scene loading, offscreen rendering, real-set access.

Joint convention (LeRobot v3.0 so_follower calibration, degrees relative to the "middle" pose; gripper 0..100):
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
XML = HERE / "so101_drawer.xml"
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))

LEROBOT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SIM_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
SIGNS = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
OFFSETS_DEG = np.array([-1.94, -94.13, 88.55, -22.47, 180.0, 0.0])   # fit_cameras.py --offsets for 0..3 (first guess [0,-90,+90,0]); roll 180 chosen visually (wrist view)
JAW_RANGE = (-0.174, 1.75)   # Jaw = -0.174 + gripper/100 * 1.924
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

    return Store().frames("so101_drawer_real")


class Sim:
    def __init__(self, xml: pathlib.Path = XML, w: int = W, h: int = H):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.w, self.h = w, h
        self.cam_id = {c: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMS}
        self.arm_geoms = self._arm_geoms()
        jaw_bodies = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("Fixed_Jaw", "Moving_Jaw")}
        jaw_meshes = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_MESH, n) for n in ("Fixed_Jaw", "Moving_Jaw")}
        is_jaw = np.array([self.m.geom_bodyid[g] in jaw_bodies and self.m.geom_dataid[g] in jaw_meshes
                           for g in self.arm_geoms])
        is_motor = np.array([self.m.geom_dataid[g] >= 0 and mujoco.mj_id2name(
            self.m, mujoco.mjtObj.mjOBJ_MESH, int(self.m.geom_dataid[g])).endswith("_Motor") for g in self.arm_geoms])
        # "nomotor": all printed parts (green/orange/white jaws), black motors are holes; "body": printed parts
        # without the white jaws; "jaws": the two white jaw meshes only
        self.mask_geoms = {"all": self.arm_geoms, "nomotor": self.arm_geoms[~is_motor],
                           "body": self.arm_geoms[~is_jaw & ~is_motor], "jaws": self.arm_geoms[is_jaw]}
        mujoco.mj_forward(self.m, self.d)

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
        """Override a camera at runtime. cam_pos / cam_quat are in the camera's parent body frame, so for the
        wrist camera ("top", inside Fixed_Jaw) the parameters are body-frame values, like in build_scene.CAMS."""
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
        self.r.update_scene(self.d, camera=cam)
        return self.r.render().copy()

    def arm_mask(self, cam: str, which: str = "all") -> np.ndarray:
        """bool HxW: pixels covered by the arm's visual geoms (segmentation render); which = all / body / jaws."""
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera=cam)
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        return np.isin(seg[..., 0], self.mask_geoms[which])

    def close(self):
        self.r.close()


def hstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=1)


def vstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=0)
