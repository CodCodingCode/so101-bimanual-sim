"""Shared helpers for the sort twin: joint mapping real -> sim, scene loading, offscreen rendering, real-set access.

Joint convention (LeRobot v3.0 `so101_follower` degrees mode, zero = calibration "middle" pose; gripper 0..100):
    sim_rad[j] = deg2rad(SIGNS[j] * real_deg[j] + OFFSETS_DEG[j])  for the five revolute joints
    Jaw        = JAW_RANGE[0] + gripper/100 * (JAW_RANGE[1] - JAW_RANGE[0])
Starting point was ../pickplace (same dataset version): SIGNS [-1,+1,+1,+1,+1], OFFSETS ~[0,-90,+90,0,180].
See README.md for how they were verified / refined here.
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
XML = HERE / "so101_sort.xml"
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))

REAL_SET = "so101_sort_real"
LEROBOT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SIM_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
SIGNS = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
OFFSETS_DEG = np.array([0.0, -90.0, 90.0, 0.0, 180.0, 0.0])   # fit_cameras.py --offsets refines 1..4 (deg, before deg2rad)
JAW_RANGE = (-0.174, 1.75)
CAMS = ("top", "side")                  # the real set's camera names (replay output)
SIM_CAMS = ("top", "top_b", "side")     # cameras in the xml
W, H = 640, 480


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
        self.cam_id = {c: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in SIM_CAMS}
        self.arm_geoms = self._arm_geoms()
        self.jaw_geoms = self._arm_geoms(bodies=("Fixed_Jaw", "Moving_Jaw"))
        self.blue_geoms = self._arm_geoms(bodies=("Wrist_Pitch_Roll",))   # the light-blue wrist + jaws
        self.all_arm_geoms = self._arm_geoms(include_motors=True)          # everything of the robot (wrist-cam fit)
        mujoco.mj_forward(self.m, self.d)

    def _arm_geoms(self, bodies=("Base",), include_motors=False) -> np.ndarray:
        roots = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies}
        ids = []
        for g in range(self.m.ngeom):
            b = self.m.geom_bodyid[g]
            while b != 0 and b not in roots:
                b = self.m.body_parentid[b]
            if b in roots and self.m.geom_group[g] == 2:   # visual meshes only (motor geoms are group 2 too)
                if include_motors or "Motor" not in mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_MESH, self.m.geom_dataid[g]):
                    ids.append(g)
        return np.array(ids)

    def set_joints(self, q_real: np.ndarray) -> None:
        self.d.qpos[:6] = real_to_sim(q_real)
        self.d.ctrl[:6] = self.d.qpos[:6]
        mujoco.mj_forward(self.m, self.d)

    def set_camera(self, name: str, pos, yaw: float, pitch: float, roll: float, fovy: float) -> None:
        """Static cams: world frame. 'side': Fixed_Jaw body frame (cam_pos/cam_quat are parent-body relative)."""
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

    def mask(self, cam: str, geoms: np.ndarray) -> np.ndarray:
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera=cam)
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        return np.isin(seg[..., 0], geoms)

    def arm_mask(self, cam: str) -> np.ndarray:
        """bool HxW: pixels covered by the arm's non-motor visual geoms (segmentation render)."""
        return self.mask(cam, self.arm_geoms)

    def jaw_mask(self, cam: str) -> np.ndarray:
        return self.mask(cam, self.jaw_geoms)

    def robot_mask(self, cam: str) -> np.ndarray:
        """Every robot geom incl. motors: from the wrist camera anything of the robot in view is a penalty."""
        return self.mask(cam, self.all_arm_geoms)

    def blue_mask(self, cam: str) -> np.ndarray:
        return self.mask(cam, self.blue_geoms)

    def close(self):
        self.r.close()


def hstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=1)


def vstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=0)
