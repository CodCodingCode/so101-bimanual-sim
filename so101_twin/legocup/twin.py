"""Shared helpers for the legocup twin: joint mapping real -> sim, scene loading, offscreen rendering, real-set access.

Joint convention (LeRobot v2.1 dataset recorded with the new `so101_follower` degrees calibration: zero = arm straight
up, rest pose ~[-10, -98, 98, 50, -3]; gripper linear 0..100):
    sim_rad[j] = deg2rad(SIGNS[j] * real_deg[j] + OFFSETS_DEG[j])  for the five revolute joints
    Jaw        = JAW_RANGE[0] + JAW_GAIN * gripper/100 * (JAW_RANGE[1] - JAW_RANGE[0])
See README.md for how SIGNS / OFFSETS_DEG were verified.
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
XML = HERE / "so101_legocup.xml"
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))

LEROBOT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SIM_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
SIGNS = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])                 # pan flipped (real +pan swings the arm to +x)
OFFSETS_DEG = np.array([0.0, -90.0, 90.0, 0.0, 180.0, 0.0])     # so101_follower zero (straight up) -> Menagerie zero (horizontal); fit_cameras.py --offsets refines
JAW_RANGE = (-0.174, 1.75)
JAW_GAIN = 1.0                   # Jaw = JAW_RANGE[0] + JAW_GAIN * gripper/100 * (JAW_RANGE[1] - JAW_RANGE[0]); fitted with the wrist camera
CAMS = ("top", "side")          # "side" is the wrist camera (child of Fixed_Jaw)
W, H = 640, 480
ROI_NPY = HERE / "out" / "mat_roi.npy"   # top-camera region of interest (the grey mat), see fit_cameras.build_roi
OFFSETS_JSON = HERE / "out" / "offsets_fit.json"
if OFFSETS_JSON.exists():                 # fitted joint offsets (fit_cameras.py --offsets) override the defaults above
    import json as _json

    OFFSETS_DEG[:] = _json.loads(OFFSETS_JSON.read_text())["offsets_deg"]
SIDE_FIT_JSON = HERE / "out" / "camera_fit_side.json"
if SIDE_FIT_JSON.exists():                # the wrist-camera fit also fits the jaw gain
    import json as _json

    JAW_GAIN = float(_json.loads(SIDE_FIT_JSON.read_text())["params"].get("jaw_gain", 1.0))


def real_to_sim(q: np.ndarray) -> np.ndarray:
    """(6,) real [deg x5, gripper 0..100] -> (6,) sim qpos [rad] in SIM_ORDER."""
    q = np.asarray(q, dtype=float)
    rad = np.deg2rad(SIGNS[:5] * q[:5] + OFFSETS_DEG[:5])
    jaw = JAW_RANGE[0] + JAW_GAIN * np.clip(q[5], 0, 100) / 100.0 * (JAW_RANGE[1] - JAW_RANGE[0])
    return np.concatenate([rad, [jaw]])


def real_set():
    from sets import Store

    return Store().frames("so101_legocup_real")


class Sim:
    def __init__(self, xml: pathlib.Path = XML, w: int = W, h: int = H):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.w, self.h = w, h
        self.cam_id = {c: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMS}
        self.arm_geoms = self._arm_geoms()
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
        """pos / angles in the camera's parent frame (world for "top", Fixed_Jaw for the wrist camera "side")."""
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
