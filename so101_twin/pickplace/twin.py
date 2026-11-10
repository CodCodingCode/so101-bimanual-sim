"""Shared helpers: compile the scene, pose the arm from a real state, render the two cameras, read the real set."""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))

import build_scene  # noqa: E402
from joints import real_to_sim  # noqa: E402

CAMS = ("top", "side")
REAL_SET = "so101_pickplace_real"


def real_frames():
    from sets import Store

    return Store().frames(REAL_SET)


class Twin:
    def __init__(self, cams: dict | None = None, w: int = build_scene.W, h: int = build_scene.H):
        self.m = build_scene.load_model(cams)
        self.d = mujoco.MjData(self.m)
        self.w, self.h = w, h
        self.r = mujoco.Renderer(self.m, h, w)
        self.cam_id = {c: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMS}
        self.jnt_adr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                        for j in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")]
        # geoms that are dark on the real arm (everything on the arm except the white fixed jaw)
        self.dark_geoms = np.array([g for g in range(self.m.ngeom)
                                    if self.m.geom_group[g] == 2 and self.m.body_weldid[self.m.geom_bodyid[g]] != 0
                                    and self.m.geom_matid[g] != mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_MATERIAL, "jaw_white")])
        self.arm_geoms = np.array([g for g in range(self.m.ngeom)
                                   if self.m.geom_group[g] == 2 and self.m.body_weldid[self.m.geom_bodyid[g]] != 0])

    def set_state(self, state: np.ndarray, **kw) -> np.ndarray:
        q = real_to_sim(state, **kw)
        for adr, v in zip(self.jnt_adr, q):
            self.d.qpos[adr] = v
        mujoco.mj_forward(self.m, self.d)
        return q

    def set_camera(self, name: str, pos, lookat, roll=0.0, fovy=None):
        cid = self.cam_id[name]
        self.m.cam_pos[cid] = np.asarray(pos, float)
        self.m.cam_quat[cid] = build_scene.look_quat(pos, lookat, roll)
        if fovy is not None:
            self.m.cam_fovy[cid] = float(fovy)
        mujoco.mj_forward(self.m, self.d)

    def render(self, cam: str) -> np.ndarray:
        self.r.disable_segmentation_rendering()
        self.r.update_scene(self.d, camera=self.cam_id[cam])
        return self.r.render().copy()

    def silhouette(self, cam: str, dark_only: bool = True) -> np.ndarray:
        """bool [h, w]: pixels covered by the arm's (dark) geoms."""
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera=self.cam_id[cam])
        seg = self.r.render()[..., 0]
        self.r.disable_segmentation_rendering()
        return np.isin(seg, self.dark_geoms if dark_only else self.arm_geoms)
