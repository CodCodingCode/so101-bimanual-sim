#!/usr/bin/env python3
"""Measure how the REAL rig looks vs the sim, per region (arm plastic, table, cube, box, background), from a saved real
overhead + wrist frame with known arm angles and cube spot. Renders the sim from the calibrated cameras at the same
pose, labels pixels by geom (segmentation, same lens distortion), compares mean colours.

    python squeeze/match_real_colors.py <dir with overhead_full.jpg, left_wrist_full.jpg, right_wrist_full.jpg, units.npy> cx cy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]


def render_seg(sim, cam):
    import cv2
    import mujoco
    spec = sim.camera_specs[cam]
    width, height, K, mapping = sim._camera_plans.get(cam) or spec.render_plan()
    sim.model.vis.global_.offwidth = max(width, sim.model.vis.global_.offwidth)
    sim.model.vis.global_.offheight = max(height, sim.model.vis.global_.offheight)
    r = mujoco.Renderer(sim.model, height=height, width=width)
    r.update_scene(sim.data, camera=cam)
    for view in r.scene.camera:
        near = view.frustum_near
        view.frustum_width = near * width / (2 * K[0, 0])
        view.frustum_center = near * (width / 2 - K[0, 2]) / K[0, 0]
        view.frustum_top = near * K[1, 2] / K[1, 1]
        view.frustum_bottom = -near * (height - K[1, 2]) / K[1, 1]
    rgb = r.render().copy()
    r.enable_segmentation_rendering()
    seg = r.render()[..., 0].copy()          # geom id (objtype in [...,1])
    r.close()
    if mapping is not None:
        rgb = cv2.remap(rgb, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR)
        seg = cv2.remap(seg.astype(np.float32), mapping[..., 0], mapping[..., 1], cv2.INTER_NEAREST).astype(np.int32)
    return rgb, seg


def main():
    import mujoco
    from PIL import Image
    import distill as Dl
    import clamp_teacher as C
    import env as E
    d_in = Path(sys.argv[1])
    cx, cy = float(sys.argv[2]), float(sys.argv[3])
    env = Dl.make_env(1)
    env.reset(seed=1)
    sim, m, d = env.base.sim, env.model, env.data
    u = np.load(d_in / "units.npy")
    glo, ghi = m.actuator_ctrlrange[sim.robot["left"].gripper_actuator]
    for i, s in enumerate(("left", "right")):
        arm = sim.robot[s]
        d.qpos[arm.qpos_ids] = np.deg2rad(u[i * 6:i * 6 + 5])
        d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = glo + np.clip(u[i * 6 + 5], 0, 100) / 100 * (ghi - glo)
    cq, bq = m.jnt_qposadr[m.joint("cube_joint").id], m.jnt_qposadr[m.joint("can_joint").id]
    d.qpos[cq:cq + 7] = [cx, cy, C.CUBE_HALF, 1, 0, 0, 0]
    d.qpos[bq:bq + 7] = [cx, cy - 0.105, 0.015, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    body = lambda g: m.body(m.geom_bodyid[g]).name if g >= 0 else ""
    cls = {}
    for g in range(m.ngeom):
        n, b = m.geom(g).name, body(g)
        if b.startswith(("left_", "right_")):
            # printed plastic vs the black servos: by the material colour the sim uses
            rgb = m.mat_rgba[m.geom_matid[g], :3] if m.geom_matid[g] >= 0 else m.geom_rgba[g, :3]
            cls[g] = "arm_plastic" if rgb.mean() > 0.35 else "arm_dark"
        elif n == "table":
            cls[g] = "table"
        elif n == "cube_geom":
            cls[g] = "cube"
        elif b == "can":
            cls[g] = "box"
        elif n == "floor" or n.startswith("room_wall"):
            cls[g] = "background"
    out = {}
    for cam, fname in (("overhead", "overhead_full.jpg"), ("left_wrist", "left_wrist_full.jpg"), ("right_wrist", "right_wrist_full.jpg")):
        real = np.asarray(Image.open(d_in / fname).convert("RGB")).astype(float)
        rgb, seg = render_seg(sim, cam)
        if rgb.shape[:2] != real.shape[:2]:
            rgb = np.asarray(Image.fromarray(rgb).resize(real.shape[1::-1]))
            seg = np.asarray(Image.fromarray(seg.astype(np.int32)).resize(real.shape[1::-1], Image.NEAREST))
        import scipy.ndimage as ndi
        res = {}
        for name in ("arm_plastic", "arm_dark", "table", "cube", "box", "background"):
            ids = [g for g, c in cls.items() if c == name]
            mask = np.isin(seg, ids)
            mask = ndi.binary_erosion(mask, iterations=6)          # stay off edges (calibration is a few px off)
            if mask.sum() < 200:
                continue
            res[name] = {"px": int(mask.sum()), "real": (real[mask].mean(0) / 255).round(3).tolist(),
                         "sim": (rgb[mask].astype(float).mean(0) / 255).round(3).tolist()}
        out[cam] = res
        Image.fromarray(np.concatenate([real.astype(np.uint8), rgb], 1)).save(HERE / "runs" / f"color_match_{cam}.jpg")
    print(json.dumps(out, indent=1))
    (HERE / "runs" / "color_match.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
