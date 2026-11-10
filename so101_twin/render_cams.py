"""Render the three policy cameras (top, left wrist, right wrist) of the bimanual SO-100 workcell.

    python so101_twin/render_cams.py                 # home keyframe -> out/cams_home.png (3 views side by side)
    python so101_twin/render_cams.py --qpos "..."    # 12 joint values (left 6, right 6) instead of the keyframe
    python so101_twin/render_cams.py --free           # also a free 3/4 view for sanity

Writes PNGs to so101_twin/out/. Offscreen rendering on macOS works from a plain python process.
"""
from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
XML = HERE / "bimanual_so100_workcell.xml"
CAMS = ("cam_top", "left_cam_wrist", "right_cam_wrist")
W, H = 640, 480


def render(m: mujoco.MjModel, d: mujoco.MjData, cam: str | int, w=W, h=H) -> np.ndarray:
    r = mujoco.Renderer(m, h, w)
    r.update_scene(d, camera=cam)
    img = r.render().copy()
    r.close()
    return img


def free_view(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.05, 0.8)
    cam.distance, cam.azimuth, cam.elevation = 1.6, 150, -25
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera=cam)
    img = r.render().copy()
    r.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", type=str, default=None, help="12 joint values, left then right")
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--tag", default="home")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if a.qpos:
        q = np.array([float(x) for x in a.qpos.split()])
        assert q.size == 12, q.size
        d.qpos[:12] = q
        d.ctrl[:12] = q
    mujoco.mj_forward(m, d)

    tiles = [render(m, d, c) for c in CAMS]
    grid = np.concatenate(tiles, axis=1)
    out = HERE / "out" / f"cams_{a.tag}.png"
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(grid).save(out)
    print("wrote", out)
    if a.free:
        fo = HERE / "out" / f"free_{a.tag}.png"
        Image.fromarray(free_view(m, d)).save(fo)
        print("wrote", fo)
    for side in ("left", "right"):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        print(f"{side}_tcp world pos:", np.round(d.site_xpos[sid], 3))


if __name__ == "__main__":
    main()
