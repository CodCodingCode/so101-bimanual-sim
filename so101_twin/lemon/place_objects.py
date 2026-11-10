"""Estimate object positions on the table from hand-picked pixel centroids in real frame 0, using the fitted cameras.

    .venv/bin/python so101_twin/lemon/place_objects.py       # prints (x, y) per object for build_scene.OBJECTS / BOWL

For each object the ray through its image centroid is intersected with the horizontal plane at the object's
centre height, for both cameras; the two estimates are averaged (the side/overhead camera gets weight 2 since
its geometry is better for the table plane). Pixel centroids were read off out/real_000.png by eye.
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
from twin import Sim  # noqa: E402

# (u, v) pixel centroids in real frame 0 (640x480), per camera; None = not visible / not used
PIX = {
    "apple": dict(top=(262, 392), side=(130, 150), z=0.038),
    "lemon": dict(top=(390, 355), side=(275, 240), z=0.023),
    "strawberry": dict(top=(495, 410), side=(120, 318), z=0.026),
    "peach": dict(top=(560, 338), side=(318, 420), z=0.030),
    "bowl": dict(top=(380, 300), side=(520, 240), z=0.028),   # bowl rim centre ~ half height
}
WEIGHT = {"top": 1.0, "side": 2.0}


def ray(sim: Sim, cam: str, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    cid = sim.cam_id[cam]
    pos = sim.d.cam_xpos[cid].copy()
    R = sim.d.cam_xmat[cid].reshape(3, 3)      # columns: camera x (right), y (up), z (backward)
    fovy = np.radians(sim.m.cam_fovy[cid])
    fy = (480 / 2) / np.tan(fovy / 2)
    d_cam = np.array([(u - 320) / fy, -(v - 240) / fy, -1.0])
    d = R @ d_cam
    return pos, d / np.linalg.norm(d)


def main():
    sim = Sim()
    mujoco.mj_forward(sim.m, sim.d)
    for name, p in PIX.items():
        ests, ws = [], []
        for cam in ("top", "side"):
            if p.get(cam) is None:
                continue
            o, d = ray(sim, cam, *p[cam])
            t = (p["z"] - o[2]) / d[2]
            hit = o + t * d
            ests.append(hit[:2])
            ws.append(WEIGHT[cam])
            print(f"{name:10s} {cam:4s} -> x={hit[0]:+.3f} y={hit[1]:+.3f}")
        xy = np.average(np.stack(ests), axis=0, weights=ws)
        print(f"{name:10s} MEAN -> x={xy[0]:+.3f} y={xy[1]:+.3f}")
    sim.close()


if __name__ == "__main__":
    main()
