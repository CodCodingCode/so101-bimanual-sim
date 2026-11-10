"""Estimate object positions on the table from hand-picked pixel centroids in the real overhead frames, using the
fitted "side" camera.

    .venv/bin/python so101_twin/drawer/place_objects.py       # prints (x, y) per object for build_scene.OBJECTS / DRAWER_UNIT

For each object the ray through its image centroid is intersected with the horizontal plane at the given height.
Only the static overhead camera is used (the "top" camera is the wrist camera). Pixel centroids were read off
out/side_007_full.png (episode 1, last frame) by eye; the blocks move between episodes, this is one layout.
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
    "purple": dict(side=(280, 170), z=0.011),
    "cyan": dict(side=(345, 170), z=0.013),
    "drawer_top_front_centre": dict(side=(245, 60), z=0.085),   # front edge of the unit's top plate
    "drawer_top_left_edge": dict(side=(190, 30), z=0.085),
    "drawer_top_right_edge": dict(side=(300, 30), z=0.085),
    "base_centre": dict(side=(195, 420), z=0.03),               # sanity check: should come out near (0, 0)
}
WEIGHT = {"side": 1.0}


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
        for cam in ("side",):
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
