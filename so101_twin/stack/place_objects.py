"""Estimate object positions on the table from hand-picked pixel centroids in real frame 0, using the fitted cameras.

    .venv/bin/python so101_twin/stack/place_objects.py       # prints (x, y) per object for build_scene.OBJECTS / DISC

For each object the ray through its image centroid is intersected with the horizontal plane at the object's
centre height, for both cameras; the two estimates are averaged (the overhead "top" camera gets weight 2 since
its geometry is better for the table plane; the low "side" camera mostly fixes the depth ordering).
Pixel centroids were read off out/real_000_top.png / out/real_000_side.png by eye (episode 0, frame t=42).
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
    "disc": dict(top=(318, 240), side=(190, 290), z=0.002),
    "cube": dict(top=(320, 285), side=(250, 245), z=build_scene.OBJECTS["cube"]["size"][2]),
    "green": dict(top=(262, 150), side=(415, 315), z=build_scene.OBJECTS["green"]["size"][2]),
}
WEIGHT = {"top": 2.0, "side": 1.0}


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
