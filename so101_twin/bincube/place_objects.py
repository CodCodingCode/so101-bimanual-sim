"""Estimate the bin position and the per-episode cube start positions on the table from the real side frames,
using the fitted side camera (run after fit_cameras.py --apply).

    .venv/bin/python so101_twin/bincube/place_objects.py      # prints BIN xy; writes out/cube_xy.json (episode -> (x, y))

Bin: the ray through a hand-picked pixel of the bin's top-rim centre (real side frame 0, lid open, arm away) is
intersected with the plane z = bin height. Cube: per episode, the yellow-pixel centroid (HSV threshold) in the
first frame whose cube state is 'table' (replay.episode_states) and whose yellow blob is big enough, intersected
with the plane z = cube half size. The side camera is a pinhole without distortion, so expect a few cm of error.
"""
from __future__ import annotations

import json
import pathlib
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
import replay  # noqa: E402
from twin import HERE, Sim, real_set  # noqa: E402

BIN_PIX = (205, 283)        # (u, v) of the bin's top-rim centre in real side frame 0
MIN_YELLOW = 600


def ray(sim: Sim, cam: str, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    cid = sim.cam_id[cam]
    pos = sim.d.cam_xpos[cid].copy()
    R = sim.d.cam_xmat[cid].reshape(3, 3)      # columns: camera x (right), y (up), z (backward)
    fovy = np.radians(sim.m.cam_fovy[cid])
    fy = (480 / 2) / np.tan(fovy / 2)
    d_cam = np.array([(u - 320) / fy, -(v - 240) / fy, -1.0])
    d = R @ d_cam
    return pos, d / np.linalg.norm(d)


def hit_plane(sim: Sim, cam: str, u: float, v: float, z: float) -> np.ndarray:
    o, d = ray(sim, cam, u, v)
    t = (z - o[2]) / d[2]
    return (o + t * d)[:2]


def yellow_centroid(rgb: np.ndarray) -> tuple[int, float, float]:
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    m = (hsv[..., 0] >= 28) & (hsv[..., 0] <= 50) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120)
    ys, xs = np.nonzero(m)
    return len(xs), (xs.mean() if len(xs) else -1), (ys.mean() if len(xs) else -1)


def main():
    fs = real_set()
    sim = Sim()
    mujoco.mj_forward(sim.m, sim.d)
    bxy = hit_plane(sim, "side", *BIN_PIX, 2 * build_scene.BIN["half"][2])
    print(f"BIN xy -> ({bxy[0]:+.3f}, {bxy[1]:+.3f})   (paste into build_scene.BIN['xy'])")
    lids, cubes = replay.all_states(fs)
    out = {}
    for e in np.unique(fs.labels["episode"]):
        idx = fs.where(episode=e)
        idx = idx[np.argsort(fs.labels["t"][idx])]
        for i in idx:
            if cubes[i] != "table":
                continue
            n, u, v = yellow_centroid(fs.obs(int(i))["images"]["side"])
            if n >= MIN_YELLOW:
                xy = hit_plane(sim, "side", u, v, build_scene.CUBE["half"])
                out[int(e)] = [round(float(xy[0]), 4), round(float(xy[1]), 4)]
                print(f"ep {e:2d} frame {i:3d} yellow n={n:5d} at ({u:5.1f},{v:5.1f}) -> x={xy[0]:+.3f} y={xy[1]:+.3f}")
                break
        else:
            print(f"ep {e:2d}: no usable cube frame, using build_scene.CUBE")
    (HERE / "out" / "cube_xy.json").write_text(json.dumps(out, indent=1))
    xs = np.array(list(out.values()))
    print(f"cube xy over {len(out)} episodes: mean ({xs[:, 0].mean():+.3f}, {xs[:, 1].mean():+.3f}), "
          f"x range [{xs[:, 0].min():+.3f}, {xs[:, 0].max():+.3f}] y range [{xs[:, 1].min():+.3f}, {xs[:, 1].max():+.3f}]")
    sim.close()


if __name__ == "__main__":
    main()
