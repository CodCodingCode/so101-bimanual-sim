"""Per-episode cube layout (and the cardboard box position) on the table from the real frames, using the fitted top cameras.

    .venv/bin/python so101_twin/sort/place_objects.py       # -> out/cube_layouts.json, out/place_check.png

Cubes: purple HSV mask on the first frame of each episode (arm at rest, cubes visible), connected components,
each centroid's camera ray intersected with the horizontal plane at the cube centre height (z = CUBE_HALF),
using the session's fitted static camera. Up to N_CUBES per episode (largest components first); episodes with no
detection fall back to build_scene.CUBES_DEFAULT. The target cube = the one nearest the gripper (site tcp) at the
episode's first 'grasp' frame. Box: ray through a hand-picked pixel of the box's bottom-front edge per session.
"""
from __future__ import annotations

import json
import pathlib
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
from twin import HERE, Sim, real_set  # noqa: E402

OUT = HERE / "out"
# (u, v) of the box's bottom-centre (front face bottom edge midpoint) in a frame of each session, at z = 0
BOX_PIX = {"top": (16, (608, 262)), "top_b": (296, (612, 180))}
MIN_AREA = 350


def purple_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 150) & (h <= 185) & (s > 120) & (v >= 40) & (v <= 130)


def ray(sim: Sim, cam: str, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    cid = sim.cam_id[cam]
    pos = sim.d.cam_xpos[cid].copy()
    R = sim.d.cam_xmat[cid].reshape(3, 3)      # columns: camera x (right), y (up), z (backward)
    fovy = np.radians(sim.m.cam_fovy[cid])
    fy = (480 / 2) / np.tan(fovy / 2)
    d_cam = np.array([(u - 320) / fy, -(v - 240) / fy, -1.0])
    d = R @ d_cam
    return pos, d / np.linalg.norm(d)


def pix_to_plane(sim: Sim, cam: str, u: float, v: float, z: float) -> np.ndarray | None:
    o, d = ray(sim, cam, u, v)
    if d[2] >= -1e-6:
        return None
    t = (z - o[2]) / d[2]
    return (o + t * d)[:2]


def components(mask: np.ndarray, min_area: int = MIN_AREA) -> list[tuple[float, float, int]]:
    lab, n = ndimage.label(ndimage.binary_opening(mask, iterations=2))
    out = []
    for k in range(1, n + 1):
        m = lab == k
        a = int(m.sum())
        if a >= min_area:
            vs, us = np.nonzero(m)
            out.append((float(us.mean()), float(vs.mean()), a))
    return sorted(out, key=lambda t: -t[2])


def main():
    fs = real_set()
    sim = Sim()
    mujoco.mj_forward(sim.m, sim.d)
    eps = np.asarray(fs.labels["episode"])
    phases = np.asarray(fs.labels["phase"])
    tcp = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    layouts = {}
    tiles = []
    for ep in sorted(set(int(e) for e in eps)):
        idx = np.nonzero(eps == ep)[0]
        i0 = int(idx[0])
        cam = build_scene.top_cam_for_episode(ep)
        rgb = fs.obs(i0)["images"]["top"]
        comps = components(purple_mask(rgb))
        cubes = []
        for u, v, a in comps:
            xy = pix_to_plane(sim, cam, u, v, build_scene.CUBE_HALF)
            if xy is None or abs(xy[0]) > 0.45 or xy[1] < 0.05 or xy[1] > 0.7:
                continue
            if all(np.linalg.norm(xy - np.array(c)) > 0.03 for c in cubes):
                cubes.append([round(float(xy[0]), 4), round(float(xy[1]), 4)])
            if len(cubes) == build_scene.N_CUBES:
                break
        fallback = not cubes
        if fallback:
            cubes = [list(c) for c in build_scene.CUBES_DEFAULT]
        # target cube: nearest to the gripper at the first grasp frame of the episode
        g = [int(i) for i in idx if phases[i] in ("grasp", "transfer_open")]
        target = 0
        if g:
            sim.set_joints(fs.state[g[0]])
            p = sim.d.site_xpos[tcp][:2]
            target = int(np.argmin([np.linalg.norm(p - np.array(c)) for c in cubes]))
            dist = float(np.linalg.norm(p - np.array(cubes[target])))
        else:
            dist = float("nan")
        layouts[str(ep)] = dict(cam=cam, frame=i0, cubes=cubes, target=target, fallback=fallback,
                                tcp_dist_at_grasp=round(dist, 3))
        print(f"ep {ep:3d} frame {i0:3d} {cam:5s} cubes {cubes} target {target} tcp-dist {dist:.3f}"
              f"{' FALLBACK' if fallback else ''}")
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        for u, v, a in comps:
            dr.ellipse([u - 6, v - 6, u + 6, v + 6], outline=(255, 0, 0), width=2)
        dr.text((4, 4), f"ep{ep} n={len(cubes)} tgt={target}", fill=(255, 255, 0))
        tiles.append(np.asarray(im.resize((320, 240))))
    rows = [np.concatenate(tiles[r * 5:(r + 1) * 5], 1) for r in range(len(tiles) // 5)]
    Image.fromarray(np.concatenate(rows, 0)).save(OUT / "place_check.png")
    box = {}
    for cam, (i, (u, v)) in BOX_PIX.items():
        xy = pix_to_plane(sim, cam, u, v, 0.0)
        box[cam] = [round(float(xy[0]), 4), round(float(xy[1]), 4)] if xy is not None else None
        print(f"box bottom-centre via {cam}: {box[cam]}  (box centre = that + half depth along the view)")
    (OUT / "cube_layouts.json").write_text(json.dumps(dict(layouts=layouts, box_front_bottom=box), indent=1))
    sim.close()
    print("wrote", OUT / "cube_layouts.json")


if __name__ == "__main__":
    main()
