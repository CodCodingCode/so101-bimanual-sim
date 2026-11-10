"""Arm silhouettes of the CURRENT (builder) twin of a scene, per real frame and camera: the prompts for segment.py.

    .venv/bin/python so101_twin/common/twin_prompts.py lemon        # one scene per process (module names collide)
    .venv/bin/python so101_twin/common/twin_prompts.py all          # runs every scene in its own subprocess

Uses each scene's own code in so101_twin/<scene>/ read-only (its Sim / Twin class, fitted cameras, joint map and the
per-episode / per-session camera switching of its replay). The silhouette is every geom of the arm subtree (body
'Base' of the Menagerie arm), not the scene's colour-specific mask. Output: so101_twin/masks/prompts/<scene>_<cam>.npy,
bool [n, 480, 640] packed with np.packbits(axis=-1) -> uint8 [n, 480, 80].
"""
from __future__ import annotations

import importlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

TWIN = pathlib.Path(__file__).resolve().parent.parent
ROOT = TWIN.parent
OUT = TWIN / "masks" / "prompts"
SCENES = ("lemon", "pickplace", "legocup", "drawer", "ringpole", "bincube", "sort", "stack")


def subtree_geoms(m, root_name: str = "Base") -> np.ndarray:
    import mujoco

    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_name)
    out = []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        while b > 0 and b != root:
            b = m.body_parentid[b]
        if b == root and m.geom_group[g] <= 2:
            out.append(g)
    return np.array(out)


def seg_mask(sim, cam, geoms) -> np.ndarray:
    r = sim.r
    r.enable_segmentation_rendering()
    r.update_scene(sim.d, camera=cam)
    seg = r.render()
    r.disable_segmentation_rendering()
    return np.isin(seg[..., 0], geoms) & (seg[..., 1] == 5)   # 5 = mjOBJ_GEOM


def run_scene(scene: str) -> None:
    sdir = TWIN / scene
    sys.path.insert(0, str(sdir))
    sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
    from sets import Store

    twin = importlib.import_module("twin")
    fs = Store().frames(f"so101_{scene}_real")
    if scene == "pickplace":
        sim = twin.Twin()
        pose = sim.set_state
    else:
        sim = twin.Sim()
        pose = sim.set_joints
    geoms = subtree_geoms(sim.m)
    cam_for = {c: (lambda i, c=c: c) for c in ("top", "side")}
    pre = lambda i: None   # noqa: E731
    if scene == "sort":
        bs = importlib.import_module("build_scene")
        cam_for["top"] = lambda i: bs.top_cam_for_episode(int(fs.labels["episode"][i]))
    if scene == "stack":
        ep_cams = json.loads((sdir / "out" / "episode_cams.json").read_text())
        ep = np.asarray(fs.labels["episode"])

        def pre(i):
            q = ep_cams["episodes"][str(int(ep[i]))]["params"]
            sim.set_camera(ep_cams["cam"], (q["x"], q["y"], q["z"]), q["yaw"], q["pitch"], q["roll"], q["fovy"])
    masks = {c: np.zeros((fs.n, 480, 80), np.uint8) for c in ("top", "side")}
    t0 = time.time()
    for i in range(fs.n):
        pre(i)
        pose(fs.state[i])
        for c in ("top", "side"):
            masks[c][i] = np.packbits(seg_mask(sim, cam_for[c](i), geoms), axis=-1)
    OUT.mkdir(parents=True, exist_ok=True)
    for c in masks:
        np.save(OUT / f"{scene}_{c}.npy", masks[c])
        area = np.unpackbits(masks[c], axis=-1).reshape(fs.n, -1).mean(1)
        print(f"{scene}/{c}: n={fs.n} arm area mean {area.mean():.3f}, empty {np.mean(area == 0):.2f}")
    print(f"{scene}: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "all":
        for s in SCENES:
            subprocess.run([sys.executable, __file__, s], check=True)
    else:
        run_scene(arg)
