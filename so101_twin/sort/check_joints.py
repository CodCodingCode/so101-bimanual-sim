"""Render a few real frames' states in the sim and write side-by-side PNGs (real top | sim top | real side | sim side).

    .venv/bin/python so101_twin/sort/check_joints.py [--idx 0 3 7 100 203 311] [--roll-offset 180]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
import twin  # noqa: E402
from twin import HERE, Sim, hstack, real_set, vstack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="*", default=[0, 3, 7, 100, 203, 311])
    ap.add_argument("--out", default="joints_check.png")
    ap.add_argument("--roll-offset", type=float, default=None)
    a = ap.parse_args()
    if a.roll_offset is not None:
        twin.OFFSETS_DEG[4] = a.roll_offset
    fs = real_set()
    sim = Sim()
    rows = []
    for i in a.idx:
        o = fs.obs(i)
        sim.set_joints(o["state"])
        top_cam = build_scene.top_cam_for_episode(fs.labels["episode"][i])
        rows.append(hstack(o["images"]["top"], sim.render(top_cam), o["images"]["side"], sim.render("side")))
        print(i, "ep", fs.labels["episode"][i], "t", fs.labels["t"][i], "state", np.round(o["state"], 1),
              "qpos", np.round(sim.d.qpos[:6], 2))
    out = HERE / "out" / a.out
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(vstack(*rows)).resize((1280, 240 * len(rows))).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
