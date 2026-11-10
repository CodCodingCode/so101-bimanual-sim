"""Render a few real frames' states in the sim and write side-by-side PNGs (real top | sim top | real side | sim side).

    .venv/bin/python so101_twin/legocup/check_joints.py [--idx 0 3 7 100 203 311]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from twin import HERE, Sim, hstack, real_set, vstack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="*", default=[0, 3, 7, 100, 203, 311])
    ap.add_argument("--out", default="joints_check.png")
    a = ap.parse_args()
    fs = real_set()
    sim = Sim()
    rows = []
    for i in a.idx:
        o = fs.obs(i)
        sim.set_joints(o["state"])
        rows.append(hstack(o["images"]["top"], sim.render("top"), o["images"]["side"], sim.render("side")))
        print(i, "ep", fs.labels["episode"][i], "t", fs.labels["t"][i], "state", np.round(o["state"], 1),
              "qpos", np.round(sim.d.qpos[:6], 2))
    out = HERE / "out" / a.out
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(vstack(*rows)).resize((1280, 240 * len(rows))).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
