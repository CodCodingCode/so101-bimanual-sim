"""Contact sheet of random real/twin pairs: rows = (real top | twin top | real side | twin side).

    .venv/bin/python so101_twin/drawer/make_sheet.py [--n 8] [--seed 0]   # -> out/pairs_sheet.jpg
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from twin import HERE, real_set  # noqa: E402  (adds neuroscope/molmoact2 to sys.path)
from sets import FrameSet  # noqa: E402

TWIN = HERE.parent.parent / "neuroscope" / "molmoact2" / "store" / "sets" / "so101_drawer_twin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pairs_sheet.jpg")
    a = ap.parse_args()
    real, tw = real_set(), FrameSet(TWIN)
    idx = np.sort(np.random.default_rng(a.seed).choice(tw.n, a.n, replace=False))
    rows = []
    for i in idx:
        j = int(tw.labels["pair"][i])
        r, t = real.obs(j), tw.obs(i)
        assert np.allclose(r["state"], t["state"])
        row = np.concatenate([r["images"]["top"], t["images"]["top"], r["images"]["side"], t["images"]["side"]], 1)
        im = Image.fromarray(row).resize((1280, 240))
        ImageDraw.Draw(im).text((4, 4), f"#{i} ep{tw.labels['episode'][i]} t{tw.labels['t'][i]} {tw.labels['phase'][i]}",
                                fill=(255, 255, 255))
        rows.append(np.asarray(im))
    out = HERE / "out" / a.out
    Image.fromarray(np.concatenate(rows, 0)).save(out, quality=90)
    print("wrote", out)


if __name__ == "__main__":
    main()
