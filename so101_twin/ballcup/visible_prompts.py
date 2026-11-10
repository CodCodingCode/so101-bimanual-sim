"""SAM prompt silhouettes = twin arm silhouette minus the REAL cup and ball masks of the dino run (dilated 2 px; ball
masks filtered by fix_ball.py). Only a hint
for SAM prompting (positives / box / candidate choice); the scored twin masks never use real-image masks.

    .venv/bin/python so101_twin/ballcup/visible_prompts.py --src prompts_r1raw --dst prompts_r1
"""
import argparse
import json
import pathlib

import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
M = HERE / "masks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--obj-tag", default="_d1")
    a = ap.parse_args()
    (M / a.dst).mkdir(exist_ok=True)
    for c in ("top", "side"):
        P = np.unpackbits(np.load(M / a.src / f"ballcup_{c}.npy"), axis=-1)[..., :640].astype(bool)
        cup = np.unpackbits(np.load(M / f"ballcup_{c}{a.obj_tag}_cup.npy"), axis=-1)[..., :640].astype(bool)
        ball = np.unpackbits(np.load(M / f"ballcup_{c}{a.obj_tag}_ball.npy"), axis=-1)[..., :640].astype(bool)
        out = np.zeros_like(P)
        removed = []
        for i in range(len(P)):
            occ = cup[i] | ball[i]   # ball masks already filtered by fix_ball.py (dark false detections dropped)
            o = ndi.binary_dilation(occ, iterations=2)
            out[i] = P[i] & ~o
            removed.append((P[i] & o).sum() / max(P[i].sum(), 1))
        np.save(M / a.dst / f"ballcup_{c}.npy", np.packbits(out, axis=-1))
        print(c, "share of twin silhouette removed: median", round(float(np.median(removed)), 3), "mean",
              round(float(np.mean(removed)), 3))


if __name__ == "__main__":
    main()
