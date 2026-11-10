"""Clean the twin-independent DINO+SAM arm masks for the FIRST fit only (seed; the scored masks come from a later
SAM round): arm minus the ball / cup masks (dilated 2 px), opened with a 7x7 square (drops the thin black power cable
and the plug's cord), components kept only if >= 15 % of the largest and within 30 px of its bounding box.

    .venv/bin/python so101_twin/ballcup/clean_masks.py --tag _d1   # -> masks/ballcup_<cam>_d1c.npy
"""
import argparse
import pathlib

import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
M = HERE / "masks"


def load(p):
    return np.unpackbits(np.load(p), axis=-1)[..., :640].astype(bool)


def clean(arm, ball, cup):
    m = arm & ~ndi.binary_dilation(ball | cup, iterations=2)
    m = ndi.binary_opening(m, structure=np.ones((7, 7), bool))
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    sl = ndi.find_objects(lab)[big - 1]
    y0, y1, x0, x1 = sl[0].start - 30, sl[0].stop + 30, sl[1].start - 30, sl[1].stop + 30
    keep = np.zeros(n + 1, bool)
    keep[big] = True
    for k, s in enumerate(ndi.find_objects(lab), start=1):
        if k == big or sizes[k - 1] < 0.15 * sizes[big - 1]:
            continue
        if s[0].stop > y0 and s[0].start < y1 and s[1].stop > x0 and s[1].start < x1:
            keep[k] = True
    return keep[lab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_d1")
    a = ap.parse_args()
    for c in ("top", "side"):
        arm, ball, cup = (load(M / f"ballcup_{c}{a.tag}{s}.npy") for s in ("", "_ball", "_cup"))
        out = np.stack([clean(arm[i], ball[i], cup[i]) for i in range(len(arm))])
        np.save(M / f"ballcup_{c}{a.tag}c.npy", np.packbits(out, axis=-1))
        print(c, "area before", arm.mean().round(4), "after", out.mean().round(4))


if __name__ == "__main__":
    main()
