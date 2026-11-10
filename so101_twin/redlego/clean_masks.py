"""REDLEGO (after jar/clean_masks.py): clean the twin-independent DINO+SAM arm masks for the first fit / seed and as the
twin-independent consensus silhouette: arm minus the lego mask and minus the box mask (both from the same real-frame
DINO+SAM run, dilated 2 px; a lego larger than 8 % of the image is DINO taking the arm for the brick and is ignored);
opened with a 5x5 square (thin servo cables); components kept if >= 15 % of the largest and within 30 px of its bbox.

    .venv/bin/python so101_twin/redlego/clean_masks.py --tag _d1   # -> masks/redlego_<cam>_d1k.npy
"""
import argparse
import pathlib

import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
M = HERE / "masks"


def load(p):
    return np.unpackbits(np.load(p), axis=-1)[..., :640].astype(bool)


def clean(arm, obj):
    m = arm & ~ndi.binary_dilation(obj, iterations=2)
    m = ndi.binary_opening(m, structure=np.ones((5, 5), bool))
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
    ap.add_argument("--suffix", default="k")
    a = ap.parse_args()
    for c in ("top", "side"):
        arm, lego, box = (load(M / f"redlego_{c}{a.tag}{s}.npy") for s in ("", "_lego", "_box"))
        out = np.stack([clean(arm[i], lego[i] | box[i]) for i in range(len(arm))])
        np.save(M / f"redlego_{c}{a.tag}{a.suffix}.npy", np.packbits(out, axis=-1))
        print(c, "area before", arm.mean().round(4), "after", out.mean().round(4))


if __name__ == "__main__":
    main()
