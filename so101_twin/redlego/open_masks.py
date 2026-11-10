"""REDLEGO: remove thin cables from a SAM arm-mask set (twin-independent post-processing, as clean_masks.py does for the
DINO masks): opening with a 5x5 square (structures thinner than 5 px: the servo cables, the black USB cable), then
components kept if >= 15 % of the largest and within 30 px of its bounding box.

    .venv/bin/python so101_twin/redlego/open_masks.py --tag _t1o   # -> masks/redlego_<cam>_t1ok.npy
"""
import argparse
import pathlib

import numpy as np

from clean_masks import clean, load

HERE = pathlib.Path(__file__).resolve().parent
M = HERE / "masks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--suffix", default="k")
    a = ap.parse_args()
    for c in ("top", "side"):
        arm = load(M / f"redlego_{c}{a.tag}.npy")
        empty = np.zeros_like(arm[0])
        out = np.stack([clean(arm[i], empty) for i in range(len(arm))])
        np.save(M / f"redlego_{c}{a.tag}{a.suffix}.npy", np.packbits(out, axis=-1))
        print(c, "area before", arm.mean().round(5), "after", out.mean().round(5))


if __name__ == "__main__":
    main()
