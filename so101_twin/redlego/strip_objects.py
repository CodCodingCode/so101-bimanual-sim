"""REDLEGO (after jar/strip_objects.py): remove the REAL lego-brick pixels (masks/redlego_<cam><obj>_lego.npy: DINO + SAM on
the real frames only, segment.py --mode dino) from a SAM arm-mask set: a held brick that SAM merged with the gripper is
not the arm. The box is NOT stripped here (the black arm never merges with the brown cardboard).

    .venv/bin/python so101_twin/redlego/strip_objects.py --tag _t1 [--obj-tag _d1]   # -> masks/redlego_<cam>_t1o.npy
"""
import argparse
import pathlib

import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
M = HERE / "masks"


def load(p):
    return np.unpackbits(np.load(p), axis=-1)[..., :640].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--obj-tag", default="_d1")
    a = ap.parse_args()
    for c in ("top", "side"):
        arm = load(M / f"redlego_{c}{a.tag}.npy")
        lego = load(M / f"redlego_{c}{a.obj_tag}_lego.npy")
        out = np.stack([arm[i] & ~ndi.binary_dilation(lego[i], iterations=1) for i in range(len(arm))])
        np.save(M / f"redlego_{c}{a.tag}o.npy", np.packbits(out, axis=-1))
        print(c, "area before", arm.mean().round(5), "after", out.mean().round(5),
              "frames changed", int((arm != out).reshape(len(arm), -1).any(1).sum()))


if __name__ == "__main__":
    main()
