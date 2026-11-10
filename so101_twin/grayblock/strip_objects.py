"""GRAYBLOCK copy of jar/strip_objects.py: remove the REAL gray-block pixels (masks/grayblock_<cam><occ>_block.npy:
DINO + SAM on the real frames only, segment.py --mode occ) from a SAM arm-mask set: a held block that SAM merged with the
gripper is not the arm. The box is NOT stripped here (its blue walls do not merge with the black arm; the occluder-
excluded reading handles the walls in front of the gripper).

    .venv/bin/python so101_twin/grayblock/strip_objects.py --tag _t1 [--occ-tag _occ]   # -> masks/grayblock_<cam>_t1o.npy
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
    ap.add_argument("--occ-tag", default="_occ")
    a = ap.parse_args()
    for c in ("top", "side"):
        arm = load(M / f"grayblock_{c}{a.tag}.npy")
        cube = load(M / f"grayblock_{c}{a.occ_tag}_block.npy")   # GRAYBLOCK
        out = np.stack([arm[i] & ~ndi.binary_dilation(cube[i], iterations=1) for i in range(len(arm))])
        np.save(M / f"grayblock_{c}{a.tag}o.npy", np.packbits(out, axis=-1))
        print(c, "area before", arm.mean().round(5), "after", out.mean().round(5),
              "frames changed", int((arm != out).reshape(len(arm), -1).any(1).sum()))


if __name__ == "__main__":
    main()
