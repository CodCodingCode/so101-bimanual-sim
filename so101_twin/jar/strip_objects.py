"""JAR (after ballcup/strip_objects.py): remove the REAL magic-cube pixels (masks/jar_<cam><occ>_cube.npy: DINO + SAM on
the real frames only, segment.py --mode occ) from a SAM arm-mask set: a held cube that SAM merged with the gripper is not
the arm. The jar is NOT stripped here (its bright wall never merges with the black arm; the white jaws in front of it
must stay).

    .venv/bin/python so101_twin/jar/strip_objects.py --tag _t1 [--occ-tag _occ]   # -> masks/jar_<cam>_t1o.npy
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
        arm = load(M / f"jar_{c}{a.tag}.npy")
        cube = load(M / f"jar_{c}{a.occ_tag}_cube.npy")
        out = np.stack([arm[i] & ~ndi.binary_dilation(cube[i], iterations=1) for i in range(len(arm))])
        np.save(M / f"jar_{c}{a.tag}o.npy", np.packbits(out, axis=-1))
        print(c, "area before", arm.mean().round(5), "after", out.mean().round(5),
              "frames changed", int((arm != out).reshape(len(arm), -1).any(1).sum()))


if __name__ == "__main__":
    main()
