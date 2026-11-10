"""GRAYBLOCK copy of jar/clean_masks.py: clean the twin-independent DINO+SAM arm masks for the FIRST fit / seed and as
the twin-independent consensus silhouette: arm minus the gray-block mask (rejected when > 2 % of the image: DINO took
the arm for the block) and minus the BLUE pixels of the box mask (b >= 70, b - r >= 45, b - g >= 20), both dilated 2 px;
opened with a 5x5 square (thin servo wires, power cable); components kept if >= 15 % of the largest and within 30 px
of its bounding box. (JAR original: arm minus the cube mask and minus the BRIGHT part of the jar mask.)

    .venv/bin/python so101_twin/grayblock/clean_masks.py --tag _d1   # -> masks/jar_<cam>_d1k.npy
(_d1c = the first version without the cube-size check: it emptied 86 side frames where DINO took the arm for the cube)
"""
import argparse
import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
M = HERE / "masks"
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))


def load(p):
    return np.unpackbits(np.load(p), axis=-1)[..., :640].astype(bool)


def clean(arm, cube, jarb):
    m = arm & ~ndi.binary_dilation(cube | jarb, iterations=2)
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
    ap.add_argument("--suffix", default="k", help="output tag suffix (first version 'c' had no cube-size check)")
    a = ap.parse_args()
    from sets import Store

    fs = Store().frames("so101_grayblock_real")
    for c in ("top", "side"):
        imgs = np.load(fs.path / f"images_{c}.npy", mmap_mode="r")
        arm, cube, jar = (load(M / f"grayblock_{c}{a.tag}{s}.npy") for s in ("", "_block", "_box"))   # GRAYBLOCK
        # a 'block' mask larger than 2 % of the image is DINO labelling the arm as the block
        okc = cube.reshape(len(cube), -1).mean(1) < 0.02

        def blue(i):
            x = np.asarray(imgs[i]).astype(np.int16)
            return (x[..., 2] >= 70) & (x[..., 2] - x[..., 0] >= 45) & (x[..., 2] - x[..., 1] >= 20)
        out = np.stack([clean(arm[i], cube[i] & okc[i], jar[i] & blue(i)) for i in range(len(arm))])
        np.save(M / f"grayblock_{c}{a.tag}{a.suffix}.npy", np.packbits(out, axis=-1))
        print(c, "block masks rejected (> 2 % of the image):", int((~okc).sum()))
        print(c, "area before", arm.mean().round(4), "after", out.mean().round(4))


if __name__ == "__main__":
    main()
