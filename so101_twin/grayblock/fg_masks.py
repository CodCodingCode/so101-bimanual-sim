"""GRAYBLOCK (after jar/fg_masks.py): rough real-arm masks WITHOUT SAM (seed for the first camera search only, never
scored, never fitted officially).
    .venv/bin/python so101_twin/grayblock/fg_masks.py   # -> masks/grayblock_{top,side}_fg.npy (packed)
Both cameras are static. Foreground = dark (max channel < 60: the black arm on the white table / wall), opened with a
5x5 square (drops the thin black power cable), components < 300 px dropped. Side view: the black chair in the
background is removed as the pixels that are dark in the per-pixel median over all 320 frames with x >= 400 (the
chair occupies the upper right; the arm base / rest pose sits left of x = 400).
"""
import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))
from sets import Store  # noqa: E402

DARK = 60


def main():
    fs = Store().frames("so101_grayblock_real")
    for cam in ("top", "side"):
        imgs = np.load(fs.path / f"images_{cam}.npy", mmap_mode="r")
        bg = np.median(np.asarray(imgs[::2]), axis=0)
        np.save(HERE / "out" / f"bg_median_{cam}.npy", bg.astype(np.uint8))
        excl = np.zeros((480, 640), bool)
        if cam == "side":
            excl = bg.max(-1) < DARK
            excl[:, :400] = False
            excl = ndi.binary_dilation(excl, iterations=3)
        out = np.zeros((fs.n, 480, 80), np.uint8)
        for i in range(fs.n):
            x = np.asarray(imgs[i])
            m = (x.max(-1) < DARK) & ~excl
            m = ndi.binary_opening(m, structure=np.ones((5, 5), bool))
            lab, n = ndi.label(m)
            if n:
                sizes = ndi.sum(m, lab, range(1, n + 1))
                m = np.isin(lab, 1 + np.nonzero(sizes >= 300)[0])
            out[i] = np.packbits(m, axis=-1)
        np.save(HERE / "masks" / f"grayblock_{cam}_fg.npy", out)
        area = np.unpackbits(out, axis=-1).reshape(fs.n, -1).mean(1)
        print(cam, "fg area mean", area.mean().round(4), "frames > 0.1%:", int((area > 0.001).sum()))


if __name__ == "__main__":
    main()
