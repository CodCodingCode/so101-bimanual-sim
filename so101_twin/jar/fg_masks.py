"""JAR copy of ballcup/fg_masks.py (jar set, DARK 75). Rough real-arm masks without SAM (seed for the first camera fit only, never scored).

    .venv/bin/python so101_twin/ballcup/fg_masks.py   # -> ballcup/masks/ballcup_{top,side}_fg.npy (packed)

Both cameras are static (static_check.py: top 0 px between episodes; side <= 3 px, the box backdrop moves, the table
and base do not). Background = per-pixel median over all 320 frames. Foreground = dark (max channel < DARK) AND
changed from the background by > 40 (grey level), not green-dominant (the dark mat), opened 1 px, components < 150 px
dropped. The always-present base is part of the background, so these masks miss the base; fine for a seed.
"""
import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))
from sets import Store  # noqa: E402

DARK = {"top": 75, "side": 75}   # JAR


def main():
    fs = Store().frames("so101_jar_real")
    for cam in ("top", "side"):
        imgs = np.load(fs.path / f"images_{cam}.npy", mmap_mode="r")
        bg = np.median(np.asarray(imgs).astype(np.float32), axis=0)
        np.save(HERE / "out" / f"bg_median_{cam}.npy", bg.astype(np.uint8))
        out = np.zeros((fs.n, 480, 80), np.uint8)
        for i in range(fs.n):
            x = np.asarray(imgs[i]).astype(np.float32)
            diff = np.abs(x.mean(-1) - bg.mean(-1))
            dark = x.max(-1) < DARK[cam]
            green = (x[..., 1] > x[..., 0] + 12) & (x[..., 1] > x[..., 2] + 4)
            m = dark & (diff > 40) & ~green
            m = ndi.binary_opening(m, iterations=1)
            lab, n = ndi.label(m)
            if n:
                sizes = ndi.sum(m, lab, range(1, n + 1))
                m = np.isin(lab, 1 + np.nonzero(sizes >= 150)[0])
            out[i] = np.packbits(m, axis=-1)
        np.save(HERE / "masks" / f"jar_{cam}_fg.npy", out)
        area = np.unpackbits(out, axis=-1).reshape(fs.n, -1).mean(1)
        print(cam, "fg area mean", area.mean().round(4), "frames > 0.1%:", int((area > 0.001).sum()))


if __name__ == "__main__":
    main()
