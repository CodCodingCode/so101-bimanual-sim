"""REDLEGO: rough real-arm masks without SAM (camera seed only, never scored): black arm on a white table / wall.

    .venv/bin/python so101_twin/redlego/fg_masks.py   # -> redlego/masks/redlego_{top,side}_fg.npy (packed)

Foreground = dark (max channel < DARK), not brown (the cardboard box in shadow: R - B > 25), closed 2 px, opened 1 px,
components < 150 px dropped. Includes the static base (no background subtraction)."""
import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))
from sets import Store  # noqa: E402

DARK = {"top": 70, "side": 70}


def main():
    fs = Store().frames("so101_redlego_real")
    for cam in ("top", "side"):
        imgs = np.load(fs.path / f"images_{cam}.npy", mmap_mode="r")
        out = np.zeros((fs.n, 480, 80), np.uint8)
        for i in range(fs.n):
            x = np.asarray(imgs[i]).astype(np.int16)
            dark = x.max(-1) < DARK[cam]
            brown = (x[..., 0] - x[..., 2]) > 25
            m = dark & ~brown
            m = ndi.binary_closing(m, iterations=2)
            m = ndi.binary_opening(m, iterations=1)
            lab, n = ndi.label(m)
            if n:
                sizes = ndi.sum(m, lab, range(1, n + 1))
                m = np.isin(lab, 1 + np.nonzero(sizes >= 150)[0])
            out[i] = np.packbits(m, axis=-1)
        np.save(HERE / "masks" / f"redlego_{cam}_fg.npy", out)
        area = np.unpackbits(out, axis=-1).reshape(fs.n, -1).mean(1)
        print(cam, "fg area mean", area.mean().round(4), "frames > 0.1%:", int((area > 0.001).sum()))


if __name__ == "__main__":
    main()
