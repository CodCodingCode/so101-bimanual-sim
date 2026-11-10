"""GRAYBLOCK: twin-independent side masks without the black chair: _d1k minus the chair's static region (pixels that are
dark (max channel < 60) in the per-pixel median of the side frames, right of x = 400, dilated 2 px), 5x5 opening, main
components (as clean_masks.py). The top masks are copied unchanged. Real frames only, no twin input.
    .venv/bin/python so101_twin/grayblock/chair_cut.py   # -> masks/grayblock_<cam>_d1kc.npy
DINO+SAM merged the black arm with the black chair behind it in ~13 % of the side frames (whenever they touch)."""
import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clean_masks import clean, load  # noqa: E402

M = HERE / "masks"
bg = np.load(HERE / "out" / "bg_median_side.npy")
chair = bg.max(-1) < 60
chair[:, :400] = False
chair = ndi.binary_dilation(chair, iterations=2)
np.save(HERE / "out" / "chair_region_side.npy", chair)
for c in ("top", "side"):
    m = load(M / f"grayblock_{c}_d1k.npy")
    if c == "side":
        z = np.zeros((480, 640), bool)
        m = np.stack([clean(m[i] & ~chair, z, z) for i in range(len(m))])
    np.save(M / f"grayblock_{c}_d1kc.npy", np.packbits(m, axis=-1))
    a = m.reshape(len(m), -1).mean(1)
    print(c, "area mean", a.mean().round(4), "frames > 1.8x median", int((a > 1.8 * np.median(a)).sum()))
