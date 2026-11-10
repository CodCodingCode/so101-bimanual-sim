"""Rough real-arm foreground masks by background subtraction (no SAM, no twin): seed for the first camera fit only.

    .venv/bin/python so101_twin/stack2/fg_masks.py     # -> stack2/masks/stack_{top,side}_fg.npy (packed, like SAM masks)

top: per camera session (scene top_group), background = per-pixel median over the session's NON-rest frames (arm moving,
so each pixel is mostly background); side: static camera, median over all frames. Foreground = colour change > 35 that
is not a shadow (darker with the same chromaticity) and not the wooden cube (wood hue), opened by 1 px; components
< 150 px dropped. The arm base / shoulder that never moves is in the background, so these masks under-cover the arm
near the base; only used to seed the first fit before SAM.
"""
import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common" / "scenes"))
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))
from cube_track import hsv  # noqa: E402
from sets import Store  # noqa: E402


def main():
    import stack as S
    fs = Store().frames("so101_stack_real")
    ep = np.asarray(fs.labels["episode"]).astype(int)
    rest = (fs.state[:, 1] < -90) & (fs.state[:, 2] > 85)
    (HERE / "masks").mkdir(exist_ok=True)
    for cam in ("top", "side"):
        imgs = np.load(fs.path / f"images_{cam}.npy", mmap_mode="r")
        grp = np.array([S.top_group(e) if cam == "top" else "all" for e in ep])
        out = np.zeros((fs.n, 480, 80), np.uint8)
        for g in sorted(set(grp)):
            idx = np.nonzero(grp == g)[0]
            ref = idx[~rest[idx]] if cam == "top" and (~rest[idx]).sum() >= 8 else idx
            bg = np.median(np.asarray(imgs[ref]).astype(np.float32), axis=0)
            hb, sb, vb = hsv(bg.astype(np.uint8))
            bg_wood = ndi.binary_dilation((hb > 12) & (hb < 45) & (sb > 0.2) & (vb > 0.3), iterations=6)
            for i in idx:
                x = np.asarray(imgs[i]).astype(np.float32)
                diff = np.abs(x - bg).max(-1)
                ratio = (x + 5) / (bg + 5)
                shadow = (x.mean(-1) < bg.mean(-1) - 8) & ((ratio.max(-1) - ratio.min(-1)) < 0.12)
                h, s, v = hsv(x.astype(np.uint8))
                wood = (h > 12) & (h < 45) & (s > 0.28) & (v > 0.35)
                m = (diff > 35) & ~shadow & ~ndi.binary_dilation(wood, iterations=3) & ~bg_wood
                if cam == "side":
                    m[330:] = False     # table below the disc: shadows / reflections only
                m = ndi.binary_opening(m, iterations=1)
                lab, n = ndi.label(m)
                if n:
                    sizes = ndi.sum(m, lab, range(1, n + 1))
                    m = np.isin(lab, 1 + np.nonzero(sizes >= 150)[0])
                out[i] = np.packbits(m, axis=-1)
        np.save(HERE / "masks" / f"stack_{cam}_fg.npy", out)
        area = np.unpackbits(out, axis=-1).reshape(fs.n, -1).mean(1)
        print(cam, "fg area mean", area.mean().round(4), "frames > 0.1%:", int((area > 0.001).sum()))


if __name__ == "__main__":
    main()
