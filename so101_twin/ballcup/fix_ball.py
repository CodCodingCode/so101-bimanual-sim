"""Reject false 'ball' detections of the dino run (twin-independent): when the ball is hidden in the cup, DINO
sometimes boxes the black gripper as "a blue ball". A real ball mask is bright (white / blue plastic); keep a ball mask
only if the median max-channel value of its pixels is >= 110. Rewrites masks/ballcup_<cam>_d1_ball.npy (the raw
dino-run masks are kept as _d1_ballraw.npy).

    .venv/bin/python so101_twin/ballcup/fix_ball.py
"""
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))
from sets import Store  # noqa: E402


def main():
    fs = Store().frames("so101_ballcup_real")
    for c in ("top", "side"):
        raw = HERE / "masks" / f"ballcup_{c}_d1_ballraw.npy"
        if not raw.exists():
            shutil.copyfile(HERE / "masks" / f"ballcup_{c}_d1_ball.npy", raw)
        m = np.unpackbits(np.load(raw), axis=-1)[..., :640].astype(bool)
        imgs = np.load(fs.path / f"images_{c}.npy", mmap_mode="r")
        drop = []
        for i in range(fs.n):
            if m[i].sum() == 0:
                continue
            v = np.median(np.asarray(imgs[i]).max(-1)[m[i]])
            if v < 110:
                m[i] = False
                drop.append(i)
        np.save(HERE / "masks" / f"ballcup_{c}_d1_ball.npy", np.packbits(m, axis=-1))
        print(c, "dropped", len(drop), "dark 'ball' masks:", drop[:30])


if __name__ == "__main__":
    main()
