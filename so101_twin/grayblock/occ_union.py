"""GRAYBLOCK: the REAL occluder masks of the occluder-excluded reading = near blue box faces (occ_near.py, _occn) UNION
the real gray-block masks (segment.py --mode occ, _occ_block: a held block can hide jaw pixels). Real frames only.
    .venv/bin/python so101_twin/grayblock/occ_union.py   # -> masks/grayblock_<cam>_occx.npy"""
import pathlib

import numpy as np

M = pathlib.Path(__file__).resolve().parent / "masks"
for c in ("top", "side"):
    a = np.load(M / f"grayblock_{c}_occn.npy")
    b = np.load(M / f"grayblock_{c}_occ_block.npy")
    np.save(M / f"grayblock_{c}_occx.npy", a | b)
    m = np.unpackbits(a | b, axis=-1)[..., :640]
    print(c, "occluder area mean", round(float(m.mean()), 4))
