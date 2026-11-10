"""Sanity stats of a SAM mask set: empty / visible frames per camera, held-out split, and an independent visibility check
(background-subtraction arm-like pixels, stack2/masks/stack_<cam>_fg.npy) to flag frames excluded from the metric
although something arm-like moved in view.

    .venv/bin/python so101_twin/stack2/mask_stats.py _s1
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

tag = sys.argv[1]
sc = F.load_scene("stack")
fs = F.real_set(sc)
ep = np.asarray(fs.labels["episode"]).astype(int)
for c in sc.CAMERAS:
    m = np.unpackbits(np.load(F.TWIN / "masks" / f"stack_{c}{tag}.npy"), axis=-1)[..., :640].astype(bool)
    fg = np.unpackbits(np.load(HERE / "masks" / f"stack_{c}_fg.npy"), axis=-1)[..., :640].astype(bool)
    area = m.reshape(fs.n, -1).mean(1)
    fga = fg.reshape(fs.n, -1).mean(1)
    vis = area >= F.VISIBLE
    sus = (~vis) & (fga >= 0.01)
    print(f"{c}: SAM visible {vis.sum()}/{fs.n} (held-out {(vis & (ep % 2 == 1)).sum()}/{(ep % 2 == 1).sum()}), "
          f"area median {np.median(area[vis]):.3f}; excluded {(~vis).sum()} (held-out {(~vis & (ep % 2 == 1)).sum()}); "
          f"excluded but >=1% moving pixels: {sus.sum()} -> {np.nonzero(sus)[0][:20].tolist()}")
