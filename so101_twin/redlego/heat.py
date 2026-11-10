"""REDLEGO: per-pixel frequency of real-only (red) and twin-only (blue) arm pixels over all frames for a fit json
(static discrepancies: base, cables, fixtures) -> out/heat_<tag>.jpg
    .venv/bin/python so101_twin/redlego/heat.py --fit out/<x>.json --mask-tag _d1k --tag x"""
import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fit", required=True)
ap.add_argument("--mask-tag", default="_d1k")
ap.add_argument("--tag", default="x")
a = ap.parse_args()
sc = F.load_scene("redlego")
fj = json.loads(pathlib.Path(a.fit).read_text())
sc.ARM = dataclasses.replace(sc.ARM, model=fj.get("model", sc.ARM.model))
P = F.reconcile(sc, fj["params"])
rig = F.Rig(sc, P)
fs = F.real_set(sc)
rows = []
for c in sc.CAMERAS:
    R = F.load_masks("redlego", c, a.mask_tag)
    ro = np.zeros((480, 640))
    to = np.zeros((480, 640))
    for i in range(0, fs.n, 2):
        T = rig.mask(c, "all", F.frame_state(sc, P, c, i), 640, 480)
        ro += R[i] & ~T
        to += T & ~R[i]
    n = len(range(0, fs.n, 2))
    bg = np.asarray(fs.obs(0)["images"][c]).astype(np.float32) * 0.4
    img = bg.copy()
    img[..., 0] += 600 * ro / n
    img[..., 2] += 600 * to / n
    rows.append(np.clip(img, 0, 255).astype(np.uint8))
    print(c, "pixels red>30%:", int((ro / n > 0.3).sum()), "blue>30%:", int((to / n > 0.3).sum()))
Image.fromarray(np.concatenate(rows, 1)).save(HERE / "out" / f"heat_{a.tag}.jpg", quality=90)
