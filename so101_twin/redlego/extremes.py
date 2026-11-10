"""REDLEGO guard-rail check: overlays (white = both, red = real only, blue = twin only) on the frames with the most
extreme joint values (min / max state of each of pan, lift, elbow, wrist flex), both cameras -> out/extremes.jpg.
Checks that the fitted per-joint scales (deg per unit) hold at the ends of each joint's range.
    .venv/bin/python so101_twin/redlego/extremes.py --fit out/final_fit.json --mask-tag _t1ok"""
import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fit", required=True)
ap.add_argument("--mask-tag", default="_t1ok")
a = ap.parse_args()
sc = F.load_scene("redlego")
fj = json.loads(pathlib.Path(a.fit).read_text())
sc.ARM = dataclasses.replace(sc.ARM, model=fj.get("model", sc.ARM.model))
P = F.reconcile(sc, fj["params"])
rig = F.Rig(sc, P)
fs = F.real_set(sc)
real = {c: F.load_masks("redlego", c, a.mask_tag) for c in sc.CAMERAS}
names = ("pan", "lift", "elbow", "wrist")
rows = []
for j, nm in enumerate(names):
    for which, i in (("min", int(np.argmin(fs.state[:, j]))), ("max", int(np.argmax(fs.state[:, j])))):
        tiles = []
        for c in sc.CAMERAS:
            im = np.asarray(fs.obs(i)["images"][c]).astype(np.float32)
            s = rig.mask(c, "all", F.frame_state(sc, P, c, i), 640, 480)
            r = real[c][i]
            ov = np.zeros((480, 640, 3), np.float32)
            ov[r & s] = 255
            ov[r & ~s] = (255, 40, 40)
            ov[~r & s] = (40, 90, 255)
            al = (r | s)[..., None]
            t = Image.fromarray(np.where(al, 0.4 * im + 0.6 * ov, im).astype(np.uint8))
            iou = (s & r).sum() / max((s | r).sum(), 1)
            ImageDraw.Draw(t).text((6, 6), f"{nm} {which} = {fs.state[i, j]:.1f} units (frame {i}) {c} IoU {iou:.3f}",
                                   fill=(255, 255, 0))
            tiles.append(np.asarray(t))
        rows.append(np.concatenate(tiles, 1))
g = Image.fromarray(np.concatenate(rows, 0))
g = g.resize((g.width // 2, g.height // 2), Image.BILINEAR)
g.save(HERE / "out" / "extremes.jpg", quality=85)
print("wrote", HERE / "out" / "extremes.jpg")
