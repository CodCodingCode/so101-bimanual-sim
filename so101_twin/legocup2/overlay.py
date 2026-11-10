"""Overlay sheet for a params json: per frame real image with SAM mask (red only), twin (blue only), both (white).
    .venv/bin/python so101_twin/legocup2/overlay.py PARAMS.json CAM OUT.jpg [--frames 1,2,3] [--tag _lc2] [--partial]"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("params")
ap.add_argument("cam")
ap.add_argument("out")
ap.add_argument("--frames", default="")
ap.add_argument("--tag", default="")
ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()
sc = F.load_scene("legocup")
fs = F.real_set(sc)
P = F.reconcile(sc, json.loads(pathlib.Path(a.params).read_text())["params"])
rig = F.Rig(sc, P)
real = F.load_masks("legocup", a.cam, a.tag)
ep = np.asarray(fs.labels["episode"]).astype(int)
if a.frames:
    idx = [int(x) for x in a.frames.split(",")]
else:
    vis = np.nonzero((real.reshape(fs.n, -1).mean(1) > 0.005) & (ep % 2 == 1))[0]
    idx = vis[np.linspace(0, len(vis) - 1, a.n).astype(int)].tolist()
tiles = []
for i in idx:
    im = np.asarray(fs.obs(i)["images"][a.cam]).astype(np.float32) * 0.5
    s = rig.mask(a.cam, "all", F.frame_state(sc, P, a.cam, i), 640, 480)
    r = real[i]
    ov = np.zeros((480, 640, 3), np.float32)
    ov[r & s] = 255
    ov[r & ~s] = (255, 40, 40)
    ov[~r & s] = (40, 90, 255)
    m = (r | s)[..., None]
    t = np.where(m, 0.35 * im + 0.65 * ov, im).astype(np.uint8)
    iou = (r & s).sum() / max((r | s).sum(), 1)
    t = Image.fromarray(t)
    ImageDraw.Draw(t).text((6, 6), f"#{i} ep{ep[i]} {fs.labels['phase'][i]} IoU {iou:.2f}", fill=(255, 255, 0))
    tiles.append(np.asarray(t))
k = 4
rows = [np.concatenate(tiles[j:j + k], 1) for j in range(0, len(tiles) - len(tiles) % k, k)]
im = Image.fromarray(np.concatenate(rows, 0))
im = im.resize((im.width // 2, im.height // 2))
im.save(a.out, quality=85)
print("frames", idx)
