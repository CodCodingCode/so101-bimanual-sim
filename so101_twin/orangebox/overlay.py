"""Overlay sheet of a (partial) fit: white = twin & real mask, red = real only, blue = twin only.
    .venv/bin/python so101_twin/orangebox/overlay.py <params json> <mask tag> <out.jpg> [frames comma list]"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

pj = json.loads(pathlib.Path(sys.argv[1]).read_text())
tag = sys.argv[2]
sc = F.load_scene("orangebox")
if pj.get("model") and pj["model"] != sc.ARM.model:
    import dataclasses
    sc.ARM = dataclasses.replace(sc.ARM, model=pj["model"])
P = F.reconcile(sc, pj["params"])
rig = F.Rig(sc, P)
fs = F.real_set(sc)
frames = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else \
    list(np.nonzero(np.asarray(fs.labels["episode"]) % 2 == 1)[0][np.linspace(0, 127, 8).astype(int)])
rows = []
for c in sc.CAMERAS:
    real = F.load_masks("orangebox", c, tag)
    grp = F.frame_groups(sc, c)
    tiles = []
    for i in frames:
        s = rig.mask(c, grp[i], F.frame_state(sc, P, c, i), 640, 480)
        r = real[i]
        im = np.asarray(fs.obs(i)["images"][c]).astype(np.float32) * 0.5
        col = np.zeros((480, 640, 3), np.float32)
        col[r & s] = 255
        col[r & ~s] = (255, 40, 40)
        col[~r & s] = (40, 90, 255)
        a = (r | s)[..., None]
        t = Image.fromarray(np.where(a, 0.35 * im + 0.65 * col, im).astype(np.uint8))
        iou = (r & s).sum() / max((r | s).sum(), 1)
        ImageDraw.Draw(t).text((6, 6), f"{c} f{i} IoU {iou:.3f}", fill=(255, 255, 0))
        tiles.append(np.asarray(t))
    rows.append(np.concatenate(tiles[:4], 1))
    rows.append(np.concatenate(tiles[4:8], 1))
Image.fromarray(np.concatenate(rows, 0)).resize((1600, 1200)).save(sys.argv[3], quality=85)
