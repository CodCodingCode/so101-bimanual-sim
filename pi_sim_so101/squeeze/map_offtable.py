#!/usr/bin/env python3
"""Where the REAL table ends (white floor beyond its curved edge) in table-plane coordinates: white pixels of a real
overhead frame back-projected with the calibrated camera onto z = 0, gridded at 1.5 cm. Writes
pi_sim/assets/offtable_cells.json; env.py draws those cells as visual-only floor-coloured patches on the sim table."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
rig = json.loads((HERE.parents[1] / "so101_actuation" / "cameras" / "so101_camera_rig.json").read_text())["cameras"]["overhead"]
fx, fy, cx, cy = (rig["intrinsics"][k] for k in ("fx", "fy", "cx", "cy"))
k1, k2 = rig["distortion"][:2]
pos = np.array(rig["pose"]["pos"])
w, x, y, z = rig["pose"]["quat_wxyz"]
Rwc = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(u, v, zp=0.0):
    xd, yd = (u - cx) / fx, (v - cy) / fy
    xn, yn = xd, yd
    for _ in range(30):
        r2 = xn * xn + yn * yn
        dd = 1 + k1 * r2 + k2 * r2 * r2
        xn, yn = xd / dd, yd / dd
    ray = Rwc @ np.array([xn, -yn, -1.0])
    return pos + (zp - pos[2]) / ray[2] * ray


img = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
H, W, _ = img.shape
bright = img.mean(2)
sat = img.max(2) - img.min(2)
white = (bright > 105) & (sat < 22)
vv, uu = np.mgrid[0:H, 0:W]
arms = ((vv > 150) & (uu < 210)) | ((vv > 160) & (uu > 360) & (uu < 555))     # the (white) arms, not floor
white &= ~arms
import scipy.ndimage as ndi                       # keep only big connected regions (light wood grain is speckle)
white = ndi.binary_opening(white, iterations=2)
lab, n = ndi.label(white)
sizes = ndi.sum(white, lab, range(1, n + 1))
white = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s > 3000])
white = ndi.binary_closing(white, iterations=3)
Image.fromarray(np.where(white[..., None], [255, 0, 255], img).astype(np.uint8)).save(HERE / "runs" / "offtable_mask.jpg")
pts = np.array([backproject(u, v)[:2] for v, u in zip(*np.nonzero(white[::3, ::3])) for v, u in [(v * 3, u * 3)]])
cell = 0.015
keys, counts = np.unique(np.round(pts / cell).astype(int), axis=0, return_counts=True)
cells = [(round(float(kx * cell), 4), round(float(ky * cell), 4)) for (kx, ky), c in zip(keys, counts) if c >= 3]
out = HERE.parent / "pi_sim" / "assets" / "offtable_cells.json"
out.write_text(json.dumps({"about": "table-plane cells (m) where the real overhead camera sees white floor beyond the "
                                    "real table's curved edge; from " + Path(sys.argv[1]).name, "cell_m": cell, "cells": cells}))
xs, ys = np.array(cells).T if cells else ([], [])
print(f"{len(cells)} off-table cells; x {min(xs):.2f}..{max(xs):.2f} m, y {min(ys):.2f}..{max(ys):.2f} m -> {out}")
