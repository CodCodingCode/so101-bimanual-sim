"""Side-camera prompts from the REAL image (not the twin): moving (background-subtracted) pixels that are white-ish
(the arm is white; the black wrist-camera cable is excluded), upper image only, components >= 300 px touching the top or left image edge (the arm enters there).

    .venv/bin/python so101_twin/stack2/side_fgw_prompts.py OUT_DIR --twin-dir DIR

OUT_DIR/stack_side.npy = that mask where it has >= 300 px, else the twin silhouette from --twin-dir; OUT_DIR/stack_top.npy
= the twin silhouette (copied). Used for the side SAM round when the twin was still misplaced at the green block.
"""
import argparse
import pathlib
import shutil
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))
from cube_track import hsv  # noqa: E402
from sets import Store  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--twin-dir", required=True)
a = ap.parse_args()
fs = Store().frames("so101_stack_real")
imgs = np.load(fs.path / "images_side.npy", mmap_mode="r")
bg = np.median(np.asarray(imgs).astype(np.float32), axis=0)
twin = np.unpackbits(np.load(pathlib.Path(a.twin_dir) / "stack_side.npy"), axis=-1)[..., :640].astype(bool)
out = np.zeros((fs.n, 480, 80), np.uint8)
used = 0
for i in range(fs.n):
    x = np.asarray(imgs[i]).astype(np.float32)
    diff = np.abs(x - bg).max(-1)
    h, s, v = hsv(x.astype(np.uint8))
    white = (v > 0.62) & (s < 0.22)
    m = (diff > 30) & white
    m[265:] = False          # the arm enters from the top; below is table / disc (lighting changes)
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        edge = set(np.unique(lab[0, :])) | set(np.unique(lab[:, 0]))      # the arm always enters from the top / left
        keep = [k + 1 for k in range(n) if sizes[k] >= 300 and (k + 1) in edge]
        m = np.isin(lab, keep)
    if m.sum() >= 300:
        used += 1
    else:
        m = twin[i]
    out[i] = np.packbits(m, axis=-1)
o = pathlib.Path(a.out)
o.mkdir(parents=True, exist_ok=True)
np.save(o / "stack_side.npy", out)
shutil.copyfile(pathlib.Path(a.twin_dir) / "stack_top.npy", o / "stack_top.npy")
print(f"side: real white-moving prompt in {used}/{fs.n} frames, twin elsewhere")
