"""Visual check of the joint mapping: for each joint (pan, lift, elbow, wrist flex, wrist roll, gripper) the frames
with the lowest and the highest state value; per frame both cameras, real image with the real mask outline (green)
and the twin silhouette outline (red).

    .venv/bin/python so101_twin/pos5/joint_check.py out/pos5_r2_fit.json --masks _c3 --out out/joint_check.jpg
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import fit as F
import p5

NAMES = ("pan", "lift", "elbow", "wflex", "roll", "grip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit")
    ap.add_argument("--masks", default="_c3")
    ap.add_argument("--out", default="out/joint_check.jpg")
    a = ap.parse_args()
    sc = p5.scene()
    fs = F.real_set(sc)
    P = p5.load_params(p5.HERE / a.fit)
    st = fs.state
    idx, labs = [], []
    for j in range(6):
        for k, f in (("min", np.argmin), ("max", np.argmax)):
            i = int(f(st[:, j]))
            idx.append(i)
            labs.append(f"{NAMES[j]} {k} {st[i, j]:.0f}")
    tw = p5.twin_masks(P, frames=idx, sc=sc)
    R = {c: p5.load(c, a.masks) for c in p5.CAMS}
    tiles = []
    for k, (i, lab) in enumerate(zip(idx, labs)):
        row = []
        for c in p5.CAMS:
            im = np.asarray(fs.obs(i)["images"][c]).copy()
            r = R[c][i]
            t = tw[c][k]
            im[ndi.binary_dilation(r & ~ndi.binary_erosion(r))] = (0, 255, 0)
            im[ndi.binary_dilation(t & ~ndi.binary_erosion(t))] = (255, 0, 0)
            u = (r | t).sum()
            im = Image.fromarray(im)
            ImageDraw.Draw(im).text((6, 6), f"{lab}  frame {i}  {c} IoU {(r & t).sum() / max(u, 1):.3f}", fill=(255, 255, 0))
            row.append(np.asarray(im))
        tiles.append(np.concatenate(row, 1))
    g = np.concatenate([np.concatenate(tiles[j:j + 2], 1) for j in range(0, 12, 2)], 0)
    img = Image.fromarray(g)
    img = img.resize((img.width // 2, img.height // 2), Image.BILINEAR)
    img.save(p5.HERE / a.out, quality=85)
    print("wrote", p5.HERE / a.out)


if __name__ == "__main__":
    main()
