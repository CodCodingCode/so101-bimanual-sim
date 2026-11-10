"""Twin-free chroma-key masks of the real arm (green screen), from the real frames only (no twin, no SAM).

    .venv/bin/python so101_twin/pos5/ck_masks.py [--tag _ck] [--sat 50]

Per camera: Bg = per-pixel median of the 320 real frames (bg.py). Per frame:
  green  = PIL-HSV hue in [80, 120) and saturation >= --sat (the screen: hue 100-112, sat 60-160; the arm: sat ~17)
  cube   = hue in [120, 150], sat >= 60, value >= 50 (the cyan cube; dilated 1 px)
  static = pixels not-green in >= 97 % of frames (box, wall corner, mat marks, clamp), outside the arm-base box
           BASE_BOX (the static arm base, read off the median image); a static pixel counts as background unless
           the frame differs there from Bg by > --chg (something, i.e. the arm, is in front of it)
  arm    = not green & not cube & not unchanged-static, 1-px opening, components >= 120 px that link (3-px dilation)
           to the base box or the image border, holes < 300 px filled.
Writes masks/pos5_<cam><tag>.npy and masks/pos5_<cam>_cube.npy (the real cube pixels, dilated 2 px: the occluder
mask for the excluded reading) and masks/pos5_<cam>_static.npy (1 frame: the static non-arm pixels).
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import fit as F
import p5

BASE_BOX = {"top": (440, 480, 250, 355), "side": (175, 285, 0, 125)}   # (y0, y1, x0, x1) full-res px


def hsv(rgb):
    return np.asarray(Image.fromarray(rgb).convert("HSV")).astype(np.int16)


def keys(rgb, sat):
    h = hsv(rgb)
    green = (h[..., 0] >= 80) & (h[..., 0] < 120) & (h[..., 1] >= sat)
    cube = (h[..., 0] >= 120) & (h[..., 0] <= 150) & (h[..., 1] >= 60) & (h[..., 2] >= 50)
    return green, cube


def base_mask(cam):
    m = np.zeros((480, 640), bool)
    y0, y1, x0, x1 = BASE_BOX[cam]
    m[y0:y1, x0:x1] = True
    return m


def clean(m, base, min_px=120, link=3, hole=300):
    m = ndi.binary_opening(m, iterations=1)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m &= np.isin(lab, 1 + np.flatnonzero(sizes >= min_px))
    grp, ng = ndi.label(ndi.binary_dilation(m, iterations=link))
    border = np.zeros_like(m)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    sel = np.unique(grp[(base | border) & (grp > 0)])
    m &= np.isin(grp, sel)
    holes, nh = ndi.label(~m)
    if nh:
        hs = ndi.sum(~m, holes, range(1, nh + 1))
        m |= np.isin(holes, 1 + np.flatnonzero(hs < hole))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_ck")
    ap.add_argument("--sat", type=int, default=50)
    ap.add_argument("--chg", type=int, default=28)
    a = ap.parse_args()
    fs = F.real_set(p5.scene())
    for cam in p5.CAMS:
        imgs = np.asarray(fs._img[cam])
        bg = np.load(p5.MASKS / f"bg_{cam}.npy").astype(np.int16)
        base = base_mask(cam)
        G = np.zeros((fs.n, 480, 640), bool)
        C = np.zeros((fs.n, 480, 640), bool)
        for i in range(fs.n):
            G[i], C[i] = keys(imgs[i], a.sat)
        static = ((~G).mean(0) >= 0.97) & ~base
        static = ndi.binary_dilation(static, iterations=2) & ~base
        out = np.zeros((fs.n, 480, 640), bool)
        cube = np.zeros((fs.n, 480, 640), bool)
        for i in range(fs.n):
            chg = np.abs(imgs[i].astype(np.int16) - bg).max(-1) > a.chg
            cb = ndi.binary_dilation(ndi.binary_opening(C[i], iterations=1), iterations=1)
            m = ~G[i] & ~cb & ~(static & ~chg)
            out[i] = clean(m, base)
            cube[i] = ndi.binary_dilation(cb, iterations=1)
        p5.save(cam, a.tag, out)
        p5.save(cam, "_cube", cube)
        p5.save(cam, "_static", static[None])
        print(cam, "mean area", round(float(out.mean()), 4), "static frac", round(float(static.mean()), 4),
              "cube px/frame", int(cube.reshape(fs.n, -1).sum(1).mean()), flush=True)


if __name__ == "__main__":
    main()
