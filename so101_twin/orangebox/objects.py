"""Pixel detections of the scene objects in the TOP (overhead) view, for placing them in the twin (RGB only; the arm
metric never uses them).
  box : per episode, in the episode's first frame: largest dark (black) connected component away from the arm ->
        centroid, min-area-rectangle angle / corners (pixels)
  cube: per frame: largest yellow-green component -> centroid, pixel count, min-area-rectangle angle

    .venv/bin/python so101_twin/orangebox/objects.py  -> so101_twin/orangebox/out/objects_px.json
"""
import json
import pathlib
import sys

import numpy as np
from scipy import ndimage

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
sys.path.insert(0, str(HERE))
from sets import Store  # noqa: E402
from hue_masks import hsv, hue_mask  # noqa: E402
from segment import dilate  # noqa: E402


def min_area_rect(ys, xs):
    """angle (deg, 0..90), centre (x, y), half sizes, corners [4, 2] of the min-area rectangle of the pixels."""
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    best = None
    for a in np.arange(0, 90, 1.0):
        t = np.radians(a)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        q = (pts - c) @ R
        lo, hi = q.min(0), q.max(0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, a, R, lo, hi)
    _, a, R, lo, hi = best
    mid = (lo + hi) / 2
    cen = c + mid @ R.T
    hs = (hi - lo) / 2
    cor = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]) @ R.T + c
    return float(a), cen.tolist(), hs.tolist(), cor.tolist()


def largest(mask, min_px):
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes))
    if sizes[k] < min_px:
        return None
    return lab == k + 1


def main():
    fs = Store().frames("so101_orangebox_real")
    ep = fs.labels["episode"].astype(int)
    t = fs.labels["t"].astype(int)
    out = dict(box={}, cube={})
    for e in sorted(set(ep.tolist())):
        i0 = int(np.nonzero(ep == e)[0][np.argmin(t[ep == e])])
        rgb = np.asarray(fs.obs(i0)["images"]["top"])
        h, s, v = hsv(rgb)
        dark = (v < 0.33) & (s < 0.6)
        arm = dilate(hue_mask(rgb, "top"), 15)
        reg = largest(dark & ~arm, 3000)
        if reg is None:
            continue
        reg = ndimage.binary_fill_holes(reg)
        ys, xs = np.nonzero(reg)
        a, cen, hs, cor = min_area_rect(ys, xs)
        out["box"][str(e)] = dict(frame=i0, angle=a, center=cen, half=hs, corners=cor, px=int(reg.sum()))
    for i in range(fs.n):
        rgb = np.asarray(fs.obs(i)["images"]["top"])
        h, s, v = hsv(rgb)
        cube = (h > 48) & (h < 85) & (s > 0.35) & (v > 0.3)
        reg = largest(cube, 150)
        if reg is None:
            continue
        ys, xs = np.nonzero(reg)
        a, cen, hs, cor = min_area_rect(ys, xs)
        out["cube"][str(i)] = dict(angle=a, center=cen, half=hs, px=int(reg.sum()))
    (HERE / "out" / "objects_px.json").write_text(json.dumps(out, indent=1))
    print("boxes", len(out["box"]), "cube frames", len(out["cube"]))
    print({k: (round(v["center"][0]), round(v["center"][1]), round(v["angle"]), [round(x) for x in v["half"]])
           for k, v in list(out["box"].items())[:40]})


if __name__ == "__main__":
    main()
