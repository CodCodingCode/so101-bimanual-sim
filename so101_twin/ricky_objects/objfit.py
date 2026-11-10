"""Shared object-placement helpers: silhouettes of simple solids projected through the frozen twin cameras,
IoU against detected masks, per-episode pose fits with shared shape parameters."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

import camgeo


def frustum_pts(x, y, z0, rb, rt, h, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    bot = np.stack([x + rb * c, y + rb * s, np.full(n, z0)], 1)
    top = np.stack([x + rt * c, y + rt * s, np.full(n, z0 + h)], 1)
    return np.concatenate([bot, top])


def box_pts(x, y, z, hx, hy, hz, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                lx, ly = sx * hx, sy * hy
                pts.append((x + c * lx - s * ly, y + s * lx + c * ly, z + sz * hz))
    return np.array(pts)


def hull_mask(p, P, h, w, crop):
    """Convex-hull silhouette of world points P on the real grid, restricted to crop (y0, y1, x0, x1)."""
    u, v, z = camgeo.project(p, P, h, w)
    y0, y1, x0, x1 = crop
    m = Image.new("L", (x1 - x0, y1 - y0), 0)
    if (z <= 0.01).any():
        return np.zeros((y1 - y0, x1 - x0), bool)
    pts = np.stack([u - x0, v - y0], 1)
    try:
        hv = ConvexHull(pts).vertices
    except Exception:
        return np.zeros((y1 - y0, x1 - x0), bool)
    ImageDraw.Draw(m).polygon([tuple(q) for q in pts[hv]], fill=1)
    return np.asarray(m, bool)


def crop_of(mask, margin, h, w):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return (max(0, ys.min() - margin), min(h, ys.max() + margin + 1), max(0, xs.min() - margin), min(w, xs.max() + margin + 1))


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def centroid_err(p, P, mask, crop, h, w):
    """Pixel distance between the model silhouette centroid and the detected mask centroid."""
    m = hull_mask(p, P, h, w, crop)
    if not m.any() or not mask.any():
        return np.nan
    a = np.argwhere(m).mean(0)
    b = np.argwhere(mask).mean(0)
    return float(np.linalg.norm(a - b))
