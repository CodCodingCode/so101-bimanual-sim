"""Projection / ray-casting through a frozen twin2 camera (same model as twin2.Twin2.render: pinhole with the fitted
fovy, pose from armfit.cam_quat, radial lens k1/k2 and principal-point offset cx/cy applied on the real pixel grid).

Pixel convention: index coordinates (u = column, v = row) of the real image; the centre of pixel [r, c] is (c, r).
"""
from __future__ import annotations

import math

import numpy as np


def cam_R(p) -> np.ndarray:
    """Camera-to-world rotation, columns = camera x (right), y (up), z (backward) in world coordinates."""
    y, pt, r = map(math.radians, (p[3], p[4], p[5]))
    f = np.array([math.cos(pt) * math.cos(y), math.cos(pt) * math.sin(y), math.sin(pt)])
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    xr = math.cos(r) * right + math.sin(r) * up
    yu = -math.sin(r) * right + math.cos(r) * up
    return np.stack([xr, yu, np.cross(xr, yu)], axis=1)


def focal(p, h: int) -> float:
    return (h / 2) / math.tan(math.radians(p[6]) / 2)


def project(p, X, h: int, w: int):
    """World points (N,3) -> (u, v, depth) on the real (distorted) index grid. depth > 0 in front of the camera."""
    X = np.atleast_2d(np.asarray(X, float))
    R = cam_R(p)
    Xc = (X - np.asarray(p[:3], float)) @ R          # camera coords
    z = -Xc[:, 2]
    xn, yn = Xc[:, 0] / z, -Xc[:, 1] / z
    r2 = xn * xn + yn * yn
    d = 1 + p[7] * r2 + p[8] * r2 * r2
    f = focal(p, h)
    u = f * xn * d + w / 2 + p[9] - 0.5
    v = f * yn * d + h / 2 + p[10] - 0.5
    return u, v, z


def undistort(p, u, v, h: int, w: int, iters: int = 30):
    f = focal(p, h)
    xd = (np.asarray(u, float) + 0.5 - w / 2 - p[9]) / f
    yd = (np.asarray(v, float) + 0.5 - h / 2 - p[10]) / f
    xn, yn = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = xn * xn + yn * yn
        dd = 1 + p[7] * r2 + p[8] * r2 * r2
        xn, yn = xd / dd, yd / dd
    return xn, yn


def rays(p, u, v, h: int, w: int):
    """Unit world ray directions (N,3) through real pixels (u, v); origin = camera position."""
    xn, yn = undistort(p, u, v, h, w)
    dc = np.stack([xn, -yn, -np.ones_like(xn)], -1)
    dw = dc @ cam_R(p).T
    return dw / np.linalg.norm(dw, axis=-1, keepdims=True)


def cast_plane_z(p, u, v, h: int, w: int, z0: float = 0.0):
    """Intersections (N,3) of the pixel rays with the horizontal plane z = z0 (NaN where the ray points away)."""
    d = rays(p, u, v, h, w)
    o = np.asarray(p[:3], float)
    t = (z0 - o[2]) / d[:, 2]
    X = o + t[:, None] * d
    X[t <= 0] = np.nan
    return X


def cast_plane(p, u, v, h: int, w: int, n, c: float):
    """Intersections with the plane n . X = c."""
    d = rays(p, u, v, h, w)
    o = np.asarray(p[:3], float)
    n = np.asarray(n, float)
    t = (c - o @ n) / (d @ n)
    X = o + t[:, None] * d
    X[t <= 0] = np.nan
    return X
