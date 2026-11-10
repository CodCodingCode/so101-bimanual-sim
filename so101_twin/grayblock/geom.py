"""GRAYBLOCK copy of so101_twin/jar/geom.py (unchanged except scene / set / mask names: grayblock, so101_grayblock_real). JAR copy of ballcup/geom.py. Camera rays for the twin (pixel -> world ray through a fitted camera with the arm.Lens model)."""
import math

import numpy as np


def mat_from_quat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cam_ray(P, cam, u, v, w=640, h=480):
    """world ray (origin, unit dir) through pixel (u, v) (pixel index; arm.Lens pixel-centre convention), undistorted."""
    p = P["cams"][cam]
    g = p["groups"]["all"]
    L = p["lens"]
    R = mat_from_quat(g["quat"])
    f = 0.5 * h / math.tan(math.radians(p["fovy"]) / 2)
    xd = (u + 0.5 - w / 2 - L["cx"] * w) / f
    yd = (v + 0.5 - h / 2 - L["cy"] * h) / f
    xu, yu = xd, yd
    for _ in range(20):
        r2 = xu * xu + yu * yu
        s = 1 + L["k1"] * r2 + L["k2"] * r2 * r2
        xu, yu = xd / s, yd / s
    d = R @ np.array([xu, -yu, -1.0])
    return np.asarray(g["pos"], float), d / np.linalg.norm(d)


def hit_z(o, d, z):
    t = (z - o[2]) / d[2]
    return o + t * d


def closest(o1, d1, o2, d2):
    w = o1 - o2
    a, b, c, dd, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w, d2 @ w
    den = a * c - b * b
    s = (b * e - c * dd) / den
    t = (a * e - b * dd) / den
    p1, p2 = o1 + s * d1, o2 + t * d2
    return 0.5 * (p1 + p2), float(np.linalg.norm(p1 - p2))
