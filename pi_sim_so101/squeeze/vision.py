"""Classical (no learning) overhead-camera detectors for the wide clamp: where is the red cube, where is the box.

Cube: the same rule as the rig runner (rig_tools/cube_locator.detect_red: brightness-relative red, largest blob), with
the blob centroid back-projected through the calibrated overhead camera onto the plane at cube-centre height.

Box: a MODEL FIT, not a centroid. The dark pixels (box, but also black servos / wires) are grouped into blobs; for each
big blob the real box (outer 10 x 10 cm, 3 cm tall) is projected through the calibrated camera at a candidate pose
(x, y, yaw) and its silhouette compared with the blob (IoU). Nelder-Mead from the back-projected centroid. The pose
with the best IoU wins if IoU >= MIN_IOU. A centroid would be biased by the box's visible outer wall (3 cm tall,
seen from behind and above); the fitted silhouette is not.

Both work on any RGB frame from the overhead camera (real 640x480 or the sim render with the same calibrated lens):
    cam = OverheadCamera.from_env(env.sim)            # calibrated intrinsics + pose from the sim model
    cube = detect_cube(rgb, cam)                       # (x, y) metres or None
    box = detect_box(rgb, cam)                         # (x, y, yaw, iou) or None
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "rig_tools")]
import cube_locator as CL          # noqa: E402  (detect_red, projection maths; read-only reuse)

CUBE_HALF = 0.016
BOX_OUTER, BOX_H = 0.10, 0.03
MIN_IOU = 0.6
# box pixels: dark AND grey. Measured 2026-11-12 on the overhead frames (V = max(R,G,B), S = max - min):
#   real box V ~23 (p95 29), S ~6 | real wood V 76-127, S 11-27 | sim box V ~59, S ~5 | sim wood V 77-121, S 6-20
# black servos / wires also pass; the model fit (IoU with the box silhouette) rejects them.
# Real shadows / lens vignetting make the wood near the bottom of the frame dark too, so several brightness cut-offs are
# tried and the best model fit (highest IoU) wins.
DARK_VS, DARK_S = (40, 55, 70), 18


@dataclass
class OverheadCamera:
    intr: tuple                     # fx, fy, cx, cy, k1, k2 (pixels of the 640x480 frame)
    pos: np.ndarray
    R: np.ndarray                   # camera-to-world rotation (MuJoCo cam_xmat)

    @classmethod
    def from_env(cls, sim, name="overhead"):
        import mujoco
        mujoco.mj_forward(sim.model, sim.data)
        pos, R = CL.camera_pose(sim.model, sim.data, name)
        return cls(CL.cam_intrinsics(sim.camera_specs[name]), pos, R)

    def project(self, p):
        return CL.project(self.intr, self.pos, self.R, p)

    def to_plane(self, u, v, z):
        o, d = CL.ray(self.intr, self.pos, self.R, u, v)
        if abs(d[2]) < 1e-6:
            return None
        t = (z - o[2]) / d[2]
        return None if t <= 0 else o + t * d


# ------------------------------------------------------------------ cube
def detect_cube(rgb, cam: OverheadCamera, min_px=30):
    """Red cube (x, y) on the table, metres, or None."""
    uv, npx = CL.detect_red(np.asarray(rgb), min_px=min_px)
    if uv is None:
        return None
    p = cam.to_plane(uv[0], uv[1], CUBE_HALF)
    return None if p is None else p[:2]


def detect_cube_pose(rgb, cam: OverheadCamera, min_px=30):
    """(x, y, yaw) of the cube, yaw in [-45, 45) deg (a cube repeats every 90 deg), or None. The red blob's outline is
    back-projected onto the cube's TOP face plane (z = 2 * half; the top face dominates the overhead view) and a
    minimum-area rectangle gives the angle; the position is the blob centroid on the cube-centre plane as in
    detect_cube. Classical, no learning."""
    import cv2
    import scipy.ndimage as ndi
    a = np.asarray(rgb).astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r > 55) & (r - g > 0.3 * r) & (r - b > 0.2 * r)          # same colour rule as cube_locator.detect_red
    mask = ndi.binary_opening(mask, iterations=1)
    lab, n = ndi.label(mask)
    if n == 0:
        return None
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < min_px:
        return None
    blob = (lab == k).astype(np.uint8)
    vs, us = np.nonzero(blob)
    c = cam.to_plane(us.mean(), vs.mean(), CUBE_HALF)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = [cam.to_plane(float(u), float(v), 2 * CUBE_HALF) for u, v in max(cnts, key=len)[:, 0, :]]
    pts = np.array([p[:2] for p in pts if p is not None], np.float32)
    if c is None or len(pts) < 5:
        return None
    (_, _), (_, _), ang = cv2.minAreaRect(pts * 1000.0)
    yaw = (np.radians(ang) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(c[0]), float(c[1]), float(yaw)


# ------------------------------------------------------------------ box
def _box_corners(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    h = BOX_OUTER / 2
    pts = []
    for z in (0.0, BOX_H):
        for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h)):
            pts.append((x + c * dx - s * dy, y + s * dx + c * dy, z))
    return np.array(pts)


def box_silhouette(cam, x, y, yaw, shape):
    """Filled image-space silhouette (bool HxW) of the box at (x, y, yaw)."""
    import cv2
    uv = [cam.project(p) for p in _box_corners(x, y, yaw)]
    out = np.zeros(shape, np.uint8)
    if any(q is None for q in uv):
        return out.astype(bool)
    hull = cv2.convexHull(np.round(np.array(uv)).astype(np.int32))
    cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool)


def dark_mask(rgb, v_max=DARK_VS[-1]):
    import scipy.ndimage as ndi
    a = np.asarray(rgb).astype(np.int32)
    v, s = a.max(2), a.max(2) - a.min(2)
    m = (v < v_max) & (s < DARK_S)
    m = ndi.binary_closing(m, iterations=2)
    return ndi.binary_opening(m, iterations=2)


def detect_box(rgb, cam: OverheadCamera, min_px=2500, candidates=3):
    """(x, y, yaw, iou) of the box on the table, or None. yaw in [-45, 45) deg (the box is square)."""
    best = None
    for v_max in DARK_VS:
        r = _fit_box(dark_mask(rgb, v_max), cam, min_px, candidates)
        if r is not None and (best is None or r[0] < best[0]):
            best = r
    if best is None or -best[0] < MIN_IOU:
        return None
    x, y, yaw = best[1]
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(x), float(y), float(yaw), float(-best[0])


def _fit_box(m, cam, min_px, candidates):
    import scipy.ndimage as ndi
    from scipy.optimize import minimize
    lab, n = ndi.label(m)
    if n == 0:
        return None
    sizes = ndi.sum(m, lab, range(1, n + 1))
    best = None
    for k in np.argsort(sizes)[::-1][:candidates]:
        if sizes[k] < min_px:
            break
        blob = lab == k + 1
        vs, us = np.nonzero(blob)
        p0 = cam.to_plane(us.mean(), vs.mean(), BOX_H / 2)
        if p0 is None:
            continue
        pad = 40
        v0, v1 = max(vs.min() - pad, 0), min(vs.max() + pad, blob.shape[0])
        u0, u1 = max(us.min() - pad, 0), min(us.max() + pad, blob.shape[1])
        roi = blob[v0:v1, u0:u1]

        def neg_iou(q):
            sil = box_silhouette(cam, q[0], q[1], q[2], blob.shape)[v0:v1, u0:u1]
            inter = np.logical_and(sil, roi).sum()
            union = np.logical_or(sil, roi).sum()
            return -inter / max(union, 1)

        fits = []
        for yaw0 in np.radians((-30, 0, 30)):
            r = minimize(neg_iou, [p0[0], p0[1], yaw0], method="Nelder-Mead",
                         options={"xatol": 5e-4, "fatol": 1e-4, "initial_simplex": [
                             [p0[0], p0[1], yaw0], [p0[0] + 0.02, p0[1], yaw0], [p0[0], p0[1] + 0.02, yaw0],
                             [p0[0], p0[1], yaw0 + 0.3]], "maxiter": 300})
            fits.append((r.fun, r.x))
        f, q = min(fits, key=lambda t: t[0])
        if best is None or f < best[0]:
            best = (f, q)
    return best
