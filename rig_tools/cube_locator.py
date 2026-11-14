"""Where is the red cube? Triangulated from all three cameras (overhead + both wrist cameras), for the REAL rig and,
with the same geometry, in sim.

Each camera that sees red gives a ray: blob centroid pixel -> undistort (calibrated k1, k2) -> camera frame -> world
(overhead: fixed calibrated pose; wrist cameras: pose from the arm's joint angles through the kinematic model, i.e.
MuJoCo's cam_xpos / cam_xmat after setting the measured joints). Two or more rays: least-squares closest point (gives
the height too, e.g. when the cube is lifted); one ray, or a bad fit: intersect it with the plane at cube-centre height.
When the overhead camera is blocked (an arm over the cube), the wrist cameras usually still see it, and vice versa.

    loc = locate([ray_from_pixel(cam, u, v, model, data) for cam, (u, v) in detections.items()])
Sim: SimCubeSensor renders nothing; it casts rays from each camera to points on the cube's visible faces, keeps the
points nothing blocks, and projects them: a partly hidden cube shifts the centroid exactly like a real red blob does.
"""
from __future__ import annotations

import mujoco
import numpy as np

CAMS = ("overhead", "left_wrist", "right_wrist")
CUBE_HALF = 0.016


# ------------------------------------------------------------------ real images
def detect_red(rgb, min_px=30):
    """Red-blob centroid (u, v) and pixel count. Brightness-independent: a pixel is "cube red" when green and blue
    are well below red RELATIVE to its brightness (the real cube is a dark maroon, ~(108, 46, 61) under the room
    light; the old rule R > 120 caught only ~5 % of its pixels). Only the largest connected blob counts, so red
    servo wires or specks do not pull the centroid."""
    import scipy.ndimage as ndi
    a = rgb.astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r > 55) & (r - g > 0.3 * r) & (r - b > 0.2 * r)
    mask = ndi.binary_opening(mask, iterations=1)
    lab, n = ndi.label(mask)
    if n == 0:
        return None, 0
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    npx = int(sizes[k - 1])
    if npx < min_px:
        return None, npx
    vs, us = np.nonzero(lab == k)
    return (float(us.mean()), float(vs.mean())), npx


# ------------------------------------------------------------------ geometry
def cam_intrinsics(spec):
    """(fx, fy, cx, cy, k1, k2, p1, p2, k3) from a pi_sim CameraSpec (or a rig-json camera entry): the full OpenCV
    Brown-Conrady model, as pi_sim renders with (cameras.py, distortion_opencv_k1_k2_p1_p2_k3). Missing terms = 0."""
    if isinstance(spec, dict):
        i = spec["intrinsics"]
        fx, fy, cx, cy, dist = i["fx"], i["fy"], i["cx"], i["cy"], spec["distortion"]
    else:
        K = spec.K
        fx, fy, cx, cy, dist = K[0, 0], K[1, 1], K[0, 2], K[1, 2], spec.distortion
    dist = list(dist) + [0.0] * (5 - len(dist))
    return (fx, fy, cx, cy, *[float(v) for v in dist[:5]])


def _intr(intr):
    """Accept the old 6-tuple (fx, fy, cx, cy, k1, k2) and the full 9-tuple."""
    return tuple(intr) + (0.0,) * (9 - len(intr))


def _distort(xn, yn, k1, k2, p1, p2, k3):
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    return (xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn),
            yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn)


def undistort(intr, u, v):
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = _intr(intr)
    xd, yd = (u - cx) / fx, (v - cy) / fy
    xn, yn = xd, yd
    for _ in range(30):
        r2 = xn * xn + yn * yn
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        dx = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        dy = p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        xn, yn = (xd - dx) / radial, (yd - dy) / radial
    return xn, yn


def project(intr, cam_pos, cam_R, p):
    """World point -> distorted pixel (u, v), or None if behind the camera."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = _intr(intr)
    c = cam_R.T @ (np.asarray(p, float) - cam_pos)
    if c[2] > -1e-4:
        return None
    xn, yn = c[0] / -c[2], -c[1] / -c[2]
    xd, yd = _distort(xn, yn, k1, k2, p1, p2, k3)
    return cx + fx * xd, cy + fy * yd


def ray(intr, cam_pos, cam_R, u, v):
    xn, yn = undistort(intr, u, v)
    d = cam_R @ np.array([xn, -yn, -1.0])
    return np.asarray(cam_pos, float).copy(), d / np.linalg.norm(d)


def camera_pose(model, data, cam):
    cid = model.camera(cam).id
    return data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3).copy()


def _triangulate(rays):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    if np.linalg.cond(A) >= 1e4:
        return None, np.inf
    x = np.linalg.solve(A, b)
    res = max(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (x - o)) for o, d in rays)
    if not all((x - o) @ d > 0 for o, d in rays) or not -0.01 < x[2] < 0.3:
        return None, np.inf
    return x, res


def locate(rays, z_plane=CUBE_HALF, max_residual=0.01, prefer=0):
    """rays: list of (origin, unit dir). Returns (xyz, how) or (None, 'none'). If all rays do not agree (one camera
    sees a partly hidden cube and its centroid is shifted), the most consistent PAIR is used instead."""
    if not rays:
        return None, "none"
    if len(rays) >= 2:
        x, res = _triangulate(rays)
        if x is not None and res < max_residual:
            return x, f"triangulated ({len(rays)} cameras)"
        if len(rays) == 3:
            best = min((_triangulate([rays[i], rays[j]]) + ((i, j),) for i, j in ((0, 1), (0, 2), (1, 2))),
                       key=lambda t: t[1])
            if best[0] is not None and best[1] < max_residual:
                return best[0], f"triangulated (2 of 3 cameras {best[2]})"
    o, d = rays[min(prefer, len(rays) - 1)]
    if abs(d[2]) < 1e-6:
        return None, "none"
    t = (z_plane - o[2]) / d[2]
    return (o + t * d if t > 0 else None), "single camera, table plane"


# ------------------------------------------------------------------ sim model of the detectors
class SimCubeSensor:
    """What the three real red-blob detectors would report, computed from the sim scene: sample points on the cube's
    faces that face each camera, drop points outside the image or blocked by anything (mj_ray), and take the centroid
    of the visible points' pixels (+ pixel noise). Partly hidden -> shifted centroid; mostly hidden -> not detected."""

    def __init__(self, model, data, camera_specs, rng, n=4, px_noise=1.5, min_visible=0.2):
        self.m, self.d, self.rng = model, data, rng
        self.intr = {c: cam_intrinsics(camera_specs[c]) for c in CAMS}
        self.size = {c: (camera_specs[c].width, camera_specs[c].height) for c in CAMS}
        self.cube_body = model.body("cube").id
        self.cam_body = {c: model.cam_bodyid[model.camera(c).id] for c in CAMS}
        g = (np.arange(n) + 0.5) / n * 2 - 1
        A, B = np.meshgrid(g, g)
        self.faces = []
        for ax in range(3):
            a, b2 = [i for i in range(3) if i != ax]
            for s in (-1, 1):
                P = np.zeros((n * n, 3))
                P[:, ax], P[:, a], P[:, b2] = s, A.ravel(), B.ravel()
                nrm = np.zeros(3)
                nrm[ax] = s
                self.faces.append((P * CUBE_HALF, nrm))
        self.px_noise, self.min_visible = px_noise, min_visible
        self._geomid = np.zeros(1, np.int32)
        self._visible = np.array([1, 1, 1, 0, 0, 0], np.uint8)

    def detections(self):
        m, d = self.m, self.d
        R_c = d.xmat[self.cube_body].reshape(3, 3)
        p_c = d.xpos[self.cube_body]
        out = {}
        for cam in CAMS:
            cpos, cR = camera_pose(m, d, cam)
            W, H = self.size[cam]
            pix, facing = [], 0
            for P, nrm in self.faces:
                nw = R_c @ nrm
                for q in P:
                    w = p_c + R_c @ q
                    if (cpos - w) @ nw <= 0:
                        continue
                    facing += 1
                    uv = project(self.intr[cam], cpos, cR, w)
                    if uv is None or not (0 <= uv[0] < W and 0 <= uv[1] < H):
                        continue
                    vec = w - cpos
                    dist = np.linalg.norm(vec)
                    # only VISIBLE geometry blocks a camera (groups 0-2); collision-only shapes (group 3: the safety
                    # guard's camera-post boxes, the gripper's CoACD pieces) are invisible
                    hit = mujoco.mj_ray(m, d, cpos, vec / dist, self._visible, 1, self.cam_body[cam], self._geomid)
                    if self._geomid[0] >= 0 and m.geom_bodyid[self._geomid[0]] == self.cube_body and hit <= dist + 1e-3:
                        pix.append(uv)
            if facing and len(pix) >= max(3, self.min_visible * facing):
                u, v = np.mean(pix, axis=0) + self.rng.normal(0, self.px_noise, 2)
                out[cam] = (float(u), float(v), len(pix) / facing)
        return out

    def locate(self):
        dets = self.detections()
        rays = [ray(self.intr[c], *camera_pose(self.m, self.d, c), u, v) for c, (u, v, _) in dets.items()]
        prefer = list(dets).index("overhead") if "overhead" in dets else 0
        xyz, how = locate(rays, prefer=prefer)
        return xyz, how, dets


def overhead_equivalent(xyz, overhead_intr, overhead_pos, overhead_R, z_plane=CUBE_HALF):
    """What the overhead red-blob detector would report for a cube centre at xyz: project through the overhead camera,
    back-project onto the plane at cube-centre height (a lifted cube reads off by the parallax, as on the rig).
    Lets a policy trained on overhead-detector readings use the triangulated position (which survives the overhead
    view being blocked)."""
    uv = project(overhead_intr, overhead_pos, overhead_R, xyz)
    if uv is None:
        return None
    o, d = ray(overhead_intr, overhead_pos, overhead_R, *uv)
    t = (z_plane - o[2]) / d[2]
    return (o + t * d)[:2]

