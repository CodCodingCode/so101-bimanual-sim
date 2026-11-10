"""Initialise both camera poses (and the disc / cube / green-block positions) from hand-read keypoints.

    .venv/bin/python so101_twin/stack/init_cameras.py      # prints CAMS / OBJECTS lines, writes out/init_cameras.json

The silhouette fit (fit_cameras.py) needs a starting pose within ~0.2 m / 20 deg of the truth; the white arm on
the light table gives too weak a mask signal for a wide random search. So: pixel positions of the held cube
(centre ~ the `tcp` site on Fixed_Jaw), the fingertips and the three static objects were read off the gridded
real frames (out/grid_<cam>_<frame>.png) by eye (+-10 px) and matched to the sim's 3D positions at the frame's
joint state. Unknowns: 7 per camera (x y z yaw pitch roll fovy) + (x, y) of the disc, cube-start and green block.
Solved by scipy least_squares (pinhole model, principal point at the image centre) from several random starts.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
from build_scene import cam_xyaxes  # noqa: E402
from twin import HERE, Sim, real_set  # noqa: E402

W, H = 640, 480
# frame -> (u, v) of the held cube centre (tcp) / fingertips in each camera
TCP_PIX = {
    "top": {3: (325, 182), 203: (350, 160), 275: (375, 190), 26: (335, 300), 147: (300, 255)},
    "side": {3: (390, 60), 203: (440, 130), 275: (430, 60), 26: (300, 140), 147: (230, 230)},
}
# static objects: pixel centre per camera and the centre height
OBJ_PIX = {
    "disc": dict(top=(318, 240), side=(190, 290), z=0.002),
    "cube": dict(top=(320, 285), side=(250, 245), z=build_scene.OBJECTS["cube"]["size"][2]),
    "green": dict(top=(262, 150), side=(415, 315), z=build_scene.OBJECTS["green"]["size"][2]),
}
OBJ_NAMES = ("disc", "cube", "green")
# priors (initial guesses): see build_scene.py docstring for the frame
P0 = {"top": [-0.25, 0.05, 0.75, 0.0, -70.0, 0.0, 50.0], "side": [0.10, 0.60, 0.06, -90.0, -8.0, 0.0, 50.0]}
SPREAD = np.array([0.15, 0.15, 0.15, 30, 15, 15, 12])


def project(p: np.ndarray, X: np.ndarray) -> np.ndarray:
    """camera params (7,) and points (n,3) -> pixels (n,2)."""
    right, up = cam_xyaxes(p[3], p[4], p[5])
    right, up = np.array(right), np.array(up)
    fwd = np.cross(up, right)
    d = X - p[:3]
    xc, yc, zc = d @ right, d @ up, d @ fwd
    fy = (H / 2) / math.tan(math.radians(p[6]) / 2)
    zc = np.maximum(zc, 1e-3)
    return np.stack([W / 2 + fy * xc / zc, H / 2 - fy * yc / zc], 1)


def main():
    fs = real_set()
    sim = Sim()
    sid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    tcp = {}
    for cam in TCP_PIX:
        for i in TCP_PIX[cam]:
            if i not in tcp:
                sim.set_joints(fs.state[i])
                tcp[i] = sim.d.site_xpos[sid].copy()
    sim.close()
    cams = ("top", "side")

    def unpack(x):
        c = {cam: x[7 * k:7 * k + 7] for k, cam in enumerate(cams)}
        obj = {n: x[14 + 2 * k:16 + 2 * k] for k, n in enumerate(OBJ_NAMES)}
        return c, obj

    def resid(x):
        c, obj = unpack(x)
        out = []
        for cam in cams:
            X = np.array([tcp[i] for i in TCP_PIX[cam]])
            uv = np.array([TCP_PIX[cam][i] for i in TCP_PIX[cam]], float)
            out.append((project(c[cam], X) - uv).ravel())
            Xo = np.array([[*obj[n], OBJ_PIX[n]["z"]] for n in OBJ_NAMES])
            uvo = np.array([OBJ_PIX[n][cam] for n in OBJ_NAMES], float)
            out.append((project(c[cam], Xo) - uvo).ravel())
        # weak prior keeping the cameras above the table and fovy sane
        out.append(np.array([max(0.0, 0.02 - c["top"][2]) * 1e3, max(0.0, 0.0 - c["side"][2]) * 1e3]))
        return np.concatenate(out)

    rng = np.random.default_rng(0)
    best = None
    lo = np.array([-1.5, -1.5, 0.02, -360, -89, -90, 25] * 2 + [-0.6, -0.6] * 3)
    hi = np.array([1.5, 1.5, 1.5, 360, 0, 90, 100] * 2 + [0.6, 0.6] * 3)
    for k in range(24):
        x0 = np.concatenate([np.array(P0[c]) + (rng.uniform(-1, 1, 7) * SPREAD if k else 0) for c in cams] +
                            [[0.0, 0.2], [0.0, 0.22], [-0.07, 0.24]])
        x0 = np.clip(x0, lo + 1e-3, hi - 1e-3)
        try:
            r = least_squares(resid, x0, method="trf", bounds=(lo, hi), max_nfev=600)
        except Exception as e:
            print("fail", e)
            continue
        rms = math.sqrt(np.mean(r.fun[:-2] ** 2))
        if best is None or rms < best[0]:
            best = (rms, r.x)
    rms, x = best
    c, obj = unpack(x)
    print(f"best rms reprojection {rms:.1f} px")
    for cam in cams:
        p = c[cam]
        print(f'    "{cam}": dict(pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}), yaw={p[3]:.2f}, pitch={p[4]:.2f}, '
              f'roll={p[5]:.2f}, fovy={p[6]:.2f}),')
        X = np.array([tcp[i] for i in TCP_PIX[cam]]); uv = np.array([TCP_PIX[cam][i] for i in TCP_PIX[cam]], float)
        print(f"      tcp residuals px:", np.round(np.linalg.norm(project(p, X) - uv, axis=1), 1))
        Xo = np.array([[*obj[n], OBJ_PIX[n]["z"]] for n in OBJ_NAMES]); uvo = np.array([OBJ_PIX[n][cam] for n in OBJ_NAMES], float)
        print(f"      obj residuals px:", np.round(np.linalg.norm(project(p, Xo) - uvo, axis=1), 1))
    for n in OBJ_NAMES:
        print(f"    {n}: xy=({obj[n][0]:.3f}, {obj[n][1]:.3f})")
    (HERE / "out" / "init_cameras.json").write_text(json.dumps(dict(
        rms_px=rms, cams={cam: dict(zip(("x", "y", "z", "yaw", "pitch", "roll", "fovy"), map(float, c[cam]))) for cam in cams},
        objects={n: [float(v) for v in obj[n]] for n in OBJ_NAMES}), indent=1))


if __name__ == "__main__":
    main()
