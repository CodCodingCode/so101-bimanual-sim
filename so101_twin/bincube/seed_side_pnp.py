"""Seed for the side camera from 2D-3D correspondences (run before fit_cameras.py --cam side --local).

    .venv/bin/python so101_twin/bincube/seed_side_pnp.py      # prints the camera params, writes out/side_pnp.json + out/side_pnp.png

Correspondences: in every frame whose cube state is 'jaws' (replay.episode_states) the yellow-cube centroid in the
real side image corresponds to the sim TCP site position (forward kinematics with twin.real_to_sim), plus the arm
base plate (world origin) at a hand-picked pixel. A 7-parameter pinhole camera (pos, yaw, pitch, roll, fovy; the
same parameterisation as build_scene.CAMS) is fitted by least-squares reprojection error with random restarts.
"""
from __future__ import annotations

import json
import pathlib
import sys

import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
import place_objects  # noqa: E402
import replay  # noqa: E402
from build_scene import cam_xyaxes  # noqa: E402
from twin import HERE, Sim, real_set  # noqa: E402

BASE_PIX = (345.0, 172.0)          # base plate centre (world origin) in the real side image
KEYS = ("x", "y", "z", "yaw", "pitch", "roll", "fovy")


def project(p: np.ndarray, X: np.ndarray) -> np.ndarray:
    right, up = cam_xyaxes(p[3], p[4], p[5])
    right, up = np.asarray(right), np.asarray(up)
    back = np.cross(right, up)
    R = np.stack([right, up, back], axis=1)
    Xc = (X - p[:3]) @ R                       # camera coords
    fy = 240.0 / np.tan(np.radians(p[6]) / 2)
    z = -Xc[:, 2]
    z = np.where(z < 1e-3, 1e-3, z)
    return np.stack([320 + fy * Xc[:, 0] / z, 240 - fy * Xc[:, 1] / z], axis=1)


def main():
    fs = real_set()
    sim = Sim()
    sid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    lids, cubes = replay.all_states(fs)
    X, U, W = [np.array([0.0, 0.0, 0.0])], [np.array(BASE_PIX)], [4.0]
    for i in np.nonzero(cubes == "jaws")[0]:
        n, u, v = place_objects.yellow_centroid(fs.obs(int(i))["images"]["side"])
        if n < 300:
            continue
        sim.set_joints(fs.state[i])
        X.append(sim.d.site_xpos[sid].copy())
        U.append(np.array([u, v]))
        W.append(1.0)
    X, U, W = np.stack(X), np.stack(U), np.array(W)
    print(f"{len(X)} correspondences")

    def resid(p):
        return ((project(p, X) - U) * W[:, None]).ravel()

    lo = np.array([-1.2, 0.2, 0.1, -120, -70, -15, 30])
    hi = np.array([0.6, 1.4, 0.8, 30, -5, 15, 80])
    rng = np.random.default_rng(0)
    best = None
    for k in range(60):
        p0 = rng.uniform(lo, hi)
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=400)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    err = np.linalg.norm(project(p, X) - U, axis=1)
    print("params", dict(zip(KEYS, np.round(p, 4))))
    print(f"reprojection error px: mean {err.mean():.1f}, median {np.median(err):.1f}, max {err.max():.1f}, base {err[0]:.1f}")
    (HERE / "out" / "side_pnp.json").write_text(json.dumps(dict(params=dict(zip(KEYS, map(float, p))), n=int(len(X)),
                                                              err_mean_px=float(err.mean()))))
    sim.set_camera("side", p[:3], p[3], p[4], p[5], p[6])
    rows = []
    for i in (0, 36, 100, 220):
        sim.set_joints(fs.state[i])
        rows.append(np.concatenate([fs.obs(i)["images"]["side"], sim.render("side")], 1))
    Image.fromarray(np.concatenate(rows, 0)).resize((1280, 480 * 2)).save(HERE / "out" / "side_pnp.png")
    sim.close()


if __name__ == "__main__":
    main()
