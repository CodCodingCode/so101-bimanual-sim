"""Seed both cameras by PnP: tool-centre point of the posed twin (true SO-101 geometry, zero offsets) <-> the wooden
cube's image centroid while the cube is held in the jaws (cube_track.py), EVEN episodes only (odd = held-out).

    .venv/bin/python so101_twin/stack2/seed_pnp.py [--pan-sign 1] [--lag -1.5] [--write]

Unknowns per camera: position, rotation, fovy, and the cube-centre offset along the gripper axis (shared). The top
camera gets one pose per episode group (scene_stack.top_group) with a shared fovy. Robust (soft-L1) least squares from
many random starts on a shell around the workspace. --write stores out/seed_cams.json (read by scene_stack.py).
"""
import argparse
import json
import math
import pathlib
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

W, H = 640, 480


def rotvec_to_mat(v):
    q = F.A.quat_from_rotvec(v)
    return F.A.mat_from_quat(q)


def project(pos, R, fovy, X):
    """MuJoCo camera (looks along -z, +y up): world points X [n,3] -> pixels [n,2]."""
    f = 0.5 * H / math.tan(math.radians(fovy) / 2)
    Xc = (X - pos) @ R          # columns of R = camera axes in world
    z = -Xc[:, 2]
    u = W / 2 + f * Xc[:, 0] / np.maximum(z, 1e-4)
    v = H / 2 - f * Xc[:, 1] / np.maximum(z, 1e-4)
    return np.stack([u, v], 1), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pan-sign", type=float, default=1.0)
    ap.add_argument("--lag", type=float, default=-1.5)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--starts", type=int, default=400)
    ap.add_argument("--one-top", action="store_true", help="one top pose for all episode groups")
    a = ap.parse_args()
    sc = F.load_scene("stack")
    P = F.init_params(sc)
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    fr = F.fullrate(sc)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    where = sc.cube_where(fs)
    trk = np.load(HERE / "out" / "cube_track.npz")
    m, d = rig.arm.m, rig.arm.d
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    result = {}
    rng = np.random.default_rng(0)
    for cam in ("top", "side"):
        grp = F.frame_groups(sc, cam)
        if a.one_top:
            grp = np.array(["all"] * fs.n)
        t = trk[cam]
        sel = [i for i in range(fs.n) if where[i] == "jaws" and ep[i] % 2 == 0 and np.isfinite(t[i, 0])
               and t[i, 3] > 2 and t[i, 4] > 2 and t[i, 5] < W - 3 and t[i, 6] < H - 3]
        X, Z, uv, g = [], [], [], []
        for i in sel:
            s = fr.state_at(i, a.lag).copy()
            s[0] *= a.pan_sign
            rig.arm.set_state(s)
            X.append(d.site_xpos[sid].copy())
            Z.append(d.site_xmat[sid].reshape(3, 3)[:, 2].copy())   # gripper-frame z axis in world
            uv.append(t[i, :2])
            g.append(grp[i])
        X, Z, uv, g = map(np.asarray, (X, Z, uv, g))
        keys = sorted(set(g))
        gi = np.array([keys.index(x) for x in g])
        print(f"{cam}: {len(sel)} held-cube frames (even episodes), groups {dict(zip(keys, np.bincount(gi).tolist()))}")
        ctr = X.mean(0)

        def unpack(p):
            fovy, off = p[0], p[1]
            poses = [(p[2 + 6 * k:5 + 6 * k], rotvec_to_mat(p[5 + 6 * k:8 + 6 * k])) for k in range(len(keys))]
            return fovy, off, poses

        def resid(p):
            fovy, off, poses = unpack(p)
            r = np.zeros((len(X), 2))
            Xo = X + off * Z
            for k, (pos, R) in enumerate(poses):
                mk = gi == k
                if mk.any():
                    pr, z = project(pos, R, fovy, Xo[mk])
                    r[mk] = (pr - uv[mk]) * np.where(z[:, None] > 0.02, 1.0, 50.0)
            zlo, zhi = (0.2, 1.2) if cam == "top" else (0.0, 0.45)
            pen = [max(0.0, zlo - pp[0][2]) * 2000 + max(0.0, pp[0][2] - zhi) * 2000 for pp in poses]
            r = np.concatenate([r, np.asarray(pen)[:, None].repeat(2, 1)]) if True else r
            # tie the groups together weakly (the camera moves little between groups)
            tie = [0.5 * (poses[k][0] - poses[0][0]) / 0.02 for k in range(1, len(keys))]
            return np.concatenate([r.ravel()] + [np.asarray(x) for x in tie]) if tie else r.ravel()

        best = None
        for s in range(a.starts):
            # random camera on a shell around the workspace, looking at the points' centroid
            if cam == "top":
                el = rng.uniform(30, 89)
                dist = rng.uniform(0.25, 0.9)
            else:
                el = rng.uniform(-5, 35)
                dist = rng.uniform(0.15, 0.7)
            az = rng.uniform(-180, 180)
            pos = ctr + dist * np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)),
                                         math.cos(math.radians(el)) * math.sin(math.radians(az)),
                                         math.sin(math.radians(el))])
            q = F.A.look_quat(pos, ctr + rng.normal(0, 0.03, 3), roll_deg=rng.uniform(-180, 180))
            R0 = F.A.mat_from_quat(q)
            rv = _rotvec(R0)
            fovy0 = rng.uniform(35, 75)
            p0 = [fovy0, 0.01] + list(np.concatenate([pos, rv])) * len(keys)
            lb = [30, -0.01] + [-np.inf] * 6 * len(keys)
            ub = [95, 0.04] + [np.inf] * 6 * len(keys)
            try:
                r = least_squares(resid, p0, loss="soft_l1", f_scale=15.0, bounds=(lb, ub), max_nfev=300)
            except Exception:
                continue
            rr = resid(r.x)[:2 * len(X)].reshape(-1, 2)
            if resid(r.x)[2 * len(X):2 * len(X) + 2 * len(keys)].max() > 0:
                continue
            med = float(np.median(np.linalg.norm(rr, axis=1)))
            if best is None or med < best[0]:
                best = (med, r.x)
                print(f"  start {s}: median reproj {med:.1f} px  fovy {r.x[0]:.1f} off {r.x[1]:.3f}", flush=True)
        med, p = best
        fovy, off, poses = unpack(p)
        rr = resid(p)[:2 * len(X)].reshape(-1, 2)
        err = np.linalg.norm(rr, axis=1)
        print(f"{cam}: BEST median {np.median(err):.1f} px, p90 {np.percentile(err, 90):.1f}, fovy {fovy:.2f}, cube offset {off:.3f}")
        for k, key in enumerate(keys):
            pos, R = poses[k]
            print(f"   {key}: pos {np.round(pos, 4).tolist()} median err {np.median(err[gi == k]):.1f} px")
        result[cam] = dict(fovy=float(fovy), cube_offset=float(off), median_px=float(np.median(err)),
                           groups={key: dict(pos=list(map(float, poses[k][0])),
                                             quat=list(map(float, F.A.quat_from_mat(poses[k][1]))))
                                   for k, key in enumerate(keys)})
    if a.write:
        cams = {}
        for cam, r in result.items():
            g0 = next(iter(r["groups"].values()))
            cams[cam] = dict(pos=g0["pos"], quat=g0["quat"], fovy=r["fovy"],
                             groups=r["groups"] if cam == "top" else None)
        (HERE / "out" / "seed_cams.json").write_text(json.dumps(dict(method="PnP tcp<->held cube centroid, even episodes",
                                                                       lag=a.lag, pan_sign=a.pan_sign, cams=cams,
                                                                       detail=result), indent=1))
        print("wrote", HERE / "out" / "seed_cams.json")


def _rotvec(R):
    q = F.A.quat_from_mat(R)
    w = np.clip(q[0], -1, 1)
    ang = 2 * math.acos(w)
    s = math.sqrt(max(1 - w * w, 1e-12))
    return q[1:] / s * ang if ang > 1e-9 else np.zeros(3)


if __name__ == "__main__":
    main()
