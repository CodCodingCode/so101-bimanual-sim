"""Seed cameras + joint map from the held wooden cube at full rate (EVEN episodes only): bundle adjustment.

    .venv/bin/python so101_twin/stack2/bundle.py [--free-joints] [--write] [--groups]

Observations: cube centroid in top / side for every full-rate frame of the even episodes where the gripper holds the
cube (gripper 5..18) (fullrate_cube.py). Model: cube centre = tcp(state at t + lag) + a * gripper axis, projected by a
pinhole camera (MuJoCo convention). Stage 1: joint map nominal (official new_calib, zero offsets): random-start PnP per
camera. Stage 2 (--free-joints): + offsets / scales of lift, elbow, wrist_flex, the cube offset a and the lag.
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
from seed_pnp import project, rotvec_to_mat, _rotvec  # noqa: E402

W, H = 640, 480


def load_obs(sc, max_per_cam=400, seed=0):
    z = np.load(HERE / "out" / "fullrate_cube.npz")
    rng = np.random.default_rng(seed)
    obs = {}
    for cam in ("top", "side"):
        rows = []
        for k in z.files:
            if not k.startswith(cam + "_"):
                continue
            e = int(k.split("_")[1])
            arr = z[k]
            st = z[f"state_{e}"]
            for t in range(len(arr)):
                g = st[t, 5]
                a = arr[t]
                if not (5 <= g <= 18) or not np.isfinite(a[0]):
                    continue
                if a[3] <= 2 or a[4] <= 2 or a[5] >= W - 3 or a[6] >= H - 3:
                    continue
                rows.append((e, t, a[0], a[1]))
        rows = np.array(rows)
        if len(rows) > max_per_cam:
            rows = rows[np.sort(rng.choice(len(rows), max_per_cam, replace=False))]
        obs[cam] = rows
    states = {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("state_")}
    return obs, states


def state_at(states, e, t, lag):
    s = states[int(e)]
    x = np.clip(t + lag, 0, len(s) - 1)
    a = int(np.floor(x))
    b = min(a + 1, len(s) - 1)
    w = x - a
    return (1 - w) * s[a].astype(float) + w * s[b].astype(float)


class FK:
    def __init__(self, sc):
        P = F.init_params(sc)
        self.rig = F.Rig(sc, P)
        self.m, self.d = self.rig.arm.m, self.rig.arm.d
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    def __call__(self, states, rows, lag, off, scl):
        X, Z = [], []
        self.rig.arm.offsets_deg = np.asarray(off, float)
        self.rig.arm.scales = np.asarray(scl, float)
        for e, t, _, _ in rows:
            self.rig.arm.set_state(state_at(states, e, t, lag))
            X.append(self.d.site_xpos[self.sid].copy())
            Z.append(self.d.site_xmat[self.sid].reshape(3, 3)[:, 2].copy())
        return np.array(X), np.array(Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-joints", action="store_true")
    ap.add_argument("--starts", type=int, default=300)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--lag", type=float, default=-1.5)
    a = ap.parse_args()
    sc = F.load_scene("stack")
    obs, states = load_obs(sc)
    fk = FK(sc)
    rng = np.random.default_rng(1)
    cams = {}
    for cam in ("top", "side"):
        rows = obs[cam]
        X, Z = fk(states, rows, a.lag, np.zeros(5), np.ones(5))
        uv = rows[:, 2:4]
        ctr = X.mean(0)
        print(f"{cam}: {len(rows)} held-cube observations from {len(set(rows[:, 0]))} even episodes; 3D spread "
              f"{np.round(X.std(0), 3).tolist()}", flush=True)

        def resid(p, X=X, Z=Z, uv=uv, cam=cam):
            pos, R, fovy, off = p[:3], rotvec_to_mat(p[3:6]), p[6], p[7]
            pr, zz = project(pos, R, fovy, X + off * Z)
            r = (pr - uv) * np.where(zz[:, None] > 0.03, 1.0, 30.0)
            zlo, zhi = (0.15, 1.5) if cam == "top" else (-0.02, 0.5)
            pen = 3000 * (max(0.0, zlo - pos[2]) + max(0.0, pos[2] - zhi))
            return np.concatenate([r.ravel(), [pen]])

        best = None
        for s in range(a.starts):
            if cam == "top":
                el, dist = rng.uniform(25, 89), rng.uniform(0.25, 1.0)
            else:
                el, dist = rng.uniform(-10, 40), rng.uniform(0.12, 0.8)
            az = rng.uniform(-180, 180)
            pos = ctr + dist * np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)),
                                         math.cos(math.radians(el)) * math.sin(math.radians(az)),
                                         math.sin(math.radians(el))])
            q = F.A.look_quat(pos, ctr + rng.normal(0, 0.04, 3), roll_deg=rng.uniform(-180, 180))
            p0 = np.concatenate([pos, _rotvec(F.A.mat_from_quat(q)), [rng.uniform(35, 85), 0.01]])
            try:
                r = least_squares(resid, p0, loss="soft_l1", f_scale=10.0, max_nfev=200,
                                  bounds=([-np.inf] * 6 + [25, -0.01], [np.inf] * 6 + [100, 0.04]))
            except Exception:
                continue
            rr = resid(r.x)
            if rr[-1] > 0:
                continue
            err = np.linalg.norm(rr[:-1].reshape(-1, 2), axis=1)
            med = float(np.median(err))
            if best is None or med < best[0]:
                best = (med, r.x)
                print(f"  start {s}: median {med:.1f} px p90 {np.percentile(err, 90):.1f} pos {np.round(r.x[:3], 3).tolist()} "
                      f"fovy {r.x[6]:.1f} a {r.x[7]:.3f}", flush=True)
        cams[cam] = best[1]
    if a.free_joints:
        # joint stage: both cameras + offsets (pan, lift, elbow, wrist) + scales (lift, elbow, wrist) + a + lag
        rows_all = {c: obs[c] for c in obs}

        def unpack(p):
            pt, ps = p[0:7], p[7:14]
            off = np.array([p[14], p[15], p[16], p[17], 0.0])
            scl = np.array([1.0, p[18], p[19], p[20], 1.0])
            return pt, ps, off, scl, p[21], p[22]

        def resid_all(p):
            pt, ps, off, scl, acube, lag = unpack(p)
            out = []
            for cam, pc in (("top", pt), ("side", ps)):
                X, Z = fk(states, rows_all[cam], lag, off, scl)
                pr, zz = project(pc[:3], rotvec_to_mat(pc[3:6]), pc[6], X + acube * Z)
                out.append(((pr - rows_all[cam][:, 2:4]) * np.where(zz[:, None] > 0.03, 1.0, 30.0)).ravel())
            out.append((off[:4] / 25.0) * 3)          # weak prior: offsets ~ 0 +- 25 deg
            out.append((scl[1:4] - 1.0) / 0.1 * 3)    # scales ~ 1 +- 0.1
            return np.concatenate(out)

        p0 = np.concatenate([cams["top"][:7], cams["side"][:7], [0, 0, 0, 0], [1, 1, 1],
                             [0.5 * (cams["top"][7] + cams["side"][7])], [a.lag]])
        lb = [-np.inf] * 6 + [25] + [-np.inf] * 6 + [25] + [-40] * 4 + [0.7] * 3 + [-0.01, -6]
        ub = [np.inf] * 6 + [100] + [np.inf] * 6 + [100] + [40] * 4 + [1.3] * 3 + [0.05, 4]
        r = least_squares(resid_all, p0, loss="soft_l1", f_scale=10.0, bounds=(lb, ub), max_nfev=400, verbose=0)
        pt, ps, off, scl, acube, lag = unpack(r.x)
        res = resid_all(r.x)
        nt = 2 * len(rows_all["top"])
        et = np.linalg.norm(res[:nt].reshape(-1, 2), axis=1)
        es = np.linalg.norm(res[nt:nt + 2 * len(rows_all["side"])].reshape(-1, 2), axis=1)
        print(f"JOINT: top median {np.median(et):.1f} px p90 {np.percentile(et, 90):.1f} | side median {np.median(es):.1f} "
              f"p90 {np.percentile(es, 90):.1f}")
        print(f"  offsets {np.round(off, 2).tolist()} scales {np.round(scl, 3).tolist()} cube a {acube:.3f} lag {lag:.2f}")
        print(f"  top pos {np.round(pt[:3], 3).tolist()} fovy {pt[6]:.1f} | side pos {np.round(ps[:3], 3).tolist()} fovy {ps[6]:.1f}")
        cams = {"top": np.concatenate([pt, [acube]]), "side": np.concatenate([ps, [acube]])}
        joint = dict(offsets=off.tolist(), scales=scl.tolist(), lag=float(lag), cube_a=float(acube))
    else:
        joint = dict(offsets=[0.0] * 5, scales=[1.0] * 5, lag=a.lag)
    if a.write:
        out = {}
        for cam, p in cams.items():
            out[cam] = dict(pos=list(map(float, p[:3])), quat=list(map(float, F.A.quat_from_mat(rotvec_to_mat(p[3:6])))),
                            fovy=float(p[6]))
        (HERE / "out" / "seed_cams.json").write_text(json.dumps(dict(method="bundle: tcp <-> held cube centroid at "
                                                                     "full rate, even episodes", joint=joint, cams=out), indent=1))
        print("wrote", HERE / "out" / "seed_cams.json")


if __name__ == "__main__":
    main()
