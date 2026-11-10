"""Joint-map hypothesis scan + bundle adjustment from the held wooden cube at full rate (EVEN episodes only).

    .venv/bin/python so101_twin/stack2/bundle2.py [--write]

1. grid over (lift, elbow, wrist_flex) offsets: for each hypothesis, random-start PnP of each camera (FK precomputed),
   score = median reprojection error (top + side), with physical bounds (top camera above the table, side camera low).
2. from the best hypotheses: joint least squares over both cameras + offsets (pan, lift, elbow, wrist) + scales
   (lift, elbow, wrist) + cube offset along the gripper axis + lag.
Writes out/seed_cams.json (cameras + joint map) with --write.
"""
import argparse
import itertools
import json
import math
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
from bundle import FK, load_obs  # noqa: E402
from seed_pnp import _rotvec, project, rotvec_to_mat  # noqa: E402

BOUNDS = {"top": (0.15, 1.5), "side": (-0.03, 0.45)}


def pnp(cam, X, Z, uv, rng, starts=40, fovy_rng=(30, 95)):
    ctr = X.mean(0)
    zlo, zhi = BOUNDS[cam]

    def resid(p):
        pos, R, fovy, off = p[:3], rotvec_to_mat(p[3:6]), p[6], p[7]
        pr, zz = project(pos, R, fovy, X + off * Z)
        r = (pr - uv) * np.where(zz[:, None] > 0.03, 1.0, 30.0)
        pen = 3000 * (max(0.0, zlo - pos[2]) + max(0.0, pos[2] - zhi))
        return np.concatenate([r.ravel(), [pen]])

    best = None
    for s in range(starts):
        if cam == "top":
            el, dist = rng.uniform(25, 89), rng.uniform(0.25, 1.0)
        else:
            el, dist = rng.uniform(-10, 40), rng.uniform(0.1, 0.7)
        az = rng.uniform(-180, 180)
        pos = ctr + dist * np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)),
                                     math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))])
        if not (zlo < pos[2] < zhi):
            continue
        q = F.A.look_quat(pos, ctr + rng.normal(0, 0.04, 3), roll_deg=rng.uniform(-180, 180))
        p0 = np.concatenate([pos, _rotvec(F.A.mat_from_quat(q)), [rng.uniform(*fovy_rng), 0.01]])
        try:
            r = least_squares(resid, p0, loss="soft_l1", f_scale=10.0, max_nfev=150,
                              bounds=([-np.inf] * 6 + [fovy_rng[0], -0.01], [np.inf] * 6 + [fovy_rng[1], 0.04]))
        except Exception:
            continue
        rr = resid(r.x)
        if rr[-1] > 0:
            continue
        med = float(np.median(np.linalg.norm(rr[:-1].reshape(-1, 2), axis=1)))
        if best is None or med < best[0]:
            best = (med, r.x)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--lag", type=float, default=-1.5)
    ap.add_argument("--n", type=int, default=250)
    a = ap.parse_args()
    sc = F.load_scene("stack")
    obs, states = load_obs(sc, max_per_cam=a.n)
    fk = FK(sc)
    rng = np.random.default_rng(0)
    t0 = time.time()
    results = []
    grid = list(itertools.product([-20, -10, 0, 10, 20], [-40, -25, -10, 0, 10], [-20, 0, 20, 35]))
    for lo, eo, wo in grid:
        off = np.array([0.0, lo, eo, wo, 0.0])
        meds, xs = {}, {}
        for cam in ("top", "side"):
            X, Z = fk(states, obs[cam], a.lag, off, np.ones(5))
            b = pnp(cam, X, Z, obs[cam][:, 2:4], rng, starts=15)
            meds[cam], xs[cam] = (b[0], b[1]) if b else (999.0, None)
        results.append((meds["top"] + meds["side"], off, meds, xs))
        print(f"[{time.time() - t0:.0f}s] off lift {lo:+d} elbow {eo:+d} wrist {wo:+d}: top {meds['top']:.1f} side "
              f"{meds['side']:.1f}", flush=True)
    results.sort(key=lambda r: r[0])
    print("best hypotheses:")
    for r in results[:6]:
        print(f"  {r[0]:.1f}  off {r[1].tolist()}  top {r[2]['top']:.1f} side {r[2]['side']:.1f}")
    # joint refinement from the best 3 hypotheses
    rows = {c: obs[c] for c in obs}

    def unpack(p):
        off = np.array([p[14], p[15], p[16], p[17], 0.0])
        scl = np.array([1.0, p[18], p[19], p[20], 1.0])
        return p[0:7], p[7:14], off, scl, p[21], p[22]

    def resid_all(p, off0):
        pt, ps, off, scl, acube, lag = unpack(p)
        out = []
        for cam, pc in (("top", pt), ("side", ps)):
            X, Z = fk(states, rows[cam], lag, off, scl)
            pr, zz = project(pc[:3], rotvec_to_mat(pc[3:6]), pc[6], X + acube * Z)
            out.append(((pr - rows[cam][:, 2:4]) * np.where(zz[:, None] > 0.03, 1.0, 30.0)).ravel())
            zlo, zhi = BOUNDS[cam]
            out.append(np.array([3000 * (max(0.0, zlo - pc[2]) + max(0.0, pc[2] - zhi))]))
        out.append((scl[1:4] - 1.0) / 0.1 * 2)
        return np.concatenate(out)

    best = None
    for r in results[:3]:
        if r[3]["top"] is None or r[3]["side"] is None:
            continue
        off0 = r[1]
        p0 = np.concatenate([r[3]["top"][:7], r[3]["side"][:7], off0[:4], [1, 1, 1],
                             [0.5 * (r[3]["top"][7] + r[3]["side"][7])], [a.lag]])
        lb = [-np.inf] * 6 + [25] + [-np.inf] * 6 + [25] + [-60] * 4 + [0.75] * 3 + [-0.01, -6]
        ub = [np.inf] * 6 + [100] + [np.inf] * 6 + [100] + [60] * 4 + [1.25] * 3 + [0.05, 4]
        sol = least_squares(resid_all, p0, args=(off0,), loss="soft_l1", f_scale=8.0, bounds=(lb, ub), max_nfev=300)
        res = resid_all(sol.x, off0)
        nt = 2 * len(rows["top"])
        et = np.linalg.norm(res[:nt].reshape(-1, 2), axis=1)
        es = np.linalg.norm(res[nt + 1:nt + 1 + 2 * len(rows["side"])].reshape(-1, 2), axis=1)
        pt, ps, off, scl, acube, lag = unpack(sol.x)
        tot = np.median(et) + np.median(es)
        print(f"JOINT from {off0.tolist()}: top median {np.median(et):.1f} p90 {np.percentile(et, 90):.1f} | side median "
              f"{np.median(es):.1f} p90 {np.percentile(es, 90):.1f}\n   offsets {np.round(off, 2).tolist()} scales "
              f"{np.round(scl, 3).tolist()} a {acube:.3f} lag {lag:.2f}\n   top pos {np.round(pt[:3], 3).tolist()} fovy "
              f"{pt[6]:.1f} | side pos {np.round(ps[:3], 3).tolist()} fovy {ps[6]:.1f}", flush=True)
        if best is None or tot < best[0]:
            best = (tot, sol.x)
    pt, ps, off, scl, acube, lag = unpack(best[1])
    # per-group residual of the top camera (does the top camera move relative to the arm?)
    X, Z = fk(states, rows["top"], lag, off, scl)
    pr, _ = project(pt[:3], rotvec_to_mat(pt[3:6]), pt[6], X + acube * Z)
    d = pr - rows["top"][:, 2:4]
    grp = np.array([sc.top_group(int(e)) for e in rows["top"][:, 0]])
    for g in sorted(set(grp)):
        mk = grp == g
        print(f"   top group {g}: n {mk.sum()} mean residual (u, v) {np.round(np.median(d[mk], 0), 1).tolist()}")
    if a.write:
        cams = {c: dict(pos=list(map(float, p[:3])), quat=list(map(float, F.A.quat_from_mat(rotvec_to_mat(p[3:6])))),
                        fovy=float(p[6])) for c, p in (("top", pt), ("side", ps))}
        (HERE / "out" / "seed_cams.json").write_text(json.dumps(dict(
            method="bundle2: joint-map grid + bundle adjustment, tcp <-> held cube centroid at full rate, even episodes",
            joint=dict(offsets=off.tolist(), scales=scl.tolist(), lag=float(lag), cube_a=float(acube)), cams=cams), indent=1))
        print("wrote", HERE / "out" / "seed_cams.json")


if __name__ == "__main__":
    main()
