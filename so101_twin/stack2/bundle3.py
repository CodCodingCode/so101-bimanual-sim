"""Bundle adjustment of both cameras AND the joint map on the held wooden cube at full rate (EVEN episodes only).

    .venv/bin/python so101_twin/stack2/bundle3.py [--write]

Unknowns: top camera = one pose per session + shared fovy (init: top_seed.json); side camera pose + fovy (random
starts: low camera beyond the green block looking back at the disc, as the side images show: the disc is a thin
ellipse, the cube on the block is 2.5x larger than on the disc); joint offsets (pan, lift, elbow, wrist_flex) and
scales (lift, elbow, wrist_flex) starting from the builder's map; cube-centre offset along the gripper axis; latency.
Observations: top / side cube centroids while the gripper holds the cube, plus two physical priors: the cube centre is
2 cm above the table at the grasp (gripper closing on the disc) -- the arm base stands on the table, so tcp z ~ 0.02.
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--n", type=int, default=160)
    a = ap.parse_args()
    sc = F.load_scene("stack")
    obs, states = load_obs(sc, max_per_cam=100000)
    fk = FK(sc)
    ts = json.loads((HERE / "out" / "top_seed.json").read_text())
    keys = sorted(ts["groups"])
    rng = np.random.default_rng(0)
    rows_t = obs["top"]
    gt = np.array([sc.top_group(int(e)) for e in rows_t[:, 0]])
    pick = []
    for g in keys:
        ii = np.nonzero(gt == g)[0]
        pick += list(rng.choice(ii, min(len(ii), a.n // 2 if g != "g2_e105-126" else 40), replace=False))
    rows_t = rows_t[np.sort(pick)]
    gi_t = np.array([keys.index(sc.top_group(int(e))) for e in rows_t[:, 0]])
    rows_s = obs["side"][np.sort(rng.choice(len(obs["side"]), min(a.n * 2, len(obs["side"])), replace=False))]
    # grasp prior: first frame of each even episode where the gripper closes below 18 after being open
    grasp = []
    for e, st in states.items():
        g = st[:, 5]
        for t in range(1, len(g)):
            if g[t - 1] > 18 and g[t] <= 18 and st[t, 1] > -60:
                grasp.append((e, t + 8, 0, 0))   # a few frames later: jaws closed on the cube, still on the disc
                break
    grasp = np.array(grasp, float)
    print(f"top obs {len(rows_t)}, side obs {len(rows_s)}, grasp priors {len(grasp)}", flush=True)
    nk = len(keys)

    def unpack(p):
        fovy_t, k = p[0], 1
        poses_t = []
        for _ in range(nk):
            poses_t.append((p[k:k + 3], p[k + 3:k + 6]))
            k += 6
        ps = p[k:k + 7]
        k += 7
        off = np.array([p[k], p[k + 1], p[k + 2], p[k + 3], 0.0])
        scl = np.array([1.0, p[k + 4], p[k + 5], p[k + 6], 1.0])
        acube, lag = p[k + 7], p[k + 8]
        return fovy_t, poses_t, ps, off, scl, acube, lag

    def resid(p):
        fovy_t, poses_t, ps, off, scl, acube, lag = unpack(p)
        out = []
        X, Z = fk(states, rows_t, lag, off, scl)
        r = np.zeros((len(X), 2))
        for k in range(nk):
            mk = gi_t == k
            pr, zz = project(poses_t[k][0], rotvec_to_mat(poses_t[k][1]), fovy_t, X[mk] + acube * Z[mk])
            r[mk] = (pr - rows_t[mk, 2:4]) * np.where(zz[:, None] > 0.03, 1.0, 30.0)
        out.append(r.ravel())
        X, Z = fk(states, rows_s, lag, off, scl)
        pr, zz = project(ps[:3], rotvec_to_mat(ps[3:6]), ps[6], X + acube * Z)
        out.append(((pr - rows_s[:, 2:4]) * np.where(zz[:, None] > 0.03, 1.0, 30.0)).ravel())
        Xg, Zg = fk(states, grasp, lag, off, scl)
        out.append(((Xg + acube * Zg)[:, 2] - 0.02) / 0.005)            # cube centre 2 cm above the table at the grasp
        out.append(np.array([300 * max(0.0, ps[2] - 0.15), 300 * max(0.0, -0.03 - ps[2])]))   # side camera is low
        for k in range(1, nk):
            out.append(0.3 * (poses_t[k][0] - poses_t[0][0]) / 0.01)
        out.append((scl[1:4] - 1.0) / 0.08)
        return np.concatenate(out)

    p_top = [ts["fovy"]]
    for g in keys:
        q = ts["groups"][g]
        p_top += list(q["pos"]) + list(_rotvec(F.A.mat_from_quat(q["quat"])))
    joint0 = list(ts["offsets"][:4]) + list(ts["scales"][1:4]) + [-0.01, -1.5]
    best = None
    t0 = time.time()
    for s in range(a.starts):
        # side camera: beyond the block (from the disc), low, looking back toward the disc
        d = rng.uniform(0.08, 0.3)
        lat = rng.uniform(-0.15, 0.15)
        blk, dsc = np.array([0.156, 0.184]), np.array([-0.055, 0.228])
        u = (blk - dsc) / np.linalg.norm(blk - dsc)
        v = np.array([-u[1], u[0]])
        xy = blk + d * u + lat * v
        pos = np.array([xy[0], xy[1], rng.uniform(0.0, 0.1)])
        tgt = np.concatenate([0.5 * (blk + dsc) + rng.normal(0, 0.05, 2), [rng.uniform(0.0, 0.08)]])
        q = F.A.look_quat(pos, tgt, roll_deg=rng.normal(0, 8))
        p0 = np.array(p_top + list(pos) + list(_rotvec(F.A.mat_from_quat(q))) + [rng.uniform(45, 80)] + joint0)
        lb = [30] + [-np.inf] * 6 * nk + [-np.inf] * 6 + [30] + [-45] * 4 + [0.75] * 3 + [-0.02, -5]
        ub = [90] + [np.inf] * 6 * nk + [np.inf] * 6 + [100] + [45] * 4 + [1.25] * 3 + [0.05, 3]
        try:
            r = least_squares(resid, p0, loss="soft_l1", f_scale=8.0, bounds=(lb, ub), max_nfev=100)
        except Exception as ex:
            print("fail", ex)
            continue
        rr = resid(r.x)
        nt, ns = 2 * len(rows_t), 2 * len(rows_s)
        et = np.linalg.norm(rr[:nt].reshape(-1, 2), axis=1)
        es = np.linalg.norm(rr[nt:nt + ns].reshape(-1, 2), axis=1)
        tot = np.median(et) + np.median(es)
        fovy_t, poses_t, ps, off, scl, acube, lag = unpack(r.x)
        print(f"[{time.time() - t0:.0f}s] start {s}: top {np.median(et):.1f} side {np.median(es):.1f} px | side pos "
              f"{np.round(ps[:3], 3).tolist()} fovy {ps[6]:.1f} | off {np.round(off[:4], 1).tolist()} scl "
              f"{np.round(scl[1:4], 3).tolist()} a {acube:.3f} lag {lag:.2f} top fovy {fovy_t:.1f}", flush=True)
        if best is None or tot < best[0]:
            best = (tot, r.x)
    fovy_t, poses_t, ps, off, scl, acube, lag = unpack(best[1])
    print("BEST", round(best[0], 1), "offsets", np.round(off, 2).tolist(), "scales", np.round(scl, 3).tolist())
    if a.write:
        groups = {g: dict(pos=list(map(float, poses_t[k][0])),
                          quat=list(map(float, F.A.quat_from_mat(rotvec_to_mat(poses_t[k][1])))))
                  for k, g in enumerate(keys)}
        g1 = groups["g1_e039-099"]
        seed = dict(method="bundle3: top sessions + side + joint map, tcp <-> held cube at full rate (even episodes), "
                           "grasp-height prior",
                    joint=dict(offsets=off.tolist(), scales=scl.tolist(), lag=float(lag), cube_a=float(acube)),
                    cams=dict(top=dict(pos=g1["pos"], quat=g1["quat"], fovy=float(fovy_t), groups=groups),
                              side=dict(pos=list(map(float, ps[:3])),
                                        quat=list(map(float, F.A.quat_from_mat(rotvec_to_mat(ps[3:6])))),
                                        fovy=float(ps[6]))))
        (HERE / "out" / "seed_cams.json").write_text(json.dumps(seed, indent=1))
        print("wrote", HERE / "out" / "seed_cams.json")


if __name__ == "__main__":
    main()
