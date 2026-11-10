"""Side camera + joint GAINS from the held cube, top cameras FIXED to a fit json (EVEN episodes only).

    .venv/bin/python so101_twin/stack2/bundle4.py FIT_JSON [--write OUT_JSON]

The dataset is in LeRobot's normalised -100..100 units (action stats hit exactly -100 / +100), so each joint needs
q = gain * units + offset with gain ~0.95-1.25 deg/unit. Unknowns: side camera pose + fovy (>= 40 deg: the cube is 2.5x
larger on the block than on the disc, so the side camera is close, not telephoto), offsets (pan, lift, elbow, wrist),
gains (pan, lift, elbow, wrist) in [0.8, 1.3], cube-centre offset, latency. Residuals: top (fixed cameras) + side
cube-centroid reprojection.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
from bundle import FK, load_obs  # noqa: E402
from seed_pnp import _rotvec, project, rotvec_to_mat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fit_json")
ap.add_argument("--write", default="")
ap.add_argument("--n", type=int, default=150)
a = ap.parse_args()
sc = F.load_scene("stack")
P = F.reconcile(sc, json.loads(pathlib.Path(a.fit_json).read_text())["params"])
obs, states = load_obs(sc, max_per_cam=100000)
fk = FK(sc)
rng = np.random.default_rng(0)
rows_t = obs["top"][np.sort(rng.choice(len(obs["top"]), min(a.n * 2, len(obs["top"])), replace=False))]
rows_s = obs["side"][np.sort(rng.choice(len(obs["side"]), min(a.n * 2, len(obs["side"])), replace=False))]
gt = [sc.top_group(int(e)) for e in rows_t[:, 0]]
tp = P["cams"]["top"]
Rt = {g: (np.asarray(v["pos"]), F.A.mat_from_quat(v["quat"])) for g, v in tp["groups"].items()}
sp = P["cams"]["side"]["groups"]["all"]


def unpack(p):
    pos, rv, fovy = p[0:3], p[3:6], p[6]
    off = np.array([p[7], p[8], p[9], p[10], 0.0])
    scl = np.array([p[11], p[12], p[13], p[14], 1.0])
    return pos, rv, fovy, off, scl, p[15], p[16]


def resid(p):
    pos, rv, fovy, off, scl, acube, lag = unpack(p)
    X, Z = fk(states, rows_t, lag, off, scl)
    Xo = X + acube * Z
    rt = np.zeros((len(X), 2))
    for g in Rt:
        mk = np.array([x == g for x in gt])
        if mk.any():
            pr, zz = project(Rt[g][0], Rt[g][1], tp["fovy"], Xo[mk])
            rt[mk] = pr - rows_t[mk, 2:4]
    X, Z = fk(states, rows_s, lag, off, scl)
    pr, zz = project(pos, rotvec_to_mat(rv), fovy, X + acube * Z)
    rs = (pr - rows_s[:, 2:4]) * np.where(zz[:, None] > 0.02, 1.0, 30.0)
    return np.concatenate([rt.ravel(), rs.ravel(), (off[:4] - np.asarray(P["offsets_deg"][:4])) / 30.0])


best = None
for s in range(6):
    p0 = np.concatenate([np.asarray(sp["pos"]) + rng.normal(0, 0.03, 3), _rotvec(F.A.mat_from_quat(sp["quat"])) + rng.normal(0, 0.05, 3),
                         [max(45.0, P["cams"]["side"]["fovy"]) + rng.normal(0, 5)], P["offsets_deg"][:4],
                         np.clip(np.asarray(P.get("scales", [1] * 5))[:4] + rng.normal(0, 0.05, 4), 0.85, 1.25),
                         [-0.01, P["cams"]["side"].get("dt_frames", -1.0)]])
    lb = [-np.inf] * 6 + [40] + [-60] * 4 + [0.8] * 4 + [-0.02, -5]
    ub = [np.inf] * 6 + [95] + [60] * 4 + [1.3] * 4 + [0.05, 3]
    r = least_squares(resid, np.clip(p0, np.array(lb) + 1e-6, np.array(ub) - 1e-6), loss="soft_l1", f_scale=8.0,
                      bounds=(lb, ub), max_nfev=80)
    rr = resid(r.x)
    nt = 2 * len(rows_t)
    et = np.linalg.norm(rr[:nt].reshape(-1, 2), axis=1)
    es = np.linalg.norm(rr[nt:nt + 2 * len(rows_s)].reshape(-1, 2), axis=1)
    pos, rv, fovy, off, scl, acube, lag = unpack(r.x)
    print(f"start {s}: top {np.median(et):.1f} px side {np.median(es):.1f} px | side pos {np.round(pos, 3).tolist()} fovy "
          f"{fovy:.1f} | off {np.round(off[:4], 1).tolist()} gains {np.round(scl[:4], 3).tolist()} lag {lag:.2f}", flush=True)
    if best is None or np.median(et) + np.median(es) < best[0]:
        best = (np.median(et) + np.median(es), r.x)
if a.write:
    pos, rv, fovy, off, scl, acube, lag = unpack(best[1])
    Q = json.loads(json.dumps(P))
    Q["cams"]["side"]["groups"]["all"] = dict(pos=list(map(float, pos)), quat=list(map(float, F.A.quat_from_mat(rotvec_to_mat(rv)))))
    Q["cams"]["side"]["fovy"] = float(fovy)
    Q["offsets_deg"] = list(map(float, off))
    Q["scales"] = list(map(float, scl))
    pathlib.Path(a.write).write_text(json.dumps(dict(params=Q, note="bundle4: side + joint gains from the held cube"), indent=1))
    print("wrote", a.write)
