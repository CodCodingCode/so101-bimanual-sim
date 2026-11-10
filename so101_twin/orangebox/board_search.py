"""Global search of the camera-board pose (gripper frame) + in-plane size, all other params fixed (fast splat IoU,
both cameras, even episodes). Random rotations / positions around the gripper, then CMA on the best candidates.
    .venv/bin/python so101_twin/orangebox/board_search.py --init out/<fit>.json --masks _c1 --out out/<name>.json
Writes the init json with params.board / board_size replaced (board = offset from the scene's cam_board_pose)."""
import argparse
import json
import math
import pathlib
import sys
import time

import cma
import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A  # noqa: E402
import fit as F  # noqa: E402
from joint_seed import closing, GW, GH  # noqa: E402
from seed_search import surface_points  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--masks", default="_c1")
ap.add_argument("--n", type=int, default=2500)
ap.add_argument("--out", required=True)
ap.add_argument("--nframes", type=int, default=48)
ap.add_argument("--model", default="so101")
a = ap.parse_args()
sc = F.load_scene("orangebox")
if a.model != sc.ARM.model:
    import dataclasses
    sc.ARM = dataclasses.replace(sc.ARM, model=a.model)
P0 = F.reconcile(sc, json.loads((HERE / a.init).read_text())["params"])
P0.setdefault("board", [0.0] * 6)
rig = F.Rig(sc, P0)
arm = rig.arm
fs = F.real_set(sc)
ep = np.asarray(fs.labels["episode"]).astype(int)
G, Pl = surface_points(arm, 7000, seed=5)
cbg = set(rig._cbg)
bmask = np.array([g in cbg for g in G])
Pl0 = Pl.copy()
data = {}
for c in sc.CAMERAS:
    real = F.load_masks("orangebox", c, a.masks)
    grp = F.frame_groups(sc, c)
    vis = np.nonzero((real.reshape(fs.n, -1).mean(1) >= F.VISIBLE) & (ep % 2 == 0))[0]
    idx = vis[F.fps_select(fs.state[vis, :6] / 30.0, a.nframes, seed=7)]
    data[c] = (idx, grp[idx], real[idx].reshape(len(idx), GH, 480 // GH, GW, 640 // GW).mean((2, 4)) >= 0.5)
# precompute per frame: world points of the non-board geoms, and the gripper pose (board points depend on board params)
gb = mujoco.mj_name2id(arm.m, mujoco.mjtObj.mjOBJ_BODY, "gripper" if a.model == "so101" else "Fixed_Jaw")
cbid = rig._cb
pre = {}
for c in sc.CAMERAS:
    idx, grp, rc = data[c]
    lst = []
    for i, g in zip(idx, grp):
        arm.set_state(F.frame_state(sc, P0, c, i))
        W = arm.d.geom_xpos[G[~bmask]] + np.einsum("nij,nj->ni", arm.d.geom_xmat[G[~bmask]].reshape(-1, 3, 3), Pl[~bmask])
        lst.append((W, arm.d.xpos[gb].copy(), arm.d.xmat[gb].reshape(3, 3).copy()))
    pre[c] = lst
# board local samples (in the board body frame): geoms' own offsets inside the body
bg = sorted(cbg)
BL = []
for g in bg:
    sel = np.array([gg == g for gg in G])
    BL.append(arm.m.geom_pos[g] + Pl0[sel] @ A.mat_from_quat(arm.m.geom_quat[g]).T)
BL = np.concatenate(BL)
BLxy = BL.copy()
base_pos, base_quat = rig._cb0
board_half = rig._cbg0[bg[0]]


def project(c, W, g):
    cp = P0["cams"][c]
    gp = cp["groups"][g]
    R = A.mat_from_quat(np.asarray(gp["quat"]) / np.linalg.norm(gp["quat"]))
    f = 0.5 * GH / math.tan(math.radians(cp["fovy"]) / 2)
    L = cp["lens"]
    pc = (W - np.asarray(gp["pos"])) @ R
    z = -pc[:, 2]
    ok = z > 0.02
    xu, yu = pc[:, 0] / np.maximum(z, 1e-3), -pc[:, 1] / np.maximum(z, 1e-3)
    r2 = xu * xu + yu * yu
    s = 1 + L["k1"] * r2 + L["k2"] * r2 * r2
    u = np.floor(f * xu * s + GW / 2 + L["cx"] * GW).astype(np.int32)
    v = np.floor(f * yu * s + GH / 2 + L["cy"] * GH).astype(np.int32)
    ok &= (u >= 0) & (u < GW) & (v >= 0) & (v < GH)
    return (v * GW + u)[ok]


# arm-only silhouettes (fixed)
base_sil = {}
for c in sc.CAMERAS:
    idx, grp, rc = data[c]
    S = np.zeros((len(idx), GH * GW), bool)
    for n, ((W, _, _), g) in enumerate(zip(pre[c], grp)):
        S[n, project(c, W, g)] = True
    base_sil[c] = S


def score(pos, quat, size):
    """board pose (gripper frame) + in-plane scale -> mean IoU over both cameras."""
    Rb = A.mat_from_quat(quat)
    loc = BL.copy()
    loc[:, :2] *= size
    loc = loc @ Rb.T + pos
    vals = []
    for c in sc.CAMERAS:
        idx, grp, rc = data[c]
        S = base_sil[c].copy()
        for n, ((_, gp, gR), g) in enumerate(zip(pre[c], grp)):
            S[n, project(c, gp + loc @ gR.T, g)] = True
        S = closing(S.reshape(-1, GH, GW))
        vals.append(float(F.iou_frames(S, rc).mean()))
    return float(np.mean(vals)), vals


def cur():
    bo = np.asarray(P0["board"])
    return base_pos + bo[:3], A.quat_mul(A.quat_from_rotvec(bo[3:6]), base_quat), float(P0.get("board_size", 1.0))


t0 = time.time()
p_c, q_c, s_c = cur()
s0, v0 = score(p_c, q_c, s_c)
print(f"current board: {s0:.4f} {np.round(v0, 3).tolist()}", flush=True)
rng = np.random.default_rng(0)
res = [(s0, p_c, q_c, s_c)]
for k in range(a.n):
    pos = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.07, 0.07), rng.uniform(-0.09, 0.02)]) if a.model == "so101" \
        else rng.uniform(-0.09, 0.09, 3)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    size = rng.uniform(1.0, 1.4)
    s, v = score(pos, q, size)
    res.append((s, pos, q, size))
res.sort(key=lambda r: -r[0])
print(f"random {a.n}: best {res[0][0]:.4f} ({time.time() - t0:.0f}s)", flush=True)
final = []
for s, pos, q, size in res[:6]:
    def fobj(x, pos=pos, q=q, size=size):
        return -score(pos + 0.005 * x[:3], A.quat_mul(A.quat_from_rotvec(0.1 * x[3:6]), q), size + 0.05 * x[6])[0]
    es = cma.CMAEvolutionStrategy(np.zeros(7), 0.8, dict(maxfevals=500, popsize=12, verbose=-9, seed=3))
    best = (-s, np.zeros(7))
    while not es.stop():
        xs = es.ask()
        fv = [fobj(x) for x in xs]
        es.tell(xs, fv)
        if min(fv) < best[0]:
            best = (min(fv), xs[int(np.argmin(fv))])
    x = best[1]
    final.append((-best[0], pos + 0.005 * x[:3], A.quat_mul(A.quat_from_rotvec(0.1 * x[3:6]), q), size + 0.05 * x[6]))
    print(f"  {s:.4f} -> {-best[0]:.4f} pos {np.round(final[-1][1], 4).tolist()} size {final[-1][3]:.3f}", flush=True)
final.sort(key=lambda r: -r[0])
s, pos, q, size = final[0]
print("best", round(s, 4), score(pos, q, size)[1], f"({time.time() - t0:.0f}s)")
# express as offset from the scene's init pose: pos = base_pos + d, quat = rotvec(r) * base_quat
d = pos - base_pos
rq = A.quat_mul(q, np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]]))
rq = rq if rq[0] >= 0 else -rq
ang = 2 * math.acos(min(1.0, rq[0]))
rv = np.zeros(3) if ang < 1e-9 else rq[1:] / math.sin(ang / 2) * ang
P = json.loads(json.dumps(P0))
P["board"] = d.tolist() + rv.tolist()
P["board_size"] = float(size)
(HERE / a.out).write_text(json.dumps(dict(scene="orangebox", model=a.model, params=P, board_search=dict(
    score=s, pos=pos.tolist(), quat=q.tolist(), size=size)), indent=1))
print("wrote", a.out, "board", np.round(P["board"], 4).tolist(), "size", round(size, 3))
