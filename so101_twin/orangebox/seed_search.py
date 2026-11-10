"""Camera seed search without any twin prior: FK surface samples of the official SO-101 splatted onto a coarse grid,
IoU against the orange hue masks (hue_masks.py) of even-episode frames; wide random search over camera poses, then CMA
on the best candidates (still splats), then a final CMA with real MuJoCo renders.

    .venv/bin/python so101_twin/orangebox/seed_search.py top|side [--n 120000] [--group e10-48]
Writes so101_twin/orangebox/out/seed_<cam>.json (pos, quat, fovy, score) and seed_<cam>.jpg (overlay).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import cma
import mujoco
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
import arm as A  # noqa: E402
from sets import Store  # noqa: E402

GW, GH = 80, 60          # coarse grid (8 px cells)
fs = Store().frames("so101_orangebox_real")
EP = fs.labels["episode"].astype(int)


def unpack(a):
    return np.unpackbits(a, axis=-1)[..., :640].astype(bool)


def coarse(m, gw=GW, gh=GH, thr=0.3):
    n = m.shape[0]
    return m.reshape(n, gh, 480 // gh, gw, 640 // gw).mean((2, 4)) >= thr


def surface_points(a: A.Arm, n_per_frame=3000, seed=0):
    """area-weighted surface samples per arm geom, in each geom's local frame: (geom ids, local points)."""
    rng = np.random.default_rng(seed)
    m = a.m
    out_g, out_p, areas = [], [], []
    for g in a.arm_geoms:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:   # e.g. the camera board: 12 triangles of its 6 faces
            hx, hy, hz = m.geom_size[g]
            V = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            F = np.array([[0, 1, 3], [0, 3, 2], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                          [0, 2, 6], [0, 6, 4], [1, 3, 7], [1, 7, 5]])
            T = V[F]
            ar = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
            out_g.append((g, T, ar))
            areas.append(ar.sum())
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        V = m.mesh_vert[v0:v0 + nv]
        F = m.mesh_face[f0:f0 + nf]
        T = V[F]
        ar = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        out_g.append((g, T, ar))
        areas.append(ar.sum())
    tot = sum(areas)
    G, P = [], []
    for g, T, ar in out_g:
        k = max(30, int(n_per_frame * ar.sum() / tot))
        idx = rng.choice(len(T), size=k, p=ar / ar.sum())
        r1, r2 = rng.random(k), rng.random(k)
        s = np.sqrt(r1)
        pts = (1 - s)[:, None] * T[idx, 0] + (s * (1 - r2))[:, None] * T[idx, 1] + (s * r2)[:, None] * T[idx, 2]
        G.append(np.full(k, g))
        P.append(pts)
    return np.concatenate(G), np.concatenate(P)


def world_points(a: A.Arm, G, P, state):
    a.set_state(state)
    xp = a.d.geom_xpos[G]
    xm = a.d.geom_xmat[G].reshape(-1, 3, 3)
    return xp + np.einsum("nij,nj->ni", xm, P)


def cam_from(pos, target, roll_deg, up):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, (1, 0, 0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    t = math.radians(roll_deg)
    r, u = math.cos(t) * r + math.sin(t) * u, -math.sin(t) * r + math.cos(t) * u
    return np.stack([r, u, -f], 1)   # columns: cam x, y, z in world


def splat(Wp, pos, R, fovy, gw=GW, gh=GH):
    """Wp [N, P, 3] world points -> bool [N, gh, gw] coarse silhouettes (pinhole, MuJoCo camera convention)."""
    pc = (Wp - pos) @ R          # [N, P, 3] in camera frame
    z = -pc[..., 2]
    f = 0.5 * gh / math.tan(math.radians(fovy) / 2)
    ok = z > 0.02
    u = np.floor(f * pc[..., 0] / np.maximum(z, 1e-3) + gw / 2).astype(np.int32)
    v = np.floor(-f * pc[..., 1] / np.maximum(z, 1e-3) + gh / 2).astype(np.int32)
    ok &= (u >= 0) & (u < gw) & (v >= 0) & (v < gh)
    out = np.zeros((Wp.shape[0], gh * gw), bool)
    n_idx = np.broadcast_to(np.arange(Wp.shape[0])[:, None], u.shape)
    out[n_idx[ok], (v * gw + u)[ok]] = True
    return out.reshape(-1, gh, gw)


def iou(sim, real):
    i = (sim & real).reshape(len(sim), -1).sum(1)
    u = (sim | real).reshape(len(sim), -1).sum(1)
    return np.where(u > 0, i / np.maximum(u, 1), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--group", default="")
    ap.add_argument("--nframes", type=int, default=28)
    ap.add_argument("--roll", type=float, default=-55.0, help="wrist roll angle (deg) used for the seed")
    ap.add_argument("--masks", default="_hue")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    hue = unpack(np.load(HERE / "masks" / f"orangebox_{a.cam}{a.masks}.npy"))
    if a.masks == "_hue":
        hue &= ~(hue.mean(0) > 0.9)[None]
    ok = (EP % 2 == 0)
    if a.group == "e00-09":
        ok &= EP <= 9
    elif a.group == "e10-48":
        ok &= EP >= 10
    cand = np.nonzero(ok)[0]
    X = fs.state[cand, :6] / 30.0
    sel = [int(rng.integers(len(cand)))]
    d = np.linalg.norm(X - X[sel[0]], axis=1)
    while len(sel) < a.nframes:
        j = int(np.argmax(d))
        sel.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    frames = cand[np.array(sorted(sel))]
    spec = A.ArmSpec(calib="new", model="so101", wrist_mount=True)
    arm = A.Arm(A.build_mjcf(spec, [A.CamSpec(a.cam, pos=(0, 0.3, 0.8), quat=(1, 0, 0, 0))]), "new", "so101")
    arm.offsets_deg = np.array([0, 0, 0, 0, a.roll - (-55.0)])   # state roll ~ -55 -> roll angle a.roll (scale 1)
    G, P = surface_points(arm, 2500)
    Wp = np.stack([world_points(arm, G, P, fs.state[i]) for i in frames]).astype(np.float32)
    real_c = coarse(hue[frames])
    t0 = time.time()

    def sample():
        if a.cam == "top":
            tgt = np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.05, 0.45), 0.0])
            pos = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.1, 0.55), rng.uniform(0.3, 1.1)])
            return pos, tgt, rng.uniform(-50, 50), rng.uniform(35, 80), np.array([0.0, -1.0, 0.0])
        tgt = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.05, 0.45), rng.uniform(0.0, 0.15)])
        az = rng.uniform(0, 2 * math.pi)
        el = math.radians(rng.uniform(3, 65))
        dist = rng.uniform(0.25, 1.1)
        pos = tgt + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return pos, tgt, rng.uniform(-70, 70), rng.uniform(35, 85), np.array([0.0, 0.0, 1.0])

    res = []
    for k in range(a.n):
        pos, tgt, roll, fovy, up = sample()
        if pos[2] < 0.02:
            continue
        R = cam_from(pos, tgt, roll, up)
        s = float(iou(splat(Wp, pos, R, fovy), real_c).mean())
        res.append((s, pos, R, fovy))
        if (k + 1) % 20000 == 0:
            res.sort(key=lambda r: -r[0])
            res = res[:200]
            print(f"{k + 1} samples {time.time() - t0:.0f}s best {res[0][0]:.3f}", flush=True)
    res.sort(key=lambda r: -r[0])
    res = res[:200]
    print(f"random: best {res[0][0]:.3f}, 10th {res[9][0]:.3f} ({time.time() - t0:.0f}s)", flush=True)

    # CMA on the splat objective from the 12 best distinct candidates: x = [dpos(3) m*0.03, rotvec(3)*0.05, dfovy*3]
    def params(base, x):
        pos0, R0, f0 = base
        R = R0 @ A.mat_from_quat(A.quat_from_rotvec(0.05 * x[3:6]))
        return pos0 + 0.03 * x[:3], R, f0 + 3.0 * x[6]

    def f_splat(base, x):
        pos, R, fovy = params(base, x)
        if not (20 < fovy < 100) or pos[2] < 0.02:
            return 1.0
        return -float(iou(splat(Wp, pos, R, fovy), real_c).mean())

    seeds, kept = [], []
    for s, pos, R, fovy in res:
        if all(np.linalg.norm(pos - p2) > 0.05 or abs(fovy - f2) > 5 for _, p2, _, f2 in kept):
            kept.append((s, pos, R, fovy))
        if len(kept) == 12:
            break
    for s, pos, R, fovy in kept:
        base = (pos, R, fovy)
        es = cma.CMAEvolutionStrategy(np.zeros(7), 0.8, dict(maxfevals=1500, popsize=14, verbose=-9, seed=1))
        best = (-s, np.zeros(7))
        while not es.stop():
            xs = es.ask()
            fv = [f_splat(base, x) for x in xs]
            es.tell(xs, fv)
            if min(fv) < best[0]:
                best = (min(fv), xs[int(np.argmin(fv))])
        p2, R2, f2 = params(base, best[1])
        seeds.append((-best[0], p2, R2, f2))
        print(f"  cand {s:.3f} -> {-best[0]:.3f}  pos {np.round(p2, 3)} fovy {f2:.1f}", flush=True)
    seeds.sort(key=lambda r: -r[0])

    # final: real renders (fine IoU at 160x120) for the top 4, CMA
    hue_f = hue[frames].reshape(len(frames), 120, 4, 160, 4).mean((2, 4)) >= 0.5

    def f_render(base, x):
        pos, R, fovy = params(base, x)
        if not (20 < fovy < 100):
            return 1.0
        arm.set_camera(a.cam, pos=pos, quat=A.quat_from_mat(R), fovy=fovy)
        sims = []
        for i in frames:
            arm.set_state(fs.state[i])
            sims.append(arm.arm_mask(a.cam, 160, 120))
        return -float(iou(np.stack(sims), hue_f).mean())

    final = []
    for s, pos, R, fovy in seeds[:4]:
        base = (pos, R, fovy)
        s0 = -f_render(base, np.zeros(7))
        es = cma.CMAEvolutionStrategy(np.zeros(7), 0.3, dict(maxfevals=500, popsize=12, verbose=-9, seed=2))
        best = (-s0, np.zeros(7))
        while not es.stop():
            xs = es.ask()
            fv = [f_render(base, x) for x in xs]
            es.tell(xs, fv)
            if min(fv) < best[0]:
                best = (min(fv), xs[int(np.argmin(fv))])
        p2, R2, f2 = params(base, best[1])
        final.append((-best[0], p2, R2, f2))
        print(f"  render {s0:.3f} -> {-best[0]:.3f} pos {np.round(p2, 3)} fovy {f2:.1f} ({time.time() - t0:.0f}s)", flush=True)
    final.sort(key=lambda r: -r[0])
    s, pos, R, fovy = final[0]
    out = dict(cam=a.cam, group=a.group, roll=a.roll, score_render160=s, pos=pos.tolist(),
               quat=A.quat_from_mat(R).tolist(), fovy=fovy, frames=frames.tolist(),
               others=[dict(score=f[0], pos=f[1].tolist(), quat=A.quat_from_mat(f[2]).tolist(), fovy=f[3]) for f in final[1:]])
    tag = f"{a.cam}{('_' + a.group) if a.group else ''}{'' if a.roll == -55 else f'_r{int(a.roll)}'}"
    (HERE / "out" / f"seed_{tag}.json").write_text(json.dumps(out, indent=1))
    arm.set_camera(a.cam, pos=pos, quat=A.quat_from_mat(R), fovy=fovy)
    tiles = []
    for i in frames[:8]:
        arm.set_state(fs.state[i])
        sm = arm.arm_mask(a.cam, 640, 480)
        im = np.asarray(fs.obs(int(i))["images"][a.cam]).astype(np.float32) * 0.6
        im[sm] = 0.5 * im[sm] + np.array([0, 0, 120])
        im[hue[i] & ~sm] = (255, 60, 60)
        tiles.append(im.astype(np.uint8))
    g = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:8], 1)], 0)
    Image.fromarray(g).resize((1280, 480)).save(HERE / "out" / f"seed_{tag}.jpg", quality=85)
    print(f"wrote seed_{tag}.json score {s:.3f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
