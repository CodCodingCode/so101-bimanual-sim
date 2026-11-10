"""Joint seed of the normalised-unit joint map AND both cameras with a fast splat objective (no MuJoCo rendering):
area-weighted surface samples of every arm geom posed by FK (mj_kinematics), projected through each camera (pinhole +
k1), splatted on a 160x120 grid (3x3 closing), IoU vs the SAM masks (4x4-pooled) of EVEN-episode frames only.
CMA over: joint scales (pan, lift, elbow, wrist flex), offsets (5), grip scale / offset, camera-board pose (6),
top pose (6) + fovy + k1, side pose per group (2 x 6) + fovy + k1.

    .venv/bin/python so101_twin/orangebox/joint_seed.py --init out/<fit>.json --masks _dino [--board] [--evals 6000]
        [--model so101|so100] --out out/<name>.json
Writes a fit-style json (params) usable as fit.py --init-json.
"""
from __future__ import annotations

import argparse
import dataclasses
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
from seed_search import surface_points  # noqa: E402

GW, GH = 160, 120


def closing(m):
    """3x3 closing on [N, H, W] bool."""
    p = np.pad(m, ((0, 0), (1, 1), (1, 1)))
    d = np.zeros_like(m)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            d |= p[:, dy:dy + m.shape[1], dx:dx + m.shape[2]]
    p = np.pad(d, ((0, 0), (1, 1), (1, 1)), constant_values=True)
    e = np.ones_like(m)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            e &= p[:, dy:dy + m.shape[1], dx:dx + m.shape[2]]
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--masks", default="_dino")
    ap.add_argument("--model", default="so101")
    ap.add_argument("--board", action="store_true")
    ap.add_argument("--evals", type=int, default=6000)
    ap.add_argument("--nframes", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--board-size0", type=float, default=0.0, help="initial board in-plane scale (default: from init)")
    ap.add_argument("--board0", default="", help="initial board offset dx,dy,dz,rx,ry,rz (default: from init)")
    a = ap.parse_args()
    sc = F.load_scene("orangebox")
    sc.ARM = dataclasses.replace(sc.ARM, model=a.model, wrist_mount=(a.model == "so101" and not a.board),
                                 cam_board=(0.019, 0.019, 0.006) if a.board else None,
                                 cam_board_pose=(0.0, 0.042, -0.04, 1, 0, 0, 0))
    init = json.loads((HERE / a.init if not pathlib.Path(a.init).is_absolute() else pathlib.Path(a.init)).read_text())
    P0 = F.reconcile(sc, init["params"])
    P0.setdefault("board", [0.0] * 6)
    if a.board_size0:
        P0["board_size"] = a.board_size0
    if a.board0:
        P0["board"] = [float(v) for v in a.board0.split(",")]
    rig = F.Rig(sc, P0)
    arm = rig.arm
    fs = F.real_set(sc)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    G, Pl = surface_points(arm, 7000, seed=a.seed)
    cbg = set(rig._cbg) if rig._cb is not None else set()
    bidx = np.array([k for k, g in enumerate(G) if g in cbg], int)
    Pl0 = Pl.copy()
    data = {}
    for c in sc.CAMERAS:
        real = F.load_masks("orangebox", c, a.masks)
        grp = F.frame_groups(sc, c)
        vis = np.nonzero((real.reshape(fs.n, -1).mean(1) >= F.VISIBLE) & (ep % 2 == 0))[0]
        X = fs.state[vis, :6] / 30.0
        sel = F.fps_select(X, a.nframes, seed=a.seed)
        idx = vis[sel]
        rc = real[idx].reshape(len(idx), GH, 480 // GH, GW, 640 // GW).mean((2, 4)) >= 0.5
        data[c] = (idx, grp[idx], rc)
        print(c, len(idx), "frames", flush=True)
    groups = {c: sorted(set(data[c][1])) for c in sc.CAMERAS}
    # parameter vector layout (scaled): see unpack()
    names = ["s_pan", "s_lift", "s_elbow", "s_wflex"] + [f"o{j}" for j in range(5)] + ["grip", "gripoff"] + \
        [f"b{j}" for j in range(6)] + ["bsize"]
    for c in sc.CAMERAS:
        for g in groups[c]:
            names += [f"{c}/{g}/{k}" for k in ("px", "py", "pz", "rx", "ry", "rz")]
        names += [f"{c}/fovy", f"{c}/k1"]
    sc_ = dict(s=0.05, o=3.0, grip=0.08, gripoff=0.05, bp=0.004, br=0.08, p=0.01, r=0.02, fovy=1.5, k1=0.05)

    def unpack(x):
        P = json.loads(json.dumps(P0))
        k = 0
        sca = np.asarray(P["scales"], float)
        sca[:4] += sc_["s"] * x[0:4]
        P["scales"] = sca.tolist()
        k = 4
        P["offsets_deg"] = (np.asarray(P["offsets_deg"]) + sc_["o"] * x[k:k + 5]).tolist()
        k += 5
        P["grip_scale"] += sc_["grip"] * x[k]
        P["grip_off"] = P.get("grip_off", 0.0) + sc_["gripoff"] * x[k + 1]
        k += 2
        bo = np.asarray(P.get("board", [0.0] * 6), float)
        bo[:3] += sc_["bp"] * x[k:k + 3]
        bo[3:] += sc_["br"] * x[k + 3:k + 6]
        P["board"] = bo.tolist()
        k += 6
        P["board_size"] = P.get("board_size", 1.0) + 0.05 * x[k]
        k += 1
        for c in sc.CAMERAS:
            cp = P["cams"][c]
            for g in groups[c]:
                gp = cp["groups"][g]
                gp["pos"] = list(np.asarray(gp["pos"]) + sc_["p"] * x[k:k + 3])
                gp["quat"] = list(A.quat_mul(gp["quat"], A.quat_from_rotvec(sc_["r"] * x[k + 3:k + 6])))
                k += 6
            cp["fovy"] += sc_["fovy"] * x[k]
            cp["lens"]["k1"] += sc_["k1"] * x[k + 1]
            k += 2
        return P

    def silhouettes(P):
        rig.set_params(P)
        if len(bidx):   # board samples scale with the fitted in-plane size (box centred at its geom origin)
            Pl[bidx, :2] = Pl0[bidx, :2] * float(P.get("board_size", 1.0))
        out = {}
        for c in sc.CAMERAS:
            idx, grp, rc = data[c]
            cp = P["cams"][c]
            f = 0.5 * GH / math.tan(math.radians(cp["fovy"]) / 2)
            L = cp["lens"]
            sims = np.zeros((len(idx), GH * GW), bool)
            for n, (i, g) in enumerate(zip(idx, grp)):
                q = A.lerobot_to_qpos(fs.state[i], arm.calib, P["offsets_deg"], P["grip_scale"], arm.model,
                                      P["scales"], P.get("grip_off", 0.0))
                arm.d.qpos[arm.qadr] = q
                mujoco.mj_kinematics(arm.m, arm.d)
                W = arm.d.geom_xpos[G] + np.einsum("nij,nj->ni", arm.d.geom_xmat[G].reshape(-1, 3, 3), Pl)
                gp = cp["groups"][g]
                R = A.mat_from_quat(np.asarray(gp["quat"]) / np.linalg.norm(gp["quat"]))
                pc = (W - np.asarray(gp["pos"])) @ R
                z = -pc[:, 2]
                ok = z > 0.02
                xu, yu = pc[:, 0] / np.maximum(z, 1e-3), -pc[:, 1] / np.maximum(z, 1e-3)
                r2 = xu * xu + yu * yu
                s = 1 + L["k1"] * r2 + L["k2"] * r2 * r2
                u = np.floor(f * xu * s + GW / 2 + L["cx"] * GW).astype(np.int32)
                v = np.floor(f * yu * s + GH / 2 + L["cy"] * GH).astype(np.int32)
                ok &= (u >= 0) & (u < GW) & (v >= 0) & (v < GH)
                sims[n, (v * GW + u)[ok]] = True
            out[c] = closing(sims.reshape(-1, GH, GW))
        return out

    def objective(x):
        P = unpack(x)
        sims = silhouettes(P)
        vals = []
        for c in sc.CAMERAS:
            rc = data[c][2]
            vals.append(float(F.iou_frames(sims[c], rc).mean()))
        reg = 0.004 * float(np.sum(((np.asarray(P["scales"]) - np.asarray(sc.SCALES_PRIOR)) / np.asarray(sc.SCALE_SIGMA)) ** 2))
        bo = np.asarray(P["board"])
        reg += 0.004 * float(np.sum((bo[:3] / 0.03) ** 2) + np.sum((bo[3:] / 0.79) ** 2))
        return -(float(np.mean(vals)) - reg), vals

    t0 = time.time()
    x0 = np.zeros(len(names))
    f0, v0 = objective(x0)
    print(f"start {-f0:.3f} per cam {np.round(v0, 3).tolist()} ({time.time() - t0:.1f}s/eval)", flush=True)
    es = cma.CMAEvolutionStrategy(x0, a.sigma, dict(maxfevals=a.evals, popsize=20, verbose=-9, seed=a.seed + 1,
                                                    bounds=[-12, 12]))
    best = (f0, x0, v0)
    it = 0
    while not es.stop():
        xs = es.ask()
        fv = []
        for x in xs:
            f, v = objective(x)
            fv.append(f)
            if f < best[0]:
                best = (f, np.array(x), v)
        es.tell(xs, fv)
        it += 1
        if it % 25 == 0:
            print(f"it {it} evals {es.countevals} best {-best[0]:.4f} cams {np.round(best[2], 3).tolist()} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    P = unpack(best[1])
    print(f"done {-best[0]:.4f} cams {np.round(best[2], 3).tolist()} in {time.time() - t0:.0f}s")
    print("scales", np.round(P["scales"], 3).tolist(), "offsets", np.round(P["offsets_deg"], 2).tolist(),
          "grip", round(P["grip_scale"], 3), round(P["grip_off"], 3), "board", np.round(P["board"], 4).tolist(),
          "board_size", round(P.get("board_size", 1.0), 3))
    for c in sc.CAMERAS:
        print(c, "fovy", round(P["cams"][c]["fovy"], 2), "k1", round(P["cams"][c]["lens"]["k1"], 3),
              {g: np.round(v["pos"], 3).tolist() for g, v in P["cams"][c]["groups"].items()})
    outp = HERE / a.out if not pathlib.Path(a.out).is_absolute() else pathlib.Path(a.out)
    outp.write_text(json.dumps(dict(scene="orangebox", model=a.model, board=a.board, params=P, splat_score=-best[0],
                                    splat_cams=best[2], masks=a.masks), indent=1))
    print("wrote", outp)


if __name__ == "__main__":
    main()
