"""REDLEGO copy of jar/jseed.py (scene redlego, scale prior N(1.0, 0.15), default masks _d1k). JAR: joint seed of the normalised-unit joint map AND both cameras (twin-free masks, EVEN episodes only).

    .venv/bin/python so101_twin/jar/jseed.py [--mask-tag _d1c] [--evals 3000] [--frames 24] [--model so101]
        [--init out/<fit>.json] --out out/jseed.json

CMA over: joint scales (pan, lift, elbow, wrist flex; roll pinned at 1), offsets (5), gripper scale, and per camera
pose (6) + fovy. Objective: 10 %-trimmed mean IoU at 160x120 (pinhole render, no lens) over frames chosen by
farthest-point sampling in joint space, both cameras averaged, minus weak priors (scales ~ N(1.1, 0.15), offsets ~
N(0, 12 deg)). Writes a fit-style json (params) usable as fit.py --init-json.
"""
import argparse
import dataclasses
import json
import pathlib
import sys
import time

import cma
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

CAMS = ("top", "side")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-tag", default="_d1k")
    ap.add_argument("--evals", type=int, default=3000)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--model", default="so101")
    ap.add_argument("--init", default="")
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roll-scan", action="store_true", help="REDLEGO: scan the wrist-roll offset (15 deg grid) first")
    ap.add_argument("--reset-joints", action="store_true", help="start offsets at 0 and scales at SCALES_INIT "
                    "(cameras from --init), e.g. for the other mesh")
    a = ap.parse_args()
    sc = F.load_scene("redlego")
    sc.ARM = dataclasses.replace(sc.ARM, model=a.model)
    P0 = F.reconcile(sc, json.loads(pathlib.Path(a.init).read_text())["params"]) if a.init else F.init_params(sc)
    if a.reset_joints:
        P0["offsets_deg"] = [0.0] * 5
        P0["scales"] = list(map(float, sc.SCALES_INIT))
        P0["grip_scale"] = 1.0
    rig = F.Rig(sc, P0)
    fs = F.real_set(sc)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    X = fs.state[:, :6] / 30.0
    data = {}
    for c in CAMS:
        real = F.load_masks("redlego", c, a.mask_tag)
        small = real.reshape(fs.n, 120, 4, 160, 4).mean((2, 4)) >= 0.5
        area = small.reshape(fs.n, -1).mean(1)
        vis = np.nonzero((ep % 2 == 0) & (area > 0.005))[0]
        sel = vis[F.fps_select(X[vis], min(a.frames, len(vis)), seed=a.seed)]
        data[c] = (sel, small[sel])
    arm = rig.arm
    cid = {c: arm.cam_id[c] for c in CAMS}

    def unpack(x):
        P = json.loads(json.dumps(P0))
        sc0 = np.asarray(P0["scales"], float)
        P["scales"] = list(np.r_[sc0[:4] + 0.05 * x[0:4], sc0[4]])
        P["offsets_deg"] = list(np.asarray(P0["offsets_deg"], float) + 5.0 * x[4:9])
        P["grip_scale"] = float(P0["grip_scale"] + 0.1 * x[9])
        k = 10
        for c in CAMS:
            g = P["cams"][c]["groups"]["all"]
            g["pos"] = list(np.asarray(g["pos"], float) + 0.02 * x[k:k + 3])
            g["quat"] = list(F.A.quat_mul(g["quat"], F.A.quat_from_rotvec(0.03 * x[k + 3:k + 6])))
            P["cams"][c]["fovy"] = float(P["cams"][c]["fovy"] + 2.0 * x[k + 6])
            k += 7
        return P

    def score(P, reg=True):
        arm.offsets_deg = np.asarray(P["offsets_deg"], float)
        arm.scales = np.asarray(P["scales"], float)
        arm.grip_scale = float(P["grip_scale"])
        vals = []
        for c in CAMS:
            g = P["cams"][c]["groups"]["all"]
            arm.m.cam_pos[cid[c]] = g["pos"]
            arm.m.cam_quat[cid[c]] = np.asarray(g["quat"]) / np.linalg.norm(g["quat"])
            arm.m.cam_fovy[cid[c]] = P["cams"][c]["fovy"]
            sel, real = data[c]
            v = []
            for i, r in zip(sel, real):
                arm.set_state(fs.state[i])
                m = arm.arm_mask(c, 160, 120)
                u = (m | r).sum()
                v.append((m & r).sum() / u if u else 0.0)
            v = np.sort(v)
            vals.append(float(v[int(0.1 * len(v)):].mean()))
        s = float(np.mean(vals))
        if reg:
            s -= 0.004 * float(np.sum(((np.asarray(P["scales"][:4]) - 1.0) / 0.15) ** 2))
            s -= 0.004 * float(np.sum(((np.asarray(P["offsets_deg"]) - np.asarray(P0["offsets_deg"])) / 12.0) ** 2))
        return s, vals

    t0 = time.time()
    if a.roll_scan:   # REDLEGO: the wrist-roll zero is arbitrary (state ~ -55 throughout): scan its offset first
        best_r = None
        for r in np.arange(-180, 180, 15.0):
            Q = json.loads(json.dumps(P0))
            Q["offsets_deg"][4] = float(r)
            v = score(Q, reg=False)[0]
            print(f"roll offset {r:+.0f}: {v:.3f}", flush=True)
            if best_r is None or v > best_r[0]:
                best_r = (v, float(r))
        P0["offsets_deg"][4] = best_r[1]
        print(f"roll scan best offset {best_r[1]:+.0f} ({best_r[0]:.3f})", flush=True)
    s0, v0 = score(P0)
    print(f"start {s0:.3f} {np.round(v0, 3).tolist()}", flush=True)
    es = cma.CMAEvolutionStrategy(np.zeros(24), a.sigma, dict(maxfevals=a.evals, popsize=16, verbose=-9,
                                                              seed=a.seed + 1, bounds=[-8, 8]))
    best = (s0, np.zeros(24))
    it = 0
    while not es.stop():
        xs = es.ask()
        fv = [-score(unpack(x))[0] for x in xs]
        es.tell(xs, fv)
        if -min(fv) > best[0]:
            best = (-min(fv), xs[int(np.argmin(fv))])
        it += 1
        if it % 20 == 0:
            Pb = unpack(best[1])
            print(f"[{time.time() - t0:.0f}s {es.countevals} evals] best {best[0]:.3f} {np.round(score(Pb)[1], 3).tolist()} "
                  f"scales {np.round(Pb['scales'], 3).tolist()} off {np.round(Pb['offsets_deg'], 1).tolist()} "
                  f"grip {Pb['grip_scale']:.2f}", flush=True)
    P = unpack(best[1])
    s, v = score(P)
    print(f"final {s:.3f} {np.round(v, 3).tolist()} scales {np.round(P['scales'], 3).tolist()} off "
          f"{np.round(P['offsets_deg'], 2).tolist()} grip {P['grip_scale']:.3f} fovy "
          f"{[round(P['cams'][c]['fovy'], 1) for c in CAMS]} ({time.time() - t0:.0f}s)", flush=True)
    pathlib.Path(a.out).write_text(json.dumps(dict(scene="redlego", model=a.model, params=P, jseed_score=s, per_cam=v,
                                                   mask_tag=a.mask_tag), indent=1))


if __name__ == "__main__":
    main()
