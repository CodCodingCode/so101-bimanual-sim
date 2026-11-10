"""Wide random + CMA search of ONE camera pose (+ fovy) scored by silhouette IoU against rough masks, joint map fixed.

    .venv/bin/python so101_twin/stack2/cam_search.py side [--mask-tag _fg] [--group all] [--n 4000]

Used once to seed the side camera (its cube-centroid PnP was ill-conditioned). EVEN episodes only. Candidate poses:
position in a box around a geometric guess, looking at a random target in the work area, random roll and fovy.
Writes out/cam_search_<cam>_<group>.json.
"""
import argparse
import json
import math
import pathlib
import sys
import time

import cma
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

GUESS = {"side": dict(pos=(0.30, 0.15, 0.08), pos_sd=(0.15, 0.15, 0.06), target=(0.05, 0.20, 0.04), target_sd=0.08),
         "top": dict(pos=(0.40, 0.23, 0.25), pos_sd=(0.15, 0.15, 0.12), target=(0.05, 0.18, 0.03), target_sd=0.08)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam")
    ap.add_argument("--mask-tag", default="_fg")
    ap.add_argument("--group", default="all")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--offsets", default="0,-2.28,-25.70,29.58,0")
    ap.add_argument("--scales", default="1,0.923,1.107,0.988,1")
    ap.add_argument("--from-seed", action="store_true", help="start CMA from out/seed_cams.json (no random search)")
    ap.add_argument("--dt", type=float, default=-1.7)
    a = ap.parse_args()
    sc = F.load_scene("stack")
    P = F.init_params(sc)
    if not a.from_seed:
        P["offsets_deg"] = [float(x) for x in a.offsets.split(",")]
        P["scales"] = [float(x) for x in a.scales.split(",")]
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    real = F.load_masks("stack", a.cam, a.mask_tag)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    grp = F.frame_groups(sc, a.cam)
    area = real.reshape(fs.n, -1).mean(1)
    ok = (ep % 2 == 0) & ((grp == a.group) if a.group != "all" else True)
    vis = np.nonzero(ok & (area > 0.005))[0]
    X = fs.state[:, :6] / 30.0
    sel = vis[F.fps_select(X[vis], min(36, len(vis)))]
    small = np.stack([real[i].reshape(120, 4, 160, 4).mean((1, 3)) >= 0.5 for i in sel])
    fr = F.fullrate(sc)
    states = [fr.state_at(int(i), a.dt) for i in sel]
    cid = rig.arm.cam_id[a.cam]
    rig.arm.set_camera(a.cam)

    def score(pos, quat, fovy):
        rig.arm.m.cam_pos[cid] = pos
        rig.arm.m.cam_quat[cid] = quat
        rig.arm.m.cam_fovy[cid] = fovy
        v = []
        for s, r in zip(states, small):
            rig.arm.set_state(s)
            m = rig.arm.arm_mask(a.cam, 160, 120)
            u = (m | r).sum()
            v.append((m & r).sum() / u if u else 0.0)
        return float(np.mean(v))

    g = GUESS[a.cam]
    rng = np.random.default_rng(0)
    t0 = time.time()
    cands = []
    if a.from_seed:
        cp = P["cams"][a.cam]
        gg = cp["groups"]["all" if a.group == "all" else a.group]
        cands.append((score(np.asarray(gg["pos"]), np.asarray(gg["quat"]), cp["fovy"]), np.asarray(gg["pos"]), None,
                      np.asarray(gg["quat"]), cp["fovy"]))
        a.n = 0
    for k in range(a.n):
        pos = np.asarray(g["pos"]) + rng.normal(0, 1, 3) * np.asarray(g["pos_sd"])
        tgt = np.asarray(g["target"]) + rng.normal(0, g["target_sd"], 3)
        q = F.A.look_quat(pos, tgt, roll_deg=rng.normal(0, 25))
        fovy = rng.uniform(35, 85)
        cands.append((score(pos, q, fovy), pos, tgt, q, fovy))
    cands.sort(key=lambda c: -c[0])
    print(f"{a.cam}: {len(sel)} frames; random search {a.n} in {time.time() - t0:.0f}s; best {[round(c[0], 3) for c in cands[:5]]}",
          flush=True)
    best = None
    for c in cands[:6 if not a.from_seed else 1]:
        x0 = np.concatenate([c[1], np.zeros(3), [c[4]]])
        q0 = c[3]

        def f(x):
            q = F.A.quat_mul(q0, F.A.quat_from_rotvec(x[3:6] * 0.05))
            return -score(x[:3], q, x[6])
        es = cma.CMAEvolutionStrategy(x0, 0.02, dict(maxfevals=700, verbose=-9, seed=1,
                                                     CMA_stds=[1, 1, 1, 1, 1, 1, 100]))
        es.optimize(f)
        xb = es.result.xbest
        sb = -es.result.fbest
        print(f"  cand {c[0]:.3f} -> {sb:.3f} pos {np.round(xb[:3], 3).tolist()} fovy {xb[6]:.1f} ({time.time() - t0:.0f}s)", flush=True)
        if best is None or sb > best[0]:
            best = (sb, xb[:3], F.A.quat_mul(q0, F.A.quat_from_rotvec(xb[3:6] * 0.05)), xb[6])
    out = dict(cam=a.cam, group=a.group, iou_fg=best[0], pos=list(map(float, best[1])), quat=list(map(float, best[2])),
               fovy=float(best[3]), offsets=P["offsets_deg"], scales=P["scales"], frames=sel.tolist())
    (HERE / "out" / f"cam_search_{a.cam}_{a.group}.json").write_text(json.dumps(out, indent=1))
    print("best", round(best[0], 3), out["pos"], out["fovy"])


if __name__ == "__main__":
    main()
