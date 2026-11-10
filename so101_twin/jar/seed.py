"""JAR: twin-free camera seed for ONE world camera (pose + fovy) against twin-independent masks (default _d1c),
joint map fixed at the scene init (normalised units x SCALES_INIT, offsets 0). EVEN episodes only.

    .venv/bin/python so101_twin/jar/seed.py top [--mask-tag _d1c] [--n 2500] [--frames 20]

Random look-at candidates around a geometric guess (top: above the work area, any roll; side: low at the front-left,
roll +-15 deg), fovy U(35, 75); score = mean IoU at 160x120 (pinhole, no lens) over frames chosen by farthest-point
sampling in joint space; CMA on the best few. Writes out/seed_<cam>.json (read by scene_jar.py).
"""
import argparse
import json
import pathlib
import sys
import time

import cma
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

GUESS = {"top": dict(pos=(0.0, 0.22, 0.55), pos_sd=(0.12, 0.12, 0.15), target=(0.0, 0.25, 0.0), target_sd=0.08,
                     roll_sd=None, up=(0, 1, 0)),
         "side": dict(pos=(-0.40, 0.20, 0.12), pos_sd=(0.12, 0.15, 0.06), target=(0.0, 0.18, 0.08), target_sd=0.08,
                      roll_sd=15.0, up=(0, 0, 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam")
    ap.add_argument("--mask-tag", default="_d1c")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--cma", type=int, default=400)
    a = ap.parse_args()
    sc = F.load_scene("jar")
    P = F.init_params(sc)
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    real = F.load_masks("jar", a.cam, a.mask_tag)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    small_all = real.reshape(fs.n, 120, 4, 160, 4).mean((2, 4)) >= 0.5
    area = small_all.reshape(fs.n, -1).mean(1)
    vis = np.nonzero((ep % 2 == 0) & (area > 0.005))[0]
    X = fs.state[:, :6] / 30.0
    sel = vis[F.fps_select(X[vis], min(a.frames, len(vis)))]
    small = small_all[sel]
    states = [fs.state[i] for i in sel]
    cid = rig.arm.cam_id[a.cam]

    def score(pos, quat, fovy):
        rig.arm.m.cam_pos[cid] = pos
        rig.arm.m.cam_quat[cid] = np.asarray(quat) / np.linalg.norm(quat)
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
    for k in range(a.n):
        pos = np.asarray(g["pos"]) + rng.normal(0, 1, 3) * np.asarray(g["pos_sd"])
        if a.cam == "side":
            pos[2] = max(pos[2], 0.03)
        tgt = np.asarray(g["target"]) + rng.normal(0, g["target_sd"], 3)
        roll = rng.uniform(-180, 180) if g["roll_sd"] is None else rng.normal(0, g["roll_sd"])
        q = F.A.look_quat(pos, tgt, roll_deg=roll, up=g["up"])
        fovy = rng.uniform(35, 75)
        cands.append((score(pos, q, fovy), pos, tgt, q, fovy))
    cands.sort(key=lambda c: -c[0])
    print(f"{a.cam}: {len(sel)} frames; random {a.n} in {time.time() - t0:.0f}s; best "
          f"{[round(c[0], 3) for c in cands[:8]]}", flush=True)
    best = None
    for c in cands[:a.keep]:
        x0 = np.concatenate([c[1], np.zeros(3), [c[4]]])
        q0 = c[3]

        def f(x):
            q = F.A.quat_mul(q0, F.A.quat_from_rotvec(x[3:6] * 0.05))
            return -score(x[:3], q, x[6])

        es = cma.CMAEvolutionStrategy(x0, 0.02, dict(maxfevals=a.cma, verbose=-9, seed=1,
                                                     CMA_stds=[1, 1, 1, 1, 1, 1, 100]))
        es.optimize(f)
        xb = es.result.xbest
        sb = -es.result.fbest
        print(f"  cand {c[0]:.3f} -> {sb:.3f} pos {np.round(xb[:3], 3).tolist()} fovy {xb[6]:.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if best is None or sb > best[0]:
            best = (sb, xb[:3], F.A.quat_mul(q0, F.A.quat_from_rotvec(xb[3:6] * 0.05)), xb[6])
    out = dict(cam=a.cam, iou=best[0], pos=list(map(float, best[1])), quat=list(map(float, best[2])),
               fovy=float(best[3]), mask_tag=a.mask_tag, frames=sel.tolist())
    (HERE / "out" / f"seed_{a.cam}.json").write_text(json.dumps(out, indent=1))
    print("best", round(best[0], 3), out["pos"], out["fovy"])


if __name__ == "__main__":
    main()
