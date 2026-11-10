"""REDLEGO copy of jar/seed.py (GUESS for this scene, default masks _fg). JAR: twin-free camera seed for ONE world camera (pose + fovy) against twin-independent masks (default _d1c),
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

GUESS = {"top": dict(pos=(0.0, 0.70, 0.45), pos_sd=(0.15, 0.15, 0.15), target=(0.0, 0.10, 0.05), target_sd=0.08,
                     roll_sd=10.0, up=(0, 0, 1)),
         "side": dict(pos=(0.55, 0.10, 0.25), pos_sd=(0.15, 0.15, 0.10), target=(0.0, 0.12, 0.10), target_sd=0.08,
                      roll_sd=10.0, up=(0, 0, 1)),
         # REDLEGO r2 of the side seed: the image-right direction is ~(-0.7, 0.7) (pan +60 -> image left, pan -5 / -48 ->
         # image right), so the camera sits in the +x+y quadrant looking at the base
         "wide": dict(pos=(0, 0, 0), pos_sd=(0, 0, 0), target=(0.0, 0.05, 0.12), target_sd=0.07, roll_sd=8.0,
                      up=(0, 0, 1), wide=True),
         "side2": dict(pos=(0.40, 0.40, 0.20), pos_sd=(0.15, 0.15, 0.10), target=(0.0, 0.03, 0.12), target_sd=0.06,
                       roll_sd=8.0, up=(0, 0, 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam")
    ap.add_argument("--mask-tag", default="_fg")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--cma", type=int, default=400)
    ap.add_argument("--guess", default="", help="REDLEGO: GUESS key (default: the camera name)")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    sc = F.load_scene("redlego")
    P = F.init_params(sc)
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    real = F.load_masks("redlego", a.cam, a.mask_tag)
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

    g = GUESS[a.guess or a.cam]
    rng = np.random.default_rng(0)
    t0 = time.time()
    cands = []
    for k in range(a.n):
        pos = np.asarray(g["pos"]) + rng.normal(0, 1, 3) * np.asarray(g["pos_sd"])
        if g.get("wide"):   # REDLEGO: any azimuth, distance 0.3-0.9 m from the base, height 0.05-0.45 m
            az, rr = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 0.9)
            pos = np.array([rr * np.cos(az), rr * np.sin(az), rng.uniform(0.05, 0.45)])
        if a.cam == "side":
            pos[2] = max(pos[2], 0.05)
        tgt = np.asarray(g["target"]) + rng.normal(0, g["target_sd"], 3)
        roll = rng.uniform(-180, 180) if g["roll_sd"] is None else rng.normal(0, g["roll_sd"])
        q = F.A.look_quat(pos, tgt, roll_deg=roll, up=g["up"])
        fovy = rng.uniform(35, 80)
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
    (HERE / "out" / f"seed_{a.cam}{a.suffix}.json").write_text(json.dumps(out, indent=1))
    print("best", round(best[0], 3), out["pos"], out["fovy"])


if __name__ == "__main__":
    main()
