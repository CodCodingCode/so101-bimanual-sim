"""Wide random + CMA search of ONE world camera (pose + fovy) against rough masks, joint map fixed (no SAM needed).

    .venv/bin/python so101_twin/ballcup/cam_search.py top [--mask-tag _fg] [--n 3000] [--calib old] [--roll 0]

EVEN episodes only. Candidates: position around a geometric guess, looking at a random target in the work area,
random roll (top: any, side: +-15 deg) and fovy. Score = mean IoU at 160x120 over up to 36 frames chosen by
farthest-point sampling in joint space, inside the camera's VALID region (top: rows above the dark mat, where the
background-subtraction masks are reliable). Writes out/cam_search_<cam>.json.
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

GUESS = {"top": dict(pos=(0.0, 0.20, 0.60), pos_sd=(0.10, 0.12, 0.15), target=(0.0, 0.22, 0.0), target_sd=0.07,
                     roll_sd=None),
         "side": dict(pos=(0.0, 0.55, 0.15), pos_sd=(0.12, 0.12, 0.08), target=(0.0, 0.05, 0.12), target_sd=0.07,
                      roll_sd=12.0)}
VALID_ROWS = {"top": 85, "side": 120}   # at 160x120: top rows >= 85 are the dark mat (fg masks miss the arm there)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam")
    ap.add_argument("--mask-tag", default="_fg")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--calib", default="old")
    ap.add_argument("--roll", type=float, default=0.0, help="wrist-roll offset (deg)")
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--cma", type=int, default=300)
    a = ap.parse_args()
    sc = F.load_scene("ballcup")
    import dataclasses

    sc.ARM = dataclasses.replace(sc.ARM, calib=a.calib)
    P = F.init_params(sc)
    P["offsets_deg"][4] = a.roll
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    real = F.load_masks("ballcup", a.cam, a.mask_tag)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    small_all = real.reshape(fs.n, 120, 4, 160, 4).mean((2, 4)) >= 0.5
    valid = np.zeros((120, 160), bool)
    valid[:VALID_ROWS[a.cam]] = True
    area = (small_all & valid).reshape(fs.n, -1).mean(1)
    vis = np.nonzero((ep % 2 == 0) & (area > 0.01))[0]
    X = fs.state[:, :6] / 30.0
    sel = vis[F.fps_select(X[vis], min(a.frames, len(vis)))]
    small = small_all[sel] & valid
    states = [fs.state[i] for i in sel]
    cid = rig.arm.cam_id[a.cam]

    def score(pos, quat, fovy):
        rig.arm.m.cam_pos[cid] = pos
        rig.arm.m.cam_quat[cid] = quat
        rig.arm.m.cam_fovy[cid] = fovy
        v = []
        for s, r in zip(states, small):
            rig.arm.set_state(s)
            m = rig.arm.arm_mask(a.cam, 160, 120) & valid
            u = (m | r).sum()
            v.append((m & r).sum() / u if u else 0.0)
        return float(np.mean(v))

    g = GUESS[a.cam]
    rng = np.random.default_rng(0)
    t0 = time.time()
    cands = []
    for k in range(a.n):
        pos = np.asarray(g["pos"]) + rng.normal(0, 1, 3) * np.asarray(g["pos_sd"])
        tgt = np.asarray(g["target"]) + rng.normal(0, g["target_sd"], 3)
        roll = rng.uniform(-180, 180) if g["roll_sd"] is None else rng.normal(0, g["roll_sd"])
        q = F.A.look_quat(pos, tgt, roll_deg=roll, up=(0, 1, 0) if a.cam == "top" else (0, 0, 1))
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
    out = dict(cam=a.cam, iou_fg=best[0], pos=list(map(float, best[1])), quat=list(map(float, best[2])),
               fovy=float(best[3]), calib=a.calib, roll=a.roll, mask_tag=a.mask_tag, frames=sel.tolist())
    (HERE / "out" / f"cam_search_{a.cam}{a.suffix}.json").write_text(json.dumps(out, indent=1))
    print("best", round(best[0], 3), out["pos"], out["fovy"])


if __name__ == "__main__":
    main()
