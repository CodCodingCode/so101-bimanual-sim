"""REDLEGO: global search of the wrist-camera bar (a rod + camera module on the gripper body, fit.py add_bar) against
the arm masks, every other parameter fixed at a fit json. EVEN episodes only, both cameras, 160x120 with the fitted
lens. Random bar poses (root within 6 cm of the gripper-body origin, uniform rotation, length 3-12 cm), then CMA on the
best few; the score is compared with the bar collapsed to nothing (length 5 mm inside the gripper).

    .venv/bin/python so101_twin/redlego/bar_search.py --fit out/<run>_fit.json --mask-tag _d1k --out out/bar.json
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

W, H = 160, 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--mask-tag", default="_d1k")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--cma", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sc = F.load_scene("redlego")
    fj = json.loads(pathlib.Path(a.fit).read_text())
    sc.ARM = dataclasses.replace(sc.ARM, model=fj.get("model", sc.ARM.model))
    sc.BAR = True
    P0 = F.reconcile(sc, fj["params"])
    rig = F.Rig(sc, P0)
    fs = F.real_set(sc)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    X = fs.state[:, :6] / 30.0
    data = {}
    for c in sc.CAMERAS:
        real = F.load_masks("redlego", c, a.mask_tag)
        small = real.reshape(fs.n, H, 480 // H, W, 640 // W).mean((2, 4)) >= 0.5
        vis = np.nonzero((ep % 2 == 0) & (small.reshape(fs.n, -1).mean(1) > 0.005))[0]
        sel = vis[F.fps_select(X[vis], min(a.frames, len(vis)), seed=3)]
        data[c] = (sel, small[sel], [F.frame_state(sc, P0, c, int(i)) for i in sel])

    def score(b):
        P = dict(P0)
        P["bar"] = list(map(float, b))
        rig.set_params(P)
        v = []
        for c, (sel, real, states) in data.items():
            for r, st in zip(real, states):
                m = rig.mask(c, "all", st, W, H)
                u = (m | r).sum()
                v.append((m & r).sum() / u if u else 0.0)
        return float(np.mean(v))

    base = score([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005])
    print(f"no bar: {base:.4f}", flush=True)
    rng = np.random.default_rng(0)
    t0 = time.time()
    cands = []
    for k in range(a.n):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        rv = F._rotvec(q)
        pos = rng.uniform(-0.06, 0.06, 3)
        L = rng.uniform(0.03, 0.12)
        b = np.r_[pos, rv, L]
        cands.append((score(b), b))
    cands.sort(key=lambda c: -c[0])
    print(f"random {a.n} in {time.time() - t0:.0f}s; best {[round(c[0], 4) for c in cands[:8]]}", flush=True)
    best = None
    for s0, b0 in cands[:a.keep]:
        def f(x, b0=b0):
            b = b0.copy()
            b[:3] += 0.01 * x[:3]
            b[3:6] = F._rotvec(F.A.quat_mul(F.A.quat_from_rotvec(b0[3:6]), F.A.quat_from_rotvec(0.1 * x[3:6])))
            b[6] += 0.01 * x[6]
            return b

        es = cma.CMAEvolutionStrategy(np.zeros(7), 0.5, dict(maxfevals=a.cma, verbose=-9, seed=1))
        while not es.stop():
            xs = es.ask()
            es.tell(xs, [-score(f(x)) for x in xs])
        bb = f(es.result.xbest)
        sb = score(bb)
        print(f"  cand {s0:.4f} -> {sb:.4f} bar {np.round(bb, 4).tolist()} ({time.time() - t0:.0f}s)", flush=True)
        if best is None or sb > best[0]:
            best = (sb, bb)
    out = dict(fit=a.fit, mask_tag=a.mask_tag, base=base, score=best[0], bar=list(map(float, best[1])))
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"best {best[0]:.4f} (no bar {base:.4f}) bar {np.round(best[1], 4).tolist()}", flush=True)


if __name__ == "__main__":
    main()
