"""Gripper calibration grid (even episodes only): grip_scale x grip_off (rad) on 64 even-episode frames per camera,
mean soft-free IoU of both cameras at 320 x 240; writes the fit json with the best pair as <out> (params only), to
start a stage-C/D polish from (the CMA in stage C moves the gripper too little from a far start).

    .venv/bin/python so101_twin/pos5/grip_grid.py out/pos5_r2_fit.json --masks _c3 --out out/pos5_r2_grip.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import fit as F
import p5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit")
    ap.add_argument("--masks", default="_c3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    sc = p5.scene()
    fj = json.loads((p5.HERE / a.fit).read_text())
    if a.model or fj.get("model"):
        import dataclasses

        sc.ARM = dataclasses.replace(sc.ARM, model=a.model or fj["model"])
    P = F.reconcile(sc, fj["params"])
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(np.flatnonzero(ep % 2 == 0), 64, replace=False))
    R = {c: F.down2(p5.load(c, a.masks)[idx]) for c in p5.CAMS}

    def score(Q):
        rig.set_params(Q)
        v = []
        for c in p5.CAMS:
            sim = np.stack([rig.mask(c, "all", F.frame_state(sc, Q, c, int(i))) for i in idx])
            v.append(float(np.nanmean(F.iou_frames(sim, R[c]))))
        return v

    best = None
    g0, o0 = P.get("grip_scale", 1.0), P.get("grip_off", 0.0)
    for gs in np.round(np.arange(0.40, 1.01, 0.075), 3):
        for go in np.round(np.arange(-0.30, 0.31, 0.075), 3):
            Q = json.loads(json.dumps(P))
            Q["grip_scale"], Q["grip_off"] = float(gs), float(go)
            v = score(Q)
            if best is None or np.mean(v) > np.mean(best[0]):
                best = (v, float(gs), float(go))
    Q = json.loads(json.dumps(P))
    v0 = score(Q)
    print(f"start grip_scale {g0:.3f} grip_off {o0:.3f}: top {v0[0]:.4f} side {v0[1]:.4f} mean {np.mean(v0):.4f}")
    print(f"best  grip_scale {best[1]:.3f} grip_off {best[2]:.3f}: top {best[0][0]:.4f} side {best[0][1]:.4f} "
          f"mean {np.mean(best[0]):.4f}")
    if np.mean(best[0]) > np.mean(v0):
        Q["grip_scale"], Q["grip_off"] = best[1], best[2]
    (p5.HERE / a.out).write_text(json.dumps(dict(params=Q, model=sc.ARM.model, source=a.fit, grip_grid=dict(
        start=[g0, o0, v0], best=[best[1], best[2], best[0]])), indent=1))
    print("wrote", p5.HERE / a.out)


if __name__ == "__main__":
    main()
