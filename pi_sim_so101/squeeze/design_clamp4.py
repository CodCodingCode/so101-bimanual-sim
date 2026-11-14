#!/usr/bin/env python3
"""Clamp design v4: cube ANYWHERE in the reachable area, box at ONE fixed spot. The clamp settings (tilt, roll, tip
offsets, tip height, press) are tuned at anchor points across the area with the same rules as v3 (real joint ranges,
gentle squeeze, full plan into the box, zero contacts, safety-guard clearances incl. the wrist camera mounts) and
blended between anchors by clamp_teacher.configure(cube_xy).

    python squeeze/design_clamp4.py --evals 160
Writes squeeze/runs/design_clamp4_candidate.json (anchors in tuning order, each with params, score, ok, logs).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "rig_tools")]
import design_clamp3 as D3      # noqa: E402

CUBE_Y = 0.0372
BOX_XY = (0.0, -0.077)                       # fixed box centre (where the real place-in-box run worked)
# run 2: the centre already works with the v3 settings (fixed-box map); tune SIDEWAYS, small neighbourhoods,
# only the depth band the arms reach (25.7 .. 27.7 cm out)
AX = (3.0, -3.0, 5.0, -5.0, 7.0, -7.0)       # cm, nearest the solved centre first
AY = (1.0,)                                  # cm from the nominal cube y
NAMES = D3.NAMES[:8]                          # box_dy is not used (the box is fixed)


def scenes_for(ax, ay):
    x, y = ax / 100, CUBE_Y + ay / 100
    return ((x, y - CUBE_Y, 0, 0.0, 0.4), (x + 0.01, y - CUBE_Y + 0.01, 6, 0.0, 0.4),
            (x - 0.01, y - CUBE_Y - 0.01, -6, 0.0, 0.4), (x + 0.01, y - CUBE_Y - 0.01, 10, 0.0, 0.4),
            (x - 0.01, y - CUBE_Y + 0.01, -10, 0.0, 0.4), (x, y - CUBE_Y, 4, 0.0, 0.35))


def full(v):
    return {**dict(zip(NAMES, v)), "box_dy_mm": -100.0}


def _obj(args):
    x, scenes = args
    lo, hi = D3.LO[:8], D3.HI[:8]
    p = dict(zip(NAMES, lo + (hi - lo) * (np.tanh(np.asarray(x)) + 1) / 2))
    try:
        return -D3.evaluate(full([p[k] for k in NAMES]), scenes=scenes, box_xy=BOX_XY)[0]
    except Exception:
        return 3.0


def judge(p, scenes):
    D3.DETAILS.clear()
    score, logs = D3.evaluate(full([p[k] for k in NAMES]), scenes=scenes, box_xy=BOX_XY)
    import safety as SF
    ok = all(r["in_box"] and r["contacts"] == 0 and all(r["closest_mm"][w][k] >= SF.MARGIN[w][k] * 1000
                                                      for w in r["closest_mm"] for k in r["closest_mm"][w])
             for r in D3.DETAILS)
    return score, ok, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=160)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    import cma
    lo, hi = D3.LO[:8], D3.HI[:8]
    to_x = lambda v: np.arctanh(np.clip(2 * (np.asarray(v) - lo) / (hi - lo) - 1, -0.97, 0.97))
    v3 = json.loads((HERE / "runs" / "design_clamp3.json").read_text())["params"]
    order = [(x, y) for y in AY for x in AX]
    done, out = [], HERE / "runs" / "design_clamp4_candidate.json"
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for ax, ay in order:
            scenes = scenes_for(ax, ay)
            # warm start: nearest anchor already tuned on this side (else the v3 design)
            same = [r for r in done if np.sign(r["x"]) == np.sign(ax)]
            near = min(same, key=lambda r: (r["x"] - ax) ** 2 + (r["y"] - ay) ** 2) if same else None
            start = near["params"] if near else {k: v3[k] for k in NAMES}
            x0 = to_x([start[k] for k in NAMES])
            s0, ok0, _ = judge(start, scenes)
            best = (-s0, x0)
            if not ok0:
                es = cma.CMAEvolutionStrategy(x0, 0.3, {"maxfevals": a.evals, "popsize": 12, "seed": 7, "verbose": -9})
                while not es.stop():
                    xs = es.ask()
                    fs = pool.map(_obj, [(x, scenes) for x in xs])
                    es.tell(xs, fs)
                    for x, f in zip(xs, fs):
                        if f < best[0]:
                            best = (f, np.asarray(x))
            p = dict(zip(NAMES, (lo + (hi - lo) * (np.tanh(best[1]) + 1) / 2).tolist()))
            score, ok, logs = judge(p, scenes)
            done.append({"x": ax, "y": ay, "params": p, "score": score, "ok": ok, "warm_ok": ok0, "logs": logs})
            print(f"anchor x {ax:+.1f} cm, y {ay:+.1f} cm ({(CUBE_Y + ay / 100 + 0.22) * 100:.1f} cm out): "
                  f"{'OK' if ok else 'not clean'} score {score:.2f} ({'start already clean' if ok0 else 'tuned'}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            out.write_text(json.dumps({"box_xy": BOX_XY, "anchors": done}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
