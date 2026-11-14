#!/usr/bin/env python3
"""Where can the clamp v3 grab at all? Sweep cube positions over the table (grab + lift only, no box), with the real
joint ranges, fitted servos and the real-robot safety guard. Per spot: lifted, contacts, closest clearances (commanded
and measured), and whether the guard would stop it.

    python squeeze/map_clamp_region.py --step 2 --workers 6
Writes squeeze/runs/clamp_region_map.json and prints a grid (rows = cube y, cols = cube x).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "rig_tools")]


BOX_XY = None          # set by --fixed-box: full plan into the box at this spot


def probe(xy, box_xy=None):
    import mujoco
    import clamp_teacher as C
    import design_squeeze as DS
    import env as E
    import safety as SF
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    cx, cy = xy
    env = E.SqueezeEnv(episode_seconds=30.0)
    m, d = env.model, env.data
    env.reset(seed=1)
    for s in C.SIDES:
        env.sim.pad_quat[s] = C.clamp_quat(s, (cx, cy))
    cq, bq = m.jnt_qposadr[m.joint("cube_joint").id], m.jnt_qposadr[m.joint("can_joint").id]
    d.qpos[cq:cq + 7] = [cx, cy, C.CUBE_HALF + 0.001, 1, 0, 0, 0]
    d.qpos[bq:bq + 7] = [2.5, 2, -1, 1, 0, 0, 0] if box_xy is None else [box_xy[0], box_xy[1], E.CAN_HEIGHT / 2, 1, 0, 0, 0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(20):
        mujoco.mj_step(m, d)
    wide = np.full(5, 400.0)
    g = SF.Guard(m, {s: (-wide, wide) for s in SF.SIDES}, box_xy=box_xy, log=lambda *a: None)
    plan = C.ClampPlan((cx, cy, 0.0), box_xy, release=box_xy is not None)
    close = {w: {} for w in ("cmd", "meas")}
    bad, z, k, ikerr = [], 0.0, 0, 0.0
    while not plan.done:
        tg = plan.next_targets(env.sim._ee_targets.copy())
        a = np.zeros((2, 4))
        a[:, :3] = (tg - env.sim._ee_targets) / env.sim.delta_scale
        env.step(np.clip(a, -1, 1).ravel())
        bad += DS.bad_contacts(m, d)
        if plan.phase == "hold":
            z = max(z, env.cube_pos()[2] - C.CUBE_HALF)
        ikerr = max(ikerr, float(np.max(env.sim._ik_errors)) if hasattr(env.sim, "_ik_errors") else 0.0)
        if k % 3 == 0:
            for w, q in (("cmd", {s: np.r_[np.rad2deg(d.ctrl[env.sim.robot[s].actuator_ids][:5]), 0.0] for s in C.SIDES}),
                         ("meas", {s: np.r_[np.rad2deg(d.qpos[env.sim.robot[s].qpos_ids][:5]), 0.0] for s in C.SIDES})):
                for kind, (dist, _) in g.distances(q).items():
                    close[w][kind] = min(close[w].get(kind, 1.0), dist)
        k += 1
    in_box = False
    if box_xy is not None:
        for _ in range(40):
            env.step(np.zeros(8))
        in_box = bool(env.on_can())
    env.close()
    trip = [f"{w}:{kk}" for w in close for kk, v in close[w].items() if v < SF.MARGIN[w][kk]]
    return {"x": cx, "y": cy, "in_box": in_box, "lifted_mm": z * 1000, "contacts": len(bad), "kinds": sorted(set(bad)),
            "closest_mm": {w: {kk: round(v * 1000, 1) for kk, v in close[w].items()} for w in close},
            "guard_would_stop": trip, "ik_err_mm": round(ikerr * 1000, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=2.0, help="cm")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fixed-box", default=None, help="x,y in m: full plan into a box at this fixed spot")
    a = ap.parse_args()
    import clamp_teacher as C
    xs = np.arange(-10, 10.01, a.step) / 100
    ys = C.CUBE_Y + np.arange(-8, 12.01, a.step) / 100
    pts = [(float(x), float(y)) for y in ys for x in xs]
    box = tuple(float(v) for v in a.fixed_box.split(",")) if a.fixed_box else None
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.starmap(probe, [(p, box) for p in pts])
    (HERE / "runs" / ("clamp_region_map_fixedbox.json" if box else "clamp_region_map.json")).write_text(json.dumps(res, indent=1))
    print("cube x (cm) ->     " + " ".join(f"{x * 100:+5.0f}" for x in xs))
    print("legend: OK = lifted (and into the box), no contact, guard quiet | g = OK but the guard would stop | m = lifted, missed the box | c = contact | . = no lift | box = on the box")
    for y in ys[::-1]:
        row = []
        for x in xs:
            r = next(r for r in res if abs(r["x"] - x) < 1e-9 and abs(r["y"] - y) < 1e-9)
            if box is not None and abs(x - box[0]) < 0.066 and abs(y - box[1]) < 0.066:
                row.append("  box")
                continue
            done_ok = r["lifted_mm"] > 25 and (box is None or r["in_box"])
            row.append("   OK" if done_ok and not r["contacts"] and not r["guard_would_stop"] else
                       "    g" if done_ok and not r["contacts"] else
                       "    m" if r["lifted_mm"] > 25 and not r["contacts"] else
                       "    c" if r["contacts"] else "    .")
        print(f"y {(y - C.CUBE_Y) * 100:+5.0f} ({(y + 0.22) * 100:4.1f} cm out) " + " ".join(row))


if __name__ == "__main__":
    main()
