#!/usr/bin/env python3
"""Clamp design v2: separate LEFT / RIGHT gripper settings, scored on the exact open-loop ClampPlan the real arms
run (no touch sensing), over 6 cube placements (offsets up to 1.5 cm, yaw up to +-10 deg).

Per arm: tilt in its arm plane, roll about the finger axis, tip y offset. Shared: tip height, final tip gap
(how far past the cube faces the commanded tips go). Realistic jaw shapes (CoACD), friction 0.4, fitted servos.
Score per placement: lifted (<=4 cm), level (tilt <= 8 deg), and NO contact between the arms (incl. wrist camera
mounts), with the camera post / head, the box, or the table (forearm / hand).

    ~/.pyenv/versions/3.10.9/bin/python squeeze/design_clamp2.py --evals 800
Writes squeeze/runs/design_clamp2.json
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
sys.path[:0] = [str(HERE), str(HERE.parent)]

NAMES = ("tilt_l", "tilt_r", "roll_l", "roll_r", "dy_l_mm", "dy_r_mm", "tip_mm", "gap_mm")
LO = np.array([30.0, 30.0, -135.0, -135.0, -20.0, -20.0, 8.0, -4.0])
HI = np.array([60.0, 60.0, -45.0, -45.0, 20.0, 20.0, 26.0, 12.0])
X0 = np.array([49.2, 49.2, -89.7, -89.7, -8.8, -8.8, 15.6, 6.0])
SCENES = ((0.0, 0.0, 0), (0.01, 0.0, 0), (-0.01, 0.01, 5), (0.0, -0.01, -5), (0.005, 0.0, 8), (-0.015, -0.015, -10))


def to_x(v):
    return np.arctanh(np.clip(2 * (np.asarray(v) - LO) / (HI - LO) - 1, -0.97, 0.97))


def unpack(x):
    return dict(zip(NAMES, LO + (HI - LO) * (np.tanh(np.asarray(x)) + 1) / 2))


def evaluate(p, scenes=SCENES, verbose=False):
    import mujoco
    import clamp_teacher as C
    import design_squeeze as DS
    import env as E
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    base_quat = C.clamp_quat

    def quat(side, cube_xy):
        C.TILT = np.radians(p["tilt_l"] if side == "left" else p["tilt_r"])
        C.ROLL = np.radians(p["roll_l"] if side == "left" else p["roll_r"])
        return base_quat(side, cube_xy)

    C.TIP_Z = p["tip_mm"] / 1000
    C.TIP_DY = 0.0
    C.SQUEEZE = (C.CUBE_HALF - 0.004) - p["gap_mm"] / 1000
    total, logs = 0.0, []
    for dx, dy, yaw in scenes:
        cx, cy = dx, C.CUBE_Y + dy
        E.pad_quaternion = lambda side, cx=cx, cy=cy: quat(side, (cx, cy))
        env = E.SqueezeEnv()
        m, d = env.model, env.data
        env.reset(seed=1)
        cq = m.jnt_qposadr[m.joint("cube_joint").id]
        bq = m.jnt_qposadr[m.joint("can_joint").id]
        y = np.radians(yaw)
        d.qpos[cq:cq + 7] = [cx, cy, C.CUBE_HALF + 0.001, np.cos(y / 2), 0, 0, np.sin(y / 2)]
        d.qpos[bq:bq + 3] = [0.5, 0.4, 0.015]
        mujoco.mj_forward(m, d)
        plan = C.ClampPlan((cx, cy, 0))
        off = np.array([[0.0, p["dy_l_mm"] / 1000, 0.0], [0.0, p["dy_r_mm"] / 1000, 0.0]])
        for i, (n, l, r, k) in enumerate(plan.waypoints):
            plan.waypoints[i] = (n, l + off[0], r + off[1], k)
        bad, zmax, tilt = [], 0.0, 0.0
        while not plan.done:
            tg = plan.next_targets(env.sim._ee_targets.copy())
            a = np.zeros((2, 4))
            a[:, :3] = (tg - env.sim._ee_targets) / env.sim.delta_scale
            env.step(np.clip(a, -1, 1).ravel())
            bad += DS.bad_contacts(m, d)
            if plan.phase == "hold":
                zmax = max(zmax, env.cube_pos()[2] - C.CUBE_HALF)
                Rc = d.xmat[m.body("cube").id].reshape(3, 3)
                tilt = max(tilt, np.degrees(np.arccos(np.clip(Rc[2, 2], -1, 1))))
        env.close()
        s = min(zmax, 0.04) / 0.04 - 0.03 * max(0.0, tilt - 8.0) - (0.5 if bad else 0.0) - 0.004 * min(len(bad), 150)
        total += s
        kinds = {k: bad.count(k) for k in set(bad)}
        logs.append(f"cube {dx * 100:+.1f},{dy * 100:+.1f} cm yaw {yaw:+3d}: lifted {zmax * 1000:4.1f} mm, tilt {tilt:4.1f}, "
                    f"bad {kinds or 'none'}")
    if verbose:
        print("\n".join(logs))
    return total / len(scenes), logs


def _obj(x):
    try:
        return -evaluate(unpack(x))[0]
    except Exception:
        return 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=800)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import cma
    es = cma.CMAEvolutionStrategy(to_x(X0), 0.35, {"maxfevals": a.evals, "popsize": 16, "seed": 11, "verbose": -9})
    best, t0 = (9.0, to_x(X0)), time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        while not es.stop():
            xs = es.ask()
            fs = pool.map(_obj, xs)
            es.tell(xs, fs)
            for x, f in zip(xs, fs):
                if f < best[0]:
                    best = (f, np.asarray(x))
            if es.countiter % 5 == 0:
                print(f"it {es.countiter:3d} ({time.time() - t0:4.0f}s) best {-best[0]:.2f} | "
                      + " ".join(f"{k} {v:.1f}" for k, v in unpack(best[1]).items()), flush=True)
    p = unpack(best[1])
    score, logs = evaluate(p, verbose=True)
    (HERE / "runs" / "design_clamp2.json").write_text(json.dumps({"params": p, "score": score, "logs": logs}, indent=1))
    print(json.dumps({"params": p, "score": score}, indent=1))


if __name__ == "__main__":
    main()
