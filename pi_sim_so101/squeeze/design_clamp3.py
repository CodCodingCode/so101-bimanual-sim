#!/usr/bin/env python3
"""Clamp design v3 (after a real run knocked a wrist camera holder off). Differences from v2:
  - the sim joints / IK are limited to each REAL servo's range (env.apply_real_ranges)
  - GENTLE squeeze: tips close to 2 mm outside the faces, then press only press_mm into each face (v2 aimed both tool
    points 0.4 mm from the centre line, i.e. through each other)
  - the FULL plan with the box in place (v2 parked the box 50 cm away and stopped after the lift)
  - scored with the real-robot safety guard (rig_tools/safety.py, exact convex distances): clearance to the other arm
    (incl. the wrist camera mounts), the overhead camera post / head and the box, in the commanded AND measured pose,
    must stay above targets that sit above the guard's own trip margins

    ~/.pyenv/versions/3.10.9/bin/python squeeze/design_clamp3.py --evals 480
Writes squeeze/runs/design_clamp3_candidate.json; promote it to design_clamp3.json (which clamp_teacher.py loads in preference to v2) only after it passes the safety preflight.
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

NAMES = ("tilt_l", "tilt_r", "roll_l", "roll_r", "dy_l_mm", "dy_r_mm", "tip_mm", "press_mm", "box_dy_mm")
# roll covers the whole circle (run 1 only searched -170..-10 and missed the flipped grippers, whose wrist camera
# mount sits on the OUTER side); box_dy: how far the box sits from the cube toward the arms
LO = np.array([5.0, 5.0, -200.0, -200.0, -35.0, -35.0, 8.0, 1.5, -140.0])
HI = np.array([60.0, 60.0, 200.0, 200.0, 35.0, 35.0, 30.0, 8.0, -85.0])
X0 = np.array([32.8, 30.2, 82.5, -112.4, -8.8, -4.0, 14.9, 3.0, -100.0])      # v2 with the LEFT gripper rolled 180
# (cube dx, cube dy, yaw deg, extra box offset toward the arms, pad friction)
SCENES = ((0.0, 0.0, 0, 0.0, 0.4), (0.015, 0.01, 6, 0.005, 0.4), (-0.015, -0.01, -6, 0.0, 0.4),
          (0.01, -0.015, 10, 0.01, 0.4), (-0.01, 0.015, -10, 0.005, 0.4), (0.0, 0.0, 4, 0.0, 0.35),
          (0.02, -0.02, 8, 0.0, 0.4), (-0.02, 0.02, -8, 0.01, 0.4))
# designed clearance (m), above the guard's trip margins (safety.MARGIN) so the guard never has to fire
TARGET = {"cmd": {"camera-mount": 0.018, "arm-arm": 0.003, "arm-camera": 0.025, "arm-box": 0.012},
          "meas": {"camera-mount": 0.012, "arm-arm": 0.006, "arm-camera": 0.020, "arm-box": 0.008}}


DETAILS = []
# optional overrides through the environment (read in every worker): a fixed box spot and a custom scene list
import os as _os
FIXED_BOX = tuple(json.loads(_os.environ["DESIGN_FIXED_BOX"])) if _os.environ.get("DESIGN_FIXED_BOX") else None
if _os.environ.get("DESIGN_SCENES"):
    SCENES = tuple(tuple(x) for x in json.loads(_os.environ["DESIGN_SCENES"]))


def to_x(v):
    return np.arctanh(np.clip(2 * (np.asarray(v) - LO) / (HI - LO) - 1, -0.97, 0.97))


def unpack(x):
    return dict(zip(NAMES, LO + (HI - LO) * (np.tanh(np.asarray(x)) + 1) / 2))


def evaluate(p, scenes=SCENES, verbose=False, box_xy=None):
    import mujoco
    import clamp_teacher as C
    import design_squeeze as DS
    import env as E
    import safety as SF
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    C.TILT = {"left": np.radians(p["tilt_l"]), "right": np.radians(p["tilt_r"])}
    C.ROLL = {"left": np.radians(p["roll_l"]), "right": np.radians(p["roll_r"])}
    C.TIP_DY = {"left": p["dy_l_mm"] / 1000, "right": p["dy_r_mm"] / 1000}
    C.TIP_Z = p["tip_mm"] / 1000
    C.DROP_Z = C.TIP_Z + 0.035
    C.GAP = C.CUBE_HALF - p["press_mm"] / 1000
    C.CLOSE = C.CUBE_HALF + 0.002
    C.BOX_DY = p["box_dy_mm"] / 1000
    wide = np.full(5, 400.0)
    total, logs = 0.0, []
    for dx, dy, yaw, box_back, fric in scenes:
        cx, cy = dx, C.CUBE_Y + dy
        box_xy = box_xy if box_xy is not None else FIXED_BOX
        bx, by = (cx, cy + C.BOX_DY - 0.005 - box_back) if box_xy is None else box_xy
        env = E.SqueezeEnv(episode_seconds=20.0)
        m, d = env.model, env.data
        env.reset(seed=1)
        pads = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and "coacd" in m.mesh(m.geom_dataid[g]).name]
        m.geom_friction[pads, 0] = fric
        for s in C.SIDES:
            env.sim.pad_quat[s] = C.clamp_quat(s, (cx, cy))
        cq, bq = m.jnt_qposadr[m.joint("cube_joint").id], m.jnt_qposadr[m.joint("can_joint").id]
        y = np.radians(yaw)
        d.qpos[cq:cq + 7] = [cx, cy, C.CUBE_HALF + 0.001, np.cos(y / 2), 0, 0, np.sin(y / 2)]
        d.qpos[bq:bq + 7] = [bx, by, E.CAN_HEIGHT / 2, 1, 0, 0, 0]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        for _ in range(20):
            mujoco.mj_step(m, d)
        guard = SF.Guard(m, {s: (-wide, wide) for s in SF.SIDES}, box_xy=(bx, by), log=lambda *a: None)
        plan = C.ClampPlan((cx, cy, 0.0), (bx, by))
        bad, zmax, tilt, k = [], 0.0, 0.0, 0
        close = {w: {kk: 0.05 for kk in TARGET[w]} for w in TARGET}
        where = {w: {} for w in TARGET}
        lo_lim = {s: np.rad2deg(m.jnt_range[[m.joint(f"{s}_{j}").id for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")], 0]) for s in C.SIDES}
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
            if k % 2 == 0:
                poses = {"cmd": {s: np.r_[np.rad2deg(d.ctrl[env.sim.robot[s].actuator_ids][:5]), 0.0] for s in C.SIDES},
                         "meas": {s: np.r_[np.rad2deg(d.qpos[env.sim.robot[s].qpos_ids][:5]), 0.0] for s in C.SIDES}}
                for w, q in poses.items():
                    for kind, (dist, pair) in guard.distances(q).items():
                        if dist < close[w][kind]:
                            close[w][kind], where[w][kind] = dist, (plan.phase, pair)
            k += 1
        for _ in range(40):                                  # let the cube settle in the box
            env.step(np.zeros(8))
        in_box = bool(env.on_can())
        env.close()
        pen_clear = sum(max(0.0, TARGET[w][kind] - close[w][kind]) / TARGET[w][kind] for w in TARGET for kind in TARGET[w])
        s = (0.4 * min(zmax, 0.04) / 0.04 + 0.6 * in_box - 0.03 * max(0.0, tilt - 8.0)
             - (0.5 if bad else 0.0) - 0.004 * min(len(bad), 150) - 0.4 * pen_clear)
        total += s
        DETAILS.append({"scene": (dx, dy, yaw, box_back, fric), "lifted_mm": zmax * 1000, "in_box": in_box,
                        "contacts": len(bad), "closest_mm": {w: {kk: close[w][kk] * 1000 for kk in close[w]} for w in close}})
        kinds = {kk: bad.count(kk) for kk in set(bad)}
        tight = {f"{w}:{kind}": f"{close[w][kind] * 1000:.0f}mm@{where[w][kind][0]}" for w in TARGET for kind in TARGET[w]
                 if close[w][kind] < TARGET[w][kind]}
        logs.append(f"cube {dx * 100:+.1f},{dy * 100:+.1f} cm yaw {yaw:+3d} mu {fric}: lifted {zmax * 1000:4.1f} mm, tilt {tilt:4.1f}, "
                    f"in box {in_box}, contacts {kinds or 'none'}, under clearance target {tight or 'none'} | closest "
                    + " ".join(f"{w}:{kind} {close[w][kind] * 1000:.0f}" for w in TARGET for kind in TARGET[w]))
    if verbose:
        print("\n".join(logs), flush=True)
    return total / len(scenes), logs


def _obj(x):
    try:
        return -evaluate(unpack(x))[0]
    except Exception:
        return 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=480)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--baseline", action="store_true", help="only score the starting point (v2 + gentle squeeze)")
    ap.add_argument("--start", default=None, help="json with params to start from (refinement)")
    ap.add_argument("--sigma", type=float, default=0.4)
    ap.add_argument("--out", default="design_clamp3_candidate.json")
    a = ap.parse_args()
    if a.baseline:
        sc, _ = evaluate(dict(zip(NAMES, X0)), verbose=True)
        print(f"baseline score {sc:.3f}")
        return
    import cma
    x0 = X0 if a.start is None else np.array([json.loads(Path(a.start).read_text())["params"][k] for k in NAMES])
    es = cma.CMAEvolutionStrategy(to_x(x0), a.sigma, {"maxfevals": a.evals, "popsize": 16, "seed": 13, "verbose": -9})
    best, t0 = (9.0, to_x(x0)), time.time()
    # maxtasksperchild: workers build a MuJoCo scene per placement and keep the memory; recycle them (user's Mac)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        while not es.stop():
            xs = es.ask()
            fs = pool.map(_obj, xs)
            es.tell(xs, fs)
            for x, f in zip(xs, fs):
                if f < best[0]:
                    best = (f, np.asarray(x))
            print(f"it {es.countiter:3d} ({time.time() - t0:4.0f}s) best {-best[0]:.3f} | "
                  + " ".join(f"{k} {v:.1f}" for k, v in unpack(best[1]).items()), flush=True)
            p = unpack(best[1])
            (HERE / "runs" / "design_clamp3_progress.json").write_text(json.dumps({"params": p, "score": -best[0]}, indent=1))
    p = unpack(best[1])
    score, logs = evaluate(p, verbose=True)
    (HERE / "runs" / a.out).write_text(json.dumps({"params": p, "score": score, "logs": logs,
                                                                  "targets_m": TARGET}, indent=1))
    print(json.dumps({"params": p, "score": score}, indent=1))


if __name__ == "__main__":
    main()
