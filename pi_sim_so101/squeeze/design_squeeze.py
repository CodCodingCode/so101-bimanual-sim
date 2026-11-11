#!/usr/bin/env python3
"""Search for a two-arm TIP squeeze that lifts the cube with the REAL jaw shapes (CoACD collision) and realistic
plastic friction (PI_SIM_PAD_FRICTION, default 0.4). No jaws: grippers stay shut; the tips press the cube's sides.

Parameters (CMA-ES): tilt of each gripper inside its arm's reaching plane (the only tilt a 5-DoF SO-101 has), roll
about the finger axis, tip height above the cube's bottom, tip y offset, squeeze depth, extra inward squeeze while
lifting, cube distance in front of the bases. Objective: cube height after lifting 4 cm and holding 1 s, minus a
penalty for sloped contacts; two cube yaws per evaluation.

    ~/.pyenv/versions/3.10.9/bin/python squeeze/design_squeeze.py --evals 600
Writes squeeze/runs/design_squeeze.json.
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

NAMES = ("tilt_deg", "roll_deg", "tip_mm", "tip_dy_mm", "squeeze_mm", "lift_squeeze_mm", "cube_y_mm")
LO = np.array([0.0, -90.0, 6.0, -15.0, 2.0, 0.0, -60.0])
HI = np.array([55.0, 180.0, 30.0, 15.0, 30.0, 15.0, 40.0])
ARM_PARTS_ON_TABLE = ("lower_arm", "wrist", "gripper", "moving_jaw_so101_v1", "camera_mount")


def bad_contacts(m, d):
    """Contacts the clamp must never make: arm-arm, arm-camera post/head, arm-box, forearm/hand on the table."""
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
        g1, g2 = m.geom(c.geom1).name, m.geom(c.geom2).name
        arm1, arm2 = b1.startswith(("left_", "right_")), b2.startswith(("left_", "right_"))
        if arm1 and arm2 and b1.split("_")[0] != b2.split("_")[0]:
            out.append("arm-arm")
        elif (arm1 and g2.startswith("mount_")) or (arm2 and g1.startswith("mount_")):
            out.append("arm-camera")
        elif (arm1 and b2 == "can") or (arm2 and b1 == "can"):
            out.append("arm-box")
        elif (arm1 and g2 == "table" and b1.split("_", 1)[1] in ARM_PARTS_ON_TABLE) or \
             (arm2 and g1 == "table" and b2.split("_", 1)[1] in ARM_PARTS_ON_TABLE):
            out.append("arm-table")
    return out


def unpack(x):
    return dict(zip(NAMES, LO + (HI - LO) * (np.tanh(np.asarray(x)) + 1) / 2))


def evaluate(p, yaws=(0.0, 6.0, -6.0, 10.0), verbose=False):
    import mujoco
    import env as E
    theta, roll = np.radians(p["tilt_deg"]), np.radians(p["roll_deg"])
    cube_y = p["cube_y_mm"] / 1000.0

    def h_dir(side):
        bx = -0.116 if side == "left" else 0.116
        v = np.array([0.0 - bx, cube_y + 0.22])
        return v / np.linalg.norm(v)

    def tip_quat(side):
        h = np.r_[h_dir(side), 0.0]
        approach = np.array([0, 0, -1.0]) * np.cos(theta) + h * np.sin(theta)
        side_ax = np.cross(approach, [0, 0, 1.0]); side_ax /= np.linalg.norm(side_ax)
        up_ax = np.cross(side_ax, approach)
        close = np.cos(roll) * side_ax + np.sin(roll) * up_ax
        R = np.column_stack([approach, np.cross(close, approach), close])
        q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel())
        return q

    E.pad_quaternion = tip_quat
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    env = E.SqueezeEnv()
    m, d = env.model, env.data
    cg = m.geom("cube_geom").id
    CH = E.CUBE_HALF
    score, logs = 0.0, []
    for yaw in yaws:
        bad = []
        env.reset(seed=1)
        cq = m.jnt_qposadr[m.joint("cube_joint").id]
        bq = m.jnt_qposadr[m.joint("can_joint").id]
        y = np.radians(yaw)
        d.qpos[cq:cq + 7] = [0, cube_y, CH + 0.001, np.cos(y / 2), 0, 0, np.sin(y / 2)]
        d.qpos[bq:bq + 3] = [0.5, 0.4, 0.015]
        mujoco.mj_forward(m, d)

        def go(tgts, steps, until_touch=False):
            for _ in range(steps):
                bad.extend(bad_contacts(m, d))
                ee = env.sim._ee_targets
                a = np.zeros((2, 4))
                for i in range(2):
                    a[i, :3] = np.clip((np.asarray(tgts[i]) - ee[i]) / env.sim.delta_scale, -0.3, 0.3)
                env.step(a.ravel())
                if until_touch and env.touching("left") and env.touching("right"):
                    return True
            return False

        z = p["tip_mm"] / 1000.0
        dy = p["tip_dy_mm"] / 1000.0
        out = CH + 0.035
        go([[-out, cube_y + dy, z + 0.05], [out, cube_y + dy, z + 0.05]], 50)
        go([[-out, cube_y + dy, z], [out, cube_y + dy, z]], 40)
        touched = go([[-CH + 0.004, cube_y + dy, z], [CH - 0.004, cube_y + dy, z]], 90, until_touch=True)
        if not touched:
            logs.append(f"yaw {yaw}: no two-sided contact")
            score -= 1.0                       # missing the cube is a failure, not a neutral outcome
            continue
        ee = env.sim._ee_targets.copy()
        sq = p["squeeze_mm"] / 1000.0
        ee[0, 0] += sq
        ee[1, 0] -= sq
        go(ee, 15)
        nz = []
        cp = env.cube_pos()
        for i in range(d.ncon):
            c = d.contact[i]
            if cg in (c.geom1, c.geom2) and m.geom_bodyid[c.geom2 if c.geom1 == cg else c.geom1] != 0:
                nz.append(abs(c.frame[2]))
        extra = p["lift_squeeze_mm"] / 1000.0
        lift = ee.copy()
        lift[:, 2] += 0.04
        lift[0, 0] += extra
        lift[1, 0] -= extra
        go(lift, 45)
        tilts, cz = [], []
        for _ in range(25):
            go(lift, 1)
            Rc = d.xmat[m.body("cube").id].reshape(3, 3)
            tilts.append(np.degrees(np.arccos(np.clip(Rc[2, 2], -1, 1))))
            cp = env.cube_pos()
            for i in range(d.ncon):
                c = d.contact[i]
                if cg in (c.geom1, c.geom2) and m.geom_bodyid[c.geom2 if c.geom1 == cg else c.geom1] != 0:
                    cz.append((Rc.T @ (c.pos - cp))[2] * 1000)
        h = env.cube_pos()[2] - CH
        slope = float(np.mean(nz)) if nz else 1.0
        tilt = float(np.max(tilts))
        zc = float(np.median(cz)) if cz else -99.0
        n_bad = len(bad)
        score += (min(h, 0.04) / 0.04 - 0.3 * slope - 0.03 * max(0.0, tilt - 5.0) - 0.01 * max(0.0, -2.0 - zc)
                  - (0.5 if n_bad else 0.0) - 0.01 * min(n_bad, 100))
        kinds = {k: bad.count(k) for k in set(bad)}
        logs.append(f"yaw {yaw}: lifted {h * 1000:5.1f} mm, contact |n_z| {slope:.2f}, cube tilt max {tilt:4.1f} deg, "
                    f"contact height {zc:+.0f} mm vs centre, bad contacts {kinds or 'none'}")
    env.close()
    if verbose:
        print("\n".join(logs))
    return score / len(yaws), logs


def _obj(x):
    try:
        return -evaluate(unpack(x))[0]
    except Exception:
        return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--init", default=None, help="warm start from a previous design json")
    ap.add_argument("--sigma", type=float, default=0.8)
    a = ap.parse_args()
    import cma
    x0 = np.zeros(len(NAMES))
    if a.init:
        p0 = json.loads(Path(a.init).read_text())["params"]
        v = np.array([p0[k] for k in NAMES])
        x0 = np.arctanh(np.clip(2 * (v - LO) / (HI - LO) - 1, -0.95, 0.95))
    es = cma.CMAEvolutionStrategy(x0, a.sigma, {"maxfevals": a.evals, "popsize": 16, "seed": 7, "verbose": -9})
    best, t0 = (1.0, None), time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        while not es.stop():
            xs = es.ask()
            fs = pool.map(_obj, xs)
            es.tell(xs, fs)
            for x, f in zip(xs, fs):
                if f < best[0]:
                    best = (f, np.asarray(x))
            p = unpack(best[1])
            print(f"it {es.countiter:3d} ({time.time() - t0:4.0f}s) best score {-best[0]:.2f} | "
                  + " ".join(f"{k} {v:.1f}" for k, v in p.items()), flush=True)
    p = unpack(best[1])
    score, logs = evaluate(p, yaws=(0.0, 5.0, 10.0, -8.0), verbose=True)
    out = {"params": p, "score_4yaws": score, "logs": logs, "friction": __import__("os").environ.get("PI_SIM_PAD_FRICTION", "0.4")}
    (HERE / "runs" / "design_squeeze.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
