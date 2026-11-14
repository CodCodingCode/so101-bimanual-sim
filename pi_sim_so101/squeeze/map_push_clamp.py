#!/usr/bin/env python3
"""End to end in sim, cube ANYWHERE: push it into the clamp's proven zone (push_teacher.py, up to 3 pushes, cube
position re-read after each), then clamp v3 + place into the box at its fixed spot. Real joint ranges, fitted servos,
the real-robot safety guard checked every few ticks (commanded + measured pose).

    python squeeze/map_push_clamp.py --step 2.5 --workers 12
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
BOX_XY = (0.0, -0.077)
CLOSED_LOOP = True


def run_case(xy, yaw_deg=0.0, max_pushes=4):
    import mujoco
    import clamp_teacher as C
    import design_squeeze as DS
    import env as E
    import push_teacher as PT
    import safety as SF
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    env = E.SqueezeEnv(episode_seconds=120.0)
    m, d, sim = env.model, env.data, env.sim
    env.reset(seed=1)
    cq, bq = m.jnt_qposadr[m.joint("cube_joint").id], m.jnt_qposadr[m.joint("can_joint").id]
    y = np.radians(yaw_deg)
    d.qpos[cq:cq + 7] = [xy[0], xy[1], C.CUBE_HALF + 0.001, np.cos(y / 2), 0, 0, np.sin(y / 2)]
    d.qpos[bq:bq + 7] = [BOX_XY[0], BOX_XY[1], E.CAN_HEIGHT / 2, 1, 0, 0, 0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(20):
        mujoco.mj_step(m, d)
    q_home = {s: d.qpos[sim.robot[s].qpos_ids].copy() for s in C.SIDES}     # the start joint angles

    def joint_home(n=60):
        """After pushing: back to the start JOINT angles (the push leaves the wrist in another configuration and the
        clamp must start from the pose it was designed / verified from; the real runner ramps to SIM_HOME too)."""
        start = {s: d.ctrl[sim.robot[s].actuator_ids].copy() for s in C.SIDES}
        for k in range(1, n + 1):
            for s in C.SIDES:
                d.ctrl[sim.robot[s].actuator_ids] = start[s] + (q_home[s] - start[s]) * k / n
            for _ in range(sim._substeps):
                mujoco.mj_step(m, d)
        for _ in range(25 * sim._substeps):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        for i, s in enumerate(C.SIDES):
            sim._ee_targets[i] = sim.robot[s].position.copy()

    w = np.full(5, 400.0)
    g = SF.Guard(m, {s: (-w, w) for s in SF.SIDES}, box_xy=BOX_XY, log=lambda *a: None)
    rec = {"start": [round(xy[0] * 100, 1), round((xy[1] - C.CUBE_Y) * 100, 1)], "pushes": 0, "contacts": {}, "guard": [],
           "result": "", "phase_fail": ""}

    def run(plan, tag):
        k = 0
        while not plan.done:
            tg = plan.next_targets(sim._ee_targets.copy())
            a = np.zeros((2, 4))
            a[:, :3] = (tg - sim._ee_targets) / sim.delta_scale
            env.step(np.clip(a, -1, 1).ravel())
            for kk in DS.bad_contacts(m, d):
                rec["contacts"][f"{tag}:{plan.phase}:{kk}"] = rec["contacts"].get(f"{tag}:{plan.phase}:{kk}", 0) + 1
            if k % 3 == 0:
                for wch, q in (("cmd", {s: np.r_[np.rad2deg(d.ctrl[sim.robot[s].actuator_ids][:5]), 0.0] for s in C.SIDES}),
                               ("meas", {s: np.r_[np.rad2deg(d.qpos[sim.robot[s].qpos_ids][:5]), 0.0] for s in C.SIDES})):
                    for kind, (dist, pair) in g.distances(q).items():
                        if dist < SF.MARGIN[wch][kind] and len(rec["guard"]) < 5:
                            rec["guard"].append(f"{tag}:{plan.phase}:{wch}:{kind} {dist * 1000:.0f}mm")
            k += 1
        for _ in range(15):
            env.step(np.zeros(8))

    def yaw_now():
        Rc = d.xmat[m.body("cube").id].reshape(3, 3)
        return float(np.degrees(np.arctan2(Rc[1, 0], Rc[0, 0])))

    for rnd in range(2 if CLOSED_LOOP else 0):        # push, then square up if twisted, then push again (depth)
        c = env.cube_pos()
        if rnd == 1:
            if PT.in_zone(c[:2], yaw_deg=yaw_now()) or abs(c[0] - PT.TARGET[0]) > 0.03:
                break
            if False and PT.yaw_off(yaw_now()) > PT.MAX_YAW_DEG:     # two-arm squaring: camera mounts collide (removed)
                sq = PT.SquarePlan(sim, c[:2], x_centre=PT.TARGET[0])
                if sq.ok:
                    rec["squared"] = True
                    run(sq, "square")
                else:
                    rec["square_why"] = sq.why
        c = env.cube_pos()
        if not PT.in_zone(c[:2], yaw_deg=yaw_now()):
            push = PT.ClosedLoopPush(sim, c[:2])
            rec["arm"], k = push.side, 0
            if push.done and push.why:
                rec["result"] = "push: " + push.why
            while not push.done:
                Rc = d.xmat[m.body("cube").id].reshape(3, 3)
                tg = push.step(sim._ee_targets.copy(), env.cube_pos()[:2], float(np.degrees(np.arctan2(Rc[1, 0], Rc[0, 0]))))
                a = np.zeros((2, 4))
                a[:, :3] = (tg - sim._ee_targets) / sim.delta_scale
                env.step(np.clip(a, -1, 1).ravel())
                for kk in DS.bad_contacts(m, d):
                    rec["contacts"][f"push:{push.mode}:{kk}"] = rec["contacts"].get(f"push:{push.mode}:{kk}", 0) + 1
                if k % 3 == 0:
                    for wch, q in (("cmd", {s: np.r_[np.rad2deg(d.ctrl[sim.robot[s].actuator_ids][:5]), 0.0] for s in C.SIDES}),
                                   ("meas", {s: np.r_[np.rad2deg(d.qpos[sim.robot[s].qpos_ids][:5]), 0.0] for s in C.SIDES})):
                        for kind, (dist, pair) in g.distances(q).items():
                            if dist < SF.MARGIN[wch][kind] and len(rec["guard"]) < 5:
                                rec["guard"].append(f"push:{push.mode}:{wch}:{kind} {dist * 1000:.0f}mm")
                k += 1
                if k > 2500:
                    rec["result"] = "push timed out"
                    break
            rec["pushes"] = k
            for _ in range(15):
                env.step(np.zeros(8))
    for _ in range(0 if CLOSED_LOOP else max_pushes):
        c = env.cube_pos()
        if PT.in_zone(c[:2]):
            break
        if c[2] < 0 or c[2] > 0.03:
            rec["result"] = "cube off the table / not flat"
            break
        plan = PT.PushPlan(sim, c[:2], PT.next_push_target(c[:2]))
        if not plan.ok:
            rec["result"] = "push: " + plan.why
            break
        rec["pushes"] += 1
        rec["arm"] = plan.side
        run(plan, f"push{rec['pushes']}")
    c = env.cube_pos()
    rec["after_push"] = [round(c[0] * 100, 1), round((c[1] - C.CUBE_Y) * 100, 1)]
    R = d.xmat[m.body("cube").id].reshape(3, 3)
    rec["cube_yaw_deg"] = round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 1)
    rec["cube_tilt_deg"] = round(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), 1)
    if not rec["result"]:
        Rc = d.xmat[m.body("cube").id].reshape(3, 3)
        if not PT.in_zone(c[:2], yaw_deg=float(np.degrees(np.arctan2(Rc[1, 0], Rc[0, 0])))):
            rec["result"] = "not in zone after pushes"
        else:
            # as on the robot: the clamp is its own run from the fixed start pose (fresh scene, same cube pose, box fixed)
            import design_clamp3 as D3
            P = json.loads((HERE / "runs" / C._D.name).read_text())["params"]
            D3.DETAILS.clear()
            _, logs = D3.evaluate({**P, "box_dy_mm": P.get("box_dy_mm", -100.0)},
                                  scenes=((float(c[0]), float(c[1] - C.CUBE_Y), int(round(yaw_now())), 0.0, 0.4),),
                                  box_xy=BOX_XY)
            r3 = D3.DETAILS[0]
            rec["clamp_log"] = logs[0].split(" | ")[0][:160]
            if r3["contacts"]:
                rec["contacts"]["clamp"] = r3["contacts"]
            for wch in r3["closest_mm"]:
                for kind, v in r3["closest_mm"][wch].items():
                    if v < SF.MARGIN[wch][kind] * 1000:
                        rec["guard"].append(f"clamp:{wch}:{kind} {v:.0f}mm")
            rec["result"] = "IN BOX" if r3["in_box"] else "clamp failed"
    rec["clean"] = rec["result"] == "IN BOX" and not rec["contacts"] and not rec["guard"]
    env.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=2.5)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import clamp_teacher as C
    xs = np.arange(-12.5, 12.51, a.step) / 100
    ys = C.CUBE_Y + np.arange(-5, 12.51, a.step) / 100
    pts = [(float(x), float(y)) for y in ys for x in xs
           if not (abs(x - BOX_XY[0]) < 0.07 and abs(y - BOX_XY[1]) < 0.07)]
    # one case per worker, then a fresh process (MuJoCo scenes are not freed; the user's Mac has limited RAM)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(run_case, pts, chunksize=1)
    (HERE / "runs" / "push_clamp_map.json").write_text(json.dumps(res, indent=1))
    print("cube x (cm) ->            " + " ".join(f"{x * 100:+6.1f}" for x in xs))
    print("legend: OK = into the box, no contact, guard quiet | g = in box but guard would stop | x = contact | "
          "- = failed | box = box footprint | number = pushes used")
    for y in ys[::-1]:
        row = []
        for x in xs:
            r = next((r for r in res if abs(r["start"][0] - x * 100) < 1e-6 and abs(r["start"][1] - (y - C.CUBE_Y) * 100) < 1e-6), None)
            if r is None:
                row.append("   box")
            elif r["clean"]:
                row.append(f"  OK{r['pushes']}")
            elif r["result"] == "IN BOX" and not r["contacts"]:
                row.append(f"   g{r['pushes']}")
            elif r["contacts"]:
                row.append(f"   x{r['pushes']}")
            else:
                row.append("     -")
        print(f"{(y - C.CUBE_Y) * 100:+5.1f} ({(y + 0.22) * 100:4.1f} cm out)     " + " ".join(row))
    n = len(res)
    print(f"clean end to end: {sum(r['clean'] for r in res)}/{n}")


if __name__ == "__main__":
    main()
