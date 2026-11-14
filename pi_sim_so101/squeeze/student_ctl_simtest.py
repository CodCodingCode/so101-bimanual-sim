#!/usr/bin/env python3
"""Sim test of the RIG controller for sim-74's chunked clamp student (rig_tools/student_ctl.py), before the real arms.

World: sim-74's stage-4 env (clamp_wide: cube +-1.5 cm, any yaw, +-1 deg servo spread; CLAMP_GRIP_DR + CLAMP_SLIP).
Rig side, exactly as real_chunk_student.py: a separate kinematic SHADOW (SqueezeEnv with the calibrated cameras) gets
the world's joints each tick; the three cameras are RENDERED at 640x480 and go through the same detectors (red-blob
triangulation, vision.detect_cube_pose, vision.detect_box); StudentCtl turns the student's targets into joint commands
(IK, rate limit, squeeze limiter), which the world executes. The rig's safety guard is run on every command; where it
would have stopped the real arms is recorded (the episode continues so the outcome is still measured).

    python squeeze/student_ctl_simtest.py --episodes 16 --workers 3 [--no-limiter]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RIG_TOOLS = HERE.parents[1] / "rig_tools"
sys.path[:0] = [str(HERE), str(HERE.parent), str(RIG_TOOLS)]
os.environ["SQUEEZE_TASK"] = "clamp"
os.environ.setdefault("CLAMP_DESIGN", "design_clamp3c_twistoff.json")
os.environ.setdefault("CLAMP_GRIP_DR", "1")
os.environ.setdefault("CLAMP_SLIP", "1")
STUDENT = Path(os.environ.get("STUDENT_PT", str(HERE / "runs_chunk" / "rotation" / "student.pt")))
STAGE = 4
REAL_RANGES = HERE.parent / "pi_sim" / "assets" / "so101_real_ranges.json"


def _episodes(job):
    seeds, limiter, release = job
    import mujoco
    import clamp_wide as W            # first: sets the design before clamp_teacher is imported
    import clamp_rl as CR
    import clamp_teacher as C
    import env as E
    import vision as V
    import cube_locator as CL
    import safety as SF
    import student_ctl as SC
    from chunk_student_real import StudentRunner
    assert C._D.name == "design_clamp3c_twistoff.json", C._D.name
    if os.environ.get("SO101_RIG"):                  # the world's cameras AND the rig-side shadow use this rig
        W.RIG = Path(os.environ["SO101_RIG"])
    lvl, yaw_max, bias = W.SCHEDULE[STAGE]
    world = W.make_env_cls()(seed=int(seeds[0]), level=lvl, yaw_max=yaw_max)
    world.set_bias(bias)
    wsim, wd = world.sim, world.data
    # world executes the rig's joint commands: its IK is overridden by them
    override = {"q": None}
    warms = [wsim.robot["left"], wsim.robot["right"]]
    wsolve = wsim._solve_ik
    wsim._solve_ik = lambda arm, pos, quat, downward=None: ((override["q"][0 if arm is warms[0] else 1], 0.0)
                                                           if override["q"] is not None else wsolve(arm, pos, quat, downward))
    # the rig's kinematic shadow (calibrated cameras, cube + box parked away)
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    base_sim = E._Sim

    class _RigSim(base_sim):
        def __init__(self, *a_, **k_):
            k_.setdefault("camera_rig", str(W.RIG))
            super().__init__(*a_, **k_)
    E._Sim = _RigSim
    shadow = E.SqueezeEnv()
    E._Sim = base_sim
    shadow.reset(seed=0)
    sim, m, d = shadow.sim, shadow.model, shadow.data
    d.joint("cube_joint").qpos[:3] = [2, 2, -1]
    d.joint("can_joint").qpos[:3] = [2.5, 2, -1]
    glo, ghi = m.actuator_ctrlrange[sim.robot["left"].gripper_actuator]
    rr = json.loads(REAL_RANGES.read_text())["ranges_deg"]
    limits = {s: (np.array([r[0] for r in rr[s]], float), np.array([r[1] for r in rr[s]], float)) for s in SC.SIDES}
    runner = StudentRunner(STUDENT, delta_scale=sim.delta_scale)
    res = []
    for seed in seeds:
        world.reset(seed=int(seed))
        world._render_overhead()                                # first GL context (serialised in clamp_wide)

        def frames():
            mujoco.mj_forward(world.model, wd)
            return {c: wsim.render_camera(c, native=True) for c in SC.CAMS}

        def to_shadow():
            for s in SC.SIDES:
                wa, sa = wsim.robot[s], sim.robot[s]
                d.qpos[sa.qpos_ids] = wd.qpos[wa.qpos_ids]
                gq = wd.qpos[world.model.joint(f"{s}_gripper").qposadr[0]]
                d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = gq
            mujoco.mj_forward(m, d)
        to_shadow()
        for i, s in enumerate(SC.SIDES):
            sim._ee_targets[i] = sim.robot[s].position.copy()
            d.ctrl[sim.robot[s].actuator_ids] = d.qpos[sim.robot[s].qpos_ids]
        guard = SF.Guard(m, limits, box_xy=None, log=lambda *a: None)
        ctl = SC.StudentCtl(m, d, sim, C, V, CL, runner, guard, squeeze_limit=limiter,
                            release_gap=SC.RELEASE_GAP if release else 0.0)
        try:
            box, cube0, yaw0 = ctl.start(frames())
        except RuntimeError as e:
            res.append({"seed": int(seed), "error": str(e)})
            continue
        guard = SF.Guard(m, limits, box_xy=(box[0], box[1]), box_yaw=box[2], log=lambda *a: None)
        ctl.guard = guard
        ctrl_before = {s: d.ctrl[sim.robot[s].actuator_ids].copy() for s in SC.SIDES}
        stop, limited, after, t0 = None, 0, 0, time.time()
        trace = []
        for k in range(world.base.max_steps):
            to_shadow()
            cmd, log = ctl.tick(frames(), ctrl_before)
            ctrl_before = {s: np.deg2rad(cmd[s][:5]) for s in SC.SIDES}
            limited += log["squeeze_limited"]
            units = {s: np.r_[np.rad2deg(wd.qpos[wsim.robot[s].qpos_ids]), 0.0] for s in SC.SIDES}
            if stop is None:
                try:
                    guard.step(units, {s: cmd[s].copy() for s in SC.SIDES}, expect={s: cmd[s].copy() for s in SC.SIDES})
                except SF.SafetyStop as e:
                    stop = (k, str(e)[:160])
            override["q"] = np.array([ctrl_before["left"], ctrl_before["right"]])
            _, _, term, trunc, info = CR.ClampRLEnv.step(world, np.zeros(6, np.float32))
            if os.environ.get("TRACE2") and 260 <= k <= 320 and k % 3 == 0:
                md = guard.distances(units)["arm-arm"]
                gq = {s: float(wd.qpos[world.model.joint(f"{s}_gripper").qposadr[0]]) for s in SC.SIDES}
                tl, tr = sim.robot["left"].position, sim.robot["right"].position
                print(f"  k {k} over_box {log['over_box']} release {log['release_gap_applied']} limited {log['squeeze_limited']} | "
                      f"cmd gap {log['cmd_gap_mm']:.1f} mm, measured gap {md[0] * 1000:.1f} mm ({md[1][:40] if len(md) > 1 else ''}) | "
                      f"gripper qpos L {gq['left']:+.2f} R {gq['right']:+.2f} | tips L {np.round(tl * 100, 1).tolist()} R {np.round(tr * 100, 1).tolist()} "
                      f"| cube z {world.base.cube_pos()[2] * 100:.1f}", flush=True)
                print(f"        ee_cmd L {np.round(np.array(log['ee_cmd'][0]) * 100, 1).tolist()} R {np.round(np.array(log['ee_cmd'][1]) * 100, 1).tolist()} | "
                      f"right joints cmd {np.round(cmd['right'][:5], 1).tolist()} world ctrl {np.round(np.rad2deg(wd.ctrl[wsim.robot['right'].actuator_ids]), 1).tolist()} "
                      f"measured {np.round(np.rad2deg(wd.qpos[wsim.robot['right'].qpos_ids]), 1).tolist()}", flush=True)
            if os.environ.get("TRACE"):
                wc = world.base.cube_pos()
                tl, tr = sim.robot["left"].position, sim.robot["right"].position
                trace.append((k, ctl.attempt_start, log.get("attempt_reason", ""), world._attempt_start, round(float(wc[2]) * 100, 1), round(float(ctl.xyz[2]) * 100, 1),
                              log["cube_seen"], round(float(np.linalg.norm(tl[:2] - tr[:2])) * 100, 1), bool(world.base.squeezed()),
                              round(float(np.linalg.norm(ctl.xyz[:2] - wc[:2])) * 100, 1), log["clock"]))
            if info["is_success"]:
                after += 1
            if term or trunc or after > 12:
                break
        override["q"] = None
        if trace:
            prev_a = None
            for row in trace:
                if row[1] != prev_a or row[0] % 25 == 0:
                    print("  k %3d  ctl_attempt %3d (%s) world_attempt %3d | cube z true %.1f read %.1f seen %d | tip gap %.1f squeezed %s | xy err %.1f cm clock %.2f" % row, flush=True)
                prev_a = row[1]
        res.append({"seed": int(seed), "delivered": bool(info["is_success"]),
                    "clean": bool(info["is_success"]) and info["bad_free_steps"] == 0,
                    "contacts": {x: int(info[f"bad_{x}"]) for x in ("arm_arm", "arm_camera", "arm_box", "arm_table") if info[f"bad_{x}"]},
                    "guard_stop": stop, "limiter_ticks": int(limited), "attempts": ctl.attempts,
                    "box_err_mm": round(float(1000 * np.linalg.norm(np.asarray(box[:2]) - world.layout["box_true"][:2])), 1),
                    "cube_start_err_mm": round(float(1000 * np.linalg.norm(cube0[:2] - np.asarray(world.layout["cube"]))), 1),
                    "ticks": k + 1, "secs": round(time.time() - t0)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=880_000)
    ap.add_argument("--no-limiter", action="store_true")
    ap.add_argument("--no-release-gap", action="store_true")
    a = ap.parse_args()
    seeds = np.arange(a.seed0, a.seed0 + a.episodes)
    jobs = [([int(x) for x in c], not a.no_limiter, not a.no_release_gap) for c in np.array_split(seeds, a.workers)]
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        res = sum(pool.map(_episodes, jobs, chunksize=1), [])
    for r in res:
        print(json.dumps(r), flush=True)
    ok = [r for r in res if "error" not in r]
    n = len(ok)
    print(f"\nrig controller + sim-74 student, squeeze limiter {'OFF' if a.no_limiter else 'ON'}, release gap "
          f"{'OFF' if a.no_release_gap else 'ON'}: delivered "
          f"{sum(r['delivered'] for r in ok)}/{n}, clean {sum(r['clean'] for r in ok)}/{n}, the real guard would have "
          f"stopped {sum(r['guard_stop'] is not None for r in ok)}/{n}, start refused {len(res) - n}", flush=True)


if __name__ == "__main__":
    main()
