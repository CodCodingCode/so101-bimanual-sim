#!/usr/bin/env python3
"""Sim test of the RIG controller for the camera-detector chunked clamp policy v2 (rig_tools/real_chunk_v2.py), before
the real arms. Same structure as student_ctl_simtest.py (sim-74's student), but the world is the one this policy was
trained in: ClampRLEnv, design_clamp3, cube +-1.5 cm / yaw +-15 deg, box at the fixed offset behind it, grip
randomisation + recovery disturbances (CLAMP_GRIP_DR=1, CLAMP_RECOVERY=1), cameras = the active (bb1b28) rig.

Rig side exactly as the live runner: a separate kinematic SHADOW gets the world's joints each tick; the three cameras
are RENDERED at 640x480 and go through the rig's detectors (red blobs, overhead ray, detect_cube_pose, detect_box);
chunk_v2_ctl.ChunkV2Ctl turns the policy's chunks into joint commands (IK, rate limit, squeeze limiter, release gap);
the command filter (cmd_filter.CmdFilter) runs on them; the world executes them. The rig's safety guard checks every
command and measured pose; where it would have stopped the real arms is recorded (the episode continues).

    python squeeze/chunk_v2_ctl_simtest.py --episodes 12 --workers 3 [--no-filter]
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
os.environ["CLAMP_DESIGN"] = "design_clamp3.json"
os.environ.setdefault("CLAMP_GRIP_DR", "1")
os.environ.setdefault("CLAMP_RECOVERY", "1")
POLICY = Path(os.environ.get("CHUNK_V2_PT", str(HERE / "runs_clamp_v3" / "chunk_policy_v2_bb1b28" / "best.pt")))
REAL_RANGES = HERE.parent / "pi_sim" / "assets" / "so101_real_ranges.json"


def _episodes(job):
    seeds, use_filter, arm_arm_guard = job
    import mujoco
    import torch
    torch.set_num_threads(1)
    import distill as D               # world = ClampRLEnv with the active rig's cameras
    import clamp_rl as CR
    import clamp_teacher as C
    import env as E
    import vision as V
    import cube_locator as CL
    import safety as SF
    import chunk_v2_ctl as CV
    from cmd_filter import CmdFilter
    if not arm_arm_guard:                                     # as the live runner (real_chunk_v2.drop_measured_arm_arm)
        SF.MARGIN["meas"].pop("arm-arm", None)
    assert C._D.name == "design_clamp3.json", C._D.name
    world = D.make_env(int(seeds[0]))
    wsim, wd = world.base.sim, world.data
    override = {"q": None}
    warms = [wsim.robot["left"], wsim.robot["right"]]
    wsolve = wsim._solve_ik
    wsim._solve_ik = lambda arm, pos, quat, downward=None: ((override["q"][0 if arm is warms[0] else 1], 0.0)
                                                           if override["q"] is not None else wsolve(arm, pos, quat, downward))
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    shadow = E.SqueezeEnv()                                   # E._Sim is distill's camera-rig sim: same cameras
    shadow.reset(seed=0)
    sim, m, d = shadow.sim, shadow.model, shadow.data
    d.joint("cube_joint").qpos[:3] = [2, 2, -1]
    d.joint("can_joint").qpos[:3] = [2.5, 2, -1]
    rr = json.loads(REAL_RANGES.read_text())["ranges_deg"]
    limits = {s: (np.array([r[0] for r in rr[s]], float), np.array([r[1] for r in rr[s]], float)) for s in CV.SC.SIDES}
    runner = CV.ChunkV2Runner(POLICY, delta_scale=sim.delta_scale)
    import rt_control as RT
    PERCEPTION = RT.load_config()["perception"]          # the live runner's cube filters (height re-read, yaw filter)
    res = []
    for seed in seeds:
        world.reset(seed=int(seed))

        def frames():
            mujoco.mj_forward(world.model, wd)
            return {c: wsim.render_camera(c, native=True) for c in CV.SC.CAMS}

        def to_shadow():
            for s in CV.SC.SIDES:
                wa, sa = wsim.robot[s], sim.robot[s]
                d.qpos[sa.qpos_ids] = wd.qpos[wa.qpos_ids]
                d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = wd.qpos[world.model.joint(f"{s}_gripper").qposadr[0]]
            mujoco.mj_forward(m, d)
        to_shadow()
        for i, s in enumerate(CV.SC.SIDES):
            sim._ee_targets[i] = sim.robot[s].position.copy()
            d.ctrl[sim.robot[s].actuator_ids] = d.qpos[sim.robot[s].qpos_ids]
        guard = SF.Guard(m, limits, box_xy=None, log=lambda *a: None)
        ctl = CV.ChunkV2Ctl(m, d, sim, C, V, CL, runner, guard, squeeze_limit=True, filters=PERCEPTION)
        try:
            box, cube0, yaw0 = ctl.start(frames())
        except RuntimeError as e:
            res.append({"seed": int(seed), "error": str(e)})
            continue
        guard = SF.Guard(m, limits, box_xy=(box[0], box[1]), box_yaw=box[2], log=lambda *a: None)
        ctl.guard = guard
        filt = CmdFilter(guard) if use_filter else None
        ctrl_before = {s: d.ctrl[sim.robot[s].actuator_ids].copy() for s in CV.SC.SIDES}
        prev_cmd = {s: np.r_[np.rad2deg(ctrl_before[s]), 0.0] for s in CV.SC.SIDES}
        stop, limited, cut, after, t0, det_seen = None, 0, 0, 0, time.time(), 0
        for k in range(world.base.max_steps):
            to_shadow()
            ee_prev = sim._ee_targets.copy()
            cmd, log = ctl.tick(frames(), ctrl_before)
            if filt is not None:
                cmd, kept = CV.filter_cmd(filt, ctl, prev_cmd, cmd, ee_prev, 1.0)
                cut += kept < 1.0
            prev_cmd = cmd
            det_seen += runner.det_seen
            ctrl_before = {s: np.deg2rad(cmd[s][:5]) for s in CV.SC.SIDES}
            limited += log["squeeze_limited"]
            if os.environ.get("TRACE") and (ctl.attempts != getattr(ctl, "_tr_a", 1) or k % 50 == 0):
                wc = world.base.cube_pos()
                tl, tr = sim.robot["left"].position, sim.robot["right"].position
                print(f"  k {k:3d} ctl attempt {ctl.attempts} ({log['attempt_reason']}) world attempt_start "
                      f"{world._attempt_start} | cube true {np.round(wc * 100, 1).tolist()} ctl {np.round(ctl.xyz * 100, 1).tolist()} "
                      f"det {np.round(runner.det_xy * 100, 1).tolist()} | tip gap {np.linalg.norm(tl[:2] - tr[:2]) * 100:.1f} cm "
                      f"held(ctl) {ctl._holding_now()} ever {ctl.ever_held} squeezed(world) {bool(world.base.squeezed())} "
                      f"clock {log['clock']:.2f}", flush=True)
                ctl._tr_a = ctl.attempts
            units = {s: np.r_[np.rad2deg(wd.qpos[wsim.robot[s].qpos_ids]), 0.0] for s in CV.SC.SIDES}
            if stop is None:
                try:
                    guard.step(units, {s: cmd[s].copy() for s in CV.SC.SIDES}, expect={s: cmd[s].copy() for s in CV.SC.SIDES})
                except SF.SafetyStop as e:
                    stop = (k, str(e)[:160])
            override["q"] = np.array([ctrl_before["left"], ctrl_before["right"]])
            _, _, term, trunc, info = CR.ClampRLEnv.step(world, np.zeros(6, np.float32))
            override["q"] = None
            if info["is_success"]:
                after += 1
            if term or trunc or after > 12:
                break
        res.append({"seed": int(seed), "delivered": bool(info["is_success"]),
                    "clean": bool(info["is_success"]) and info["bad_free_steps"] == 0,
                    "contacts": {x: int(info[f"bad_{x}"]) for x in ("arm_arm", "arm_camera", "arm_box", "arm_table")
                                 if info.get(f"bad_{x}")},
                    "guard_stop": stop, "limiter_ticks": int(limited), "filter_cut_ticks": int(cut),
                    "attempts": ctl.attempts, "overhead_seen": round(det_seen / (k + 1), 2),
                    "box_err_mm": round(float(1000 * np.linalg.norm(np.asarray(box[:2]) - np.asarray(world._box_xy))), 1),
                    "cube_start_err_mm": round(float(1000 * np.linalg.norm(cube0[:2] - world._attempt_cube)), 1),
                    "perturbation": info.get("perturbation"), "ticks": k + 1, "secs": round(time.time() - t0)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=960_000)       # chunk_policy_v2's own eval seeds
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--arm-arm-guard", action="store_true", help="keep the measured arm-arm stop (removed by default, "
                    "as in the live runner)")
    a = ap.parse_args()
    seeds = np.arange(a.seed0, a.seed0 + a.episodes)
    jobs = [([int(x) for x in c], not a.no_filter, a.arm_arm_guard) for c in np.array_split(seeds, a.workers)]
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        res = sum(pool.map(_episodes, jobs, chunksize=1), [])
    for r in res:
        print(json.dumps(r), flush=True)
    ok = [r for r in res if "error" not in r]
    n = len(ok)
    print(f"\nrig controller + chunk policy v2 ({POLICY.parent.name}), cmd filter {'OFF' if a.no_filter else 'ON'}, measured arm-arm stop "
          f"{'ON' if a.arm_arm_guard else 'OFF'}: "
          f"delivered {sum(r['delivered'] for r in ok)}/{n}, clean {sum(r['clean'] for r in ok)}/{n}, the real guard would "
          f"have stopped {sum(r['guard_stop'] is not None for r in ok)}/{n}, start refused {len(res) - n}", flush=True)


if __name__ == "__main__":
    main()
