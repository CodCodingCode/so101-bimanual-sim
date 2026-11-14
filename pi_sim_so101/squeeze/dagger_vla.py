#!/usr/bin/env python3
"""VLA DAgger for push + rotation (2026-11-13): SmolVLA drives, the scripted teacher takes over from ITS state.

Why: the clamp_mix SmolVLA (5k / 10k checkpoints, sim push + rotation 55-60 %) pushes the cube most of the way, then
stalls a few cm short of the clamp's spot with both arms parked -- it never makes the small second push. Cube starting
on the left: 3/9 at both checkpoints. Those states never occur in the teacher's own demos (the scripted pusher always
reaches its zone before it parks), so the VLA never saw how to get out of them.

Each episode: the VLA drives from the start (same layout sampling as record_vla.py --pushclamp, biased to the left)
until a random switch tick, or earlier as soon as it STALLS (no joint moving > STALL_DEG over STALL_TICKS while not
squeezing). Then the teacher (push_clamp_wide.PushClamp: closed-loop box-aware push + the rotation clamp policy, the
same teacher and env flags that recorded the demos) takes over from that exact state and finishes. Only the teacher's
part is recorded (frames, state, action as record_vla.py), and only takeovers that deliver the cube are kept: recovery
demonstrations on the VLA's own states (DAgger with expert take-over).

    cd ~/so101_vla/sim/pi_sim_so101 && MUJOCO_GL=egl CLAMP_GUARD=1 CLAMP_GUARD_CMD=1 CLAMP_GRIP_DR=1 CLAMP_SLIP=1 \
      WIDE_RIG_RULES=1 WIDE_RIG_LIMITER=1 python squeeze/dagger_vla.py --ckpt <run>/checkpoints/020000/pretrained_model \
        --teacher runs_clamp_wide_guard3/rotation_servo_done.zip --episodes 160 --seed0 500000 \
        --out ~/so101_vla/dagger_r1 --workers 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # warprm is shared (see eval_smolvla.py)
    os.environ.setdefault(_v, "2")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

EPISODE_S = 120.0                 # VLA part + a full teacher push / rehome / clamp after it
SWITCH_MIN, SWITCH_MAX = 60, 900  # random take-over tick (25 Hz)
STALL_TICKS, STALL_DEG = 50, 2.0  # stalled: no joint target moved more than this over that many ticks


def _worker(job):
    ckpt, teacher_path, seeds, tmp, band, nas, left_frac, grace, cmd_filter = job
    import mujoco
    import torch
    from PIL import Image
    import chunk_student as CS
    import clamp_teacher as C
    import clamp_wide as W
    import push_clamp_wide as P
    import record_vla as RV
    torch.set_num_threads(2)

    env = W.make_env_cls()(seed=int(seeds[0]), level=0.0, yaw_max=45.0, episode_seconds=EPISODE_S)
    env.set_bias(1.0)
    m, d, sim = env.model, env.data, env.sim
    for c in RV.CAMS:                      # GL context BEFORE CUDA (EGL failed when created after CUDA init)
        sim.render_camera(c, native=True)
    teacher = CS.Teacher(teacher_path)
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    ptype = json.loads((Path(ckpt).expanduser() / "config.json").read_text())["type"]   # SmolVLA, ACT, ...
    policy = get_policy_class(ptype).from_pretrained(ckpt).to("cuda").eval()
    policy.config.n_action_steps = nas
    use_task = ptype in ("smolvla", "pi0", "pi05")
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt)
    rnd = RV._randomizer(env)              # VLA_DR=1: the recorded teacher frames look like the --dr training data

    kept, log = [], []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        env.reset(seed=int(seed))
        x = rng.uniform(-0.12, 0.0) if rng.random() < left_frac else rng.uniform(0.0, 0.12)
        start = (x, C.CUBE_Y + rng.uniform(-band, band) / 100)
        P._set_start(env, start)
        if rnd is not None:                # AFTER reset (reset restores the nominal look)
            rnd.apply(np.random.default_rng([int(seed), 99]))
        q_home = {s: d.ctrl[arm.actuator_ids].copy() for s, arm in sim.robot.items()}
        ee_home = sim._ee_targets.copy()
        k_switch = int(rng.integers(SWITCH_MIN, SWITCH_MAX))
        policy.reset()
        hist, why, vla_ok, t0 = [], "random", False, time.time()
        on_steps = 0
        filt = None
        if cmd_filter:                                 # the VLA as deployed: commands through vla_safety.CmdFilter
            from vla_safety import CmdFilter
            filt = CmdFilter(env)
        # ---------------- VLA part (exactly as eval_smolvla.py drives it)
        for t in range(k_switch + 400):
            imgs = RV.frames(env)
            obs = {f"observation.images.{c}": torch.from_numpy(imgs[c]).permute(2, 0, 1).float() / 255.0 for c in RV.CAMS}
            obs["observation.state"] = torch.from_numpy(RV.state_vec(env)).float()
            if use_task:
                obs["task"] = RV.TASK
            with torch.no_grad():
                act = np.asarray(post(policy.select_action(pre(obs))).squeeze(0).cpu().numpy(), float)
            q_prev, q_new = {}, {}
            for i, s in enumerate(("left", "right")):
                arm = sim.robot[s]
                tgt = np.deg2rad(act[i * 6:i * 6 + 5])
                cur = d.ctrl[arm.actuator_ids]
                lo, hi = m.actuator_ctrlrange[arm.gripper_actuator]
                q_prev[s] = np.r_[np.rad2deg(cur), 100.0 * (d.ctrl[arm.gripper_actuator] - lo) / (hi - lo)]
                q_new[s] = np.r_[np.rad2deg(cur + np.clip(tgt - cur, -sim._rate_limit, sim._rate_limit)),
                                 np.clip(act[i * 6 + 5], 0.0, 100.0)]
            if filt is not None:
                q_new, _ = filt(q_prev, q_new)
            for s in ("left", "right"):
                arm = sim.robot[s]
                lo, hi = m.actuator_ctrlrange[arm.gripper_actuator]
                d.ctrl[arm.actuator_ids] = np.deg2rad(q_new[s][:5])
                d.ctrl[arm.gripper_actuator] = lo + q_new[s][5] / 100.0 * (hi - lo)
            for _ in range(sim._substeps):
                mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            env.base._step += 1
            on_steps = on_steps + 1 if env.base.on_can() else 0
            if on_steps >= env.base.settle_steps:
                vla_ok = True
                break
            if env.base.can_tipped():
                break
            sq = env.base.squeezed()
            if getattr(env, "_guard", None) is not None and env._guard_check()[1]:
                why = "guard"                          # the real rig would stop HERE: the teacher shows the way out
                break
            hist.append(np.concatenate([d.ctrl[a.actuator_ids] for a in sim.robot.values()]))
            stalled = (t > 150 and len(hist) > STALL_TICKS and not sq and
                       np.degrees(np.abs(hist[-1] - hist[-1 - STALL_TICKS]).max()) < STALL_DEG)
            if stalled:
                why = "stall"
            if (t + 1 >= k_switch or stalled) and not sq:        # never take over in the middle of a squeeze
                break
        t_vla = env.base._step
        if vla_ok or env.base.can_tipped():
            log.append({"seed": int(seed), "x": x, "vla_ok": vla_ok, "tipped": bool(env.base.can_tipped()), "t_vla": t_vla})
            print(f"dagger seed {seed} x {x * 100:+.1f} cm: VLA {'delivered' if vla_ok else 'tipped the box'} "
                  f"by tick {t_vla} (no take-over)", flush=True)
            continue
        # ---------------- teacher take-over from this state
        sim._ee_targets[:] = np.array([a.position for a in sim.robot.values()])   # IK continues from where the arms ARE
        pc = P.PushClamp(env, teacher)
        pc.start(env._observe())
        pc.q_start, pc.ee_start = q_home, ee_home          # the clamp's trained start pose, not the VLA's parked pose
        # round 1: 75 of 81 failed take-overs were guard stops, many on the FIRST tick -- the VLA had already left the
        # arms commanded inside the guard's margins. The teacher's way OUT of such a pose is exactly what the VLA has
        # to learn, so the guard is paused for the first `grace` take-over ticks (clamp_rl's own pause for training
        # shoves), then enforced for the rest of the recovery.
        env._grace = grace
        S, A, F, info, Q = [], [], {c: [] for c in RV.CAMS}, {}, []
        while env.base._step < env.base.max_steps:
            fr = RV.frames(env)
            s = RV.state_vec(env)
            Q.append(env.data.qpos.copy())
            _, _, term, trunc, info = pc.step()
            S.append(s)
            A.append(RV.action_vec(env))
            for c in RV.CAMS:
                F[c].append(fr[c])
            if term or trunc or info.get("is_success"):
                break
        ok = bool(info.get("is_success"))
        end = ("delivered" if ok else f"guard stop ({info.get('guard_stop')})" if info.get("guard_stop") else
               "box tipped" if info.get("can_tipped") else "out of time")
        rec = {"seed": int(seed), "x": x, "vla_ok": False, "why": why, "t_switch": t_vla, "teacher_ok": ok, "end": end,
               "teacher_frames": len(S), "clean": int(info.get("bad_contact_steps", 0) == 0)}
        log.append(rec)
        if ok:
            out = Path(tmp) / f"dg_seed{int(seed)}.npz"
            np.savez_compressed(out, state=np.array(S), action=np.array(A), qpos=np.array(Q, np.float32), stage=-2, clean=rec["clean"],
                                **{f"img_{c}": np.array(F[c], np.uint8) for c in RV.CAMS})
            kept.append(str(out))
        print(f"dagger seed {seed} x {x * 100:+.1f} cm: take-over at tick {t_vla} ({why}), teacher "
              f"{'delivered -> kept' if ok else 'failed: ' + end + ', dropped'} ({len(S)} frames, "
              f"{time.time() - t0:.0f}s)", flush=True)
    env.close()
    return kept, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--episodes", type=int, default=160)
    ap.add_argument("--seed0", type=int, default=500000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--band", type=float, default=3.0, help="cube start y band (cm), as record_vla.py --pushclamp")
    ap.add_argument("--n-action-steps", type=int, default=25)
    ap.add_argument("--left-frac", type=float, default=0.6, help="share of starts on the left (x < 0)")
    ap.add_argument("--grace", type=int, default=40, help="take-over ticks with the safety guard paused (0 = round 1)")
    ap.add_argument("--cmd-filter", action="store_true",
                    help="VLA commands through vla_safety.CmdFilter (as deployed); take-overs then come from real "
                         "(measured) guard trips, stalls and random ticks")
    ap.add_argument("--dr", action="store_true", help="per-episode appearance randomisation (as record_vla.py --dr)")
    a = ap.parse_args()
    if a.dr:
        os.environ["VLA_DR"] = "1"                     # inherited by the spawned workers (record_vla._randomizer)
    out = Path(a.out).expanduser()
    tmp = out / "_episodes"
    tmp.mkdir(parents=True, exist_ok=True)
    seeds = np.arange(a.seed0, a.seed0 + a.episodes)
    teacher = str((HERE / a.teacher) if not Path(a.teacher).is_absolute() else a.teacher)
    jobs = [(a.ckpt, teacher, [int(s) for s in c], str(tmp), a.band, a.n_action_steps, a.left_frac, a.grace,
             a.cmd_filter) for c in np.array_split(seeds, a.workers) if len(c)]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(_worker, jobs)
    files = [f for k, _ in parts for f in k]
    log = [r for _, lg in parts for r in lg]
    (out / "dagger_log.json").write_text(json.dumps(log, indent=1))
    n_vla = sum(r["vla_ok"] for r in log)
    tk = [r for r in log if "teacher_ok" in r]
    print(f"SUMMARY episodes {len(log)}: VLA delivered alone {n_vla}; take-overs {len(tk)} "
          f"({sum(r['why'] == 'stall' for r in tk)} after a stall, {sum(r['why'] == 'guard' for r in tk)} at a guard "
          f"stop), teacher delivered {sum(r['teacher_ok'] for r in tk)} "
          f"-> kept {len(files)}", flush=True)
    import record_vla as RV
    RV.build_dataset(out, files)


if __name__ == "__main__":
    main()
