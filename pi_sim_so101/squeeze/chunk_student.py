#!/usr/bin/env python3
"""ACTION-CHUNKED student for the wide clamp (user, 2026-11-12: "as the architecture for all of our robot setups it
should be action chunked"). Distilled by DAgger from the clamp_wide teacher (scripted IK plan + residual PPO, which
sees sim-privileged state) into a policy that only uses what the real rig measures.

Inputs (all measurable on the rig):
  per arm: joint angles / 90 deg, gripper opening, tool point by forward kinematics x5, commanded tool point x5
  cube: three-camera triangulated position (cube_locator.SimCubeSensor: ray-cast visibility, partial occlusion, pixel
        noise; holds the last fix while hidden) + "seen", and its yaw from the overhead red blob (vision.detect_cube_pose
        on the real rig; modelled here as the true yaw mod 90 deg + 1.5 deg noise, held while hidden) as sin/cos(4 yaw)
  box: the classical overhead box fit at reset (vision.detect_box), x, y, sin/cos(4 yaw)
  the grab-attempt clock, the previous executed tool-point step
Output: a CHUNK of K = 8 future tool-point targets per arm, relative to the commanded tool point at prediction time,
metres x 20 (linear head). Execution (ChunkEnsembler, same convention as sim-e5's real runner): at each tick average
every chunk that covers the tick, weights exp(-0.1 * age index) oldest first, normalised; clip the step to the
training per-tick limit; IK.

Labels: round 0 = the teacher's own executed targets t..t+K-1 (behaviour cloning on teacher rollouts). DAgger rounds
= at each state the STUDENT visits, the env is snapshotted (MuJoCo state + the Python-side plan / targets / counters),
the teacher is run K ticks ahead, its commanded targets recorded, and the env restored.

    python squeeze/chunk_student.py --teacher runs_clamp_wide/rotation_done.zip --stage 3 --rounds 4 --workers 3
    python squeeze/chunk_student.py --eval runs_chunk/rotation/student.pt --stage 3 --episodes 48
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "rig_tools")]

K = 8                     # chunk length (ticks, 40 ms each)
SCALE = 20.0              # metres -> net units
ENSEMBLE_M = 0.1          # ACT temporal-ensembling weight exp(-m * age index), oldest first
YAW_NOISE_DEG = 1.5
OBS_DIM = 41
SIDES = ("left", "right")


# ------------------------------------------------------------------ env snapshot (for K-tick teacher lookahead)
_SKIP = {"model", "data", "_ik_data", "robot", "_renderers", "_camera_plans", "camera_specs", "cam", "action_space",
         "observation_space", "_baseline", "sim", "base", "np_random", "_np_random", "spec", "metadata"}


def _py_state(obj):
    out = {}
    for k, v in vars(obj).items():
        if k in _SKIP or callable(v) or type(v).__module__.startswith("mujoco"):
            continue
        try:
            out[k] = copy.deepcopy(v)
        except Exception:
            pass
    return out


def snapshot(env):
    import mujoco
    m, d = env.model, env.data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    st = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, st, spec)
    return (st, env.sim._ik_data.qpos.copy(), _py_state(env), _py_state(env.base), _py_state(env.sim))


def restore(env, snap):
    import mujoco
    st, ikq, pe, pb, ps = snap
    mujoco.mj_setState(env.model, env.data, st, mujoco.mjtState.mjSTATE_INTEGRATION)
    env.sim._ik_data.qpos[:] = ikq
    for obj, saved in ((env, pe), (env.base, pb), (env.sim, ps)):
        for k, v in saved.items():
            setattr(obj, k, copy.deepcopy(v))
    mujoco.mj_forward(env.model, env.data)


# ------------------------------------------------------------------ teacher
class Teacher:
    def __init__(self, path):
        from stable_baselines3 import PPO
        self.net = PPO.load(path, device="cpu")
        st = json.loads(Path(str(path).replace(".zip", "_obs_stats.json")).read_text())
        self.mu, self.var = np.array(st["mean"]), np.array(st["var"])

    def act(self, obs):
        o = np.clip((obs - self.mu) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)
        return np.asarray(self.net.predict(o, deterministic=True)[0], np.float64)


def teacher_step(env, teacher, obs):
    """One full teacher tick (plan + residual) through ClampWideEnv.step."""
    return env.step(teacher.act(obs))


def lookahead(env, teacher, obs):
    """Teacher's commanded tool-point targets for the next K ticks from the current state (env restored after)."""
    snap = snapshot(env)
    ee0 = env.sim._ee_targets.copy()
    out, o = [], obs
    for _ in range(K):
        o, _, term, trunc, _ = teacher_step(env, teacher, o)
        out.append(env.sim._ee_targets.copy())
        if term or trunc:
            break
    while len(out) < K:
        out.append(out[-1])
    restore(env, snap)
    return (np.array(out) - ee0) * SCALE                          # (K, 2, 3)


# ------------------------------------------------------------------ sensors (what the rig sees)
class Sensors:
    def __init__(self, env, rng):
        import cube_locator as CL
        self.env, self.rng = env, rng
        self.tri = CL.SimCubeSensor(env.model, env.data, env.sim.camera_specs, rng)
        self.xyz, self.seen, self.yaw = None, 0.0, 0.0

    def read(self):
        import clamp_teacher as C
        xyz, how, dets = self.tri.locate()
        if xyz is not None:
            self.xyz, self.seen = xyz, 1.0
            self.yaw = self.env.base.cube_yaw() + np.radians(self.rng.normal(0, YAW_NOISE_DEG))
        else:
            self.seen = 0.0
        if self.xyz is None:                   # the runner refuses to start blind; the start spot stands in
            c = self.env.base.cube_pos()
            self.xyz = np.array([c[0], c[1], C.CUBE_HALF])
            self.yaw = self.env.base.cube_yaw()
        return self.xyz, self.seen, self.yaw


def student_obs(env, sens, prev_step):
    import clamp_teacher as C
    b, sim = env.base, env.sim
    xyz, seen, yaw = sens.read()
    parts = []
    for i, s in enumerate(SIDES):
        arm = sim.robot[s]
        parts += [np.rad2deg(b.data.qpos[arm.qpos_ids]) / 90.0, [arm.gripper_opening], arm.position * 5.0,
                  sim._ee_targets[i] * 5.0]
    bd = env.box_det
    parts += [[xyz[0] / 0.05, (xyz[1] - C.CUBE_Y) / 0.05, (xyz[2] - C.CUBE_HALF) / 0.05, seen,
               np.sin(4 * yaw), np.cos(4 * yaw)],
              [bd[0] / 0.05, (bd[1] + 0.077) / 0.05, np.sin(4 * bd[2]), np.cos(4 * bd[2])],
              [(b._step - env._attempt_start) / 450.0], prev_step.ravel() * SCALE]
    return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])


# ------------------------------------------------------------------ student + chunk execution
def build_net():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(OBS_DIM, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                         nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, K * 6))


class ChunkEnsembler:
    """ACT temporal ensembling: chunks are stored as ABSOLUTE tool-point targets; each tick averages every chunk that
    covers it, weights exp(-m * i) with i = 0 for the oldest."""

    def __init__(self, m=ENSEMBLE_M):
        self.m, self.chunks, self.t = m, [], 0

    def add(self, chunk_abs):
        self.chunks.append((self.t, np.asarray(chunk_abs, float)))
        self.chunks = [(t0, c) for t0, c in self.chunks if self.t - t0 < K]

    def next(self):
        preds = [c[self.t - t0] for t0, c in self.chunks if 0 <= self.t - t0 < K]
        w = np.exp(-self.m * np.arange(len(preds)))
        self.t += 1
        return np.tensordot(w / w.sum(), np.array(preds), axes=1)


def execute_target(env, target):
    """Drive the arms to a commanded tool-point target (2, 3) for one tick (per-tick clip = the training limit), via
    ClampRLEnv.step (no residual), keeping the teacher's plan bookkeeping anchored to where the arms actually are."""
    import clamp_rl as CR
    ee = env.sim._ee_targets.copy()
    target = env.rig_rules(target)                                 # the rig controller's preload cap, as deployed
    step = np.clip(target - ee, -CR.ACTION_SCALE * env.sim.delta_scale, CR.ACTION_SCALE * env.sim.delta_scale)
    env._nom = env.plan.next_targets(env._nom)                     # the plan's clock keeps running
    env._place_feedback()                                          # the teacher's closed-loop drop, shadowed
    out = CR.ClampRLEnv.step(env, np.clip(step / (env.sim.delta_scale * CR.ACTION_SCALE), -1, 1).ravel())
    if id(env.plan) != env._plan_id:
        env._plan_id = id(env.plan)
    env._nom = env.sim._ee_targets - env.offset - env.place_corr   # re-anchor: teacher continues from here
    return out, env.sim._ee_targets - ee


# ------------------------------------------------------------------ rollouts
# PUSH-THEN-CLAMP mode (user, 2026-11-13: "teach the chunked model ... push then clamp"): CS_PUSH_BAND=<cm> makes every
# episode start with the cube anywhere in x +-12 cm, y +-band cm (as record_vla.py --pushclamp / the VLA test); the
# scripted box-aware pusher (push_clamp_wide.PushClamp, cube from the cameras) pushes it to the clamp's spot and
# re-homes, and only THEN does the student / teacher loop below run -- the student learns the clamp from post-push states
# (the rotation-trained student alone: 18/60 after a push vs the teacher's 44/60).
PUSH_BAND = float(os.environ.get("CS_PUSH_BAND", "0"))


def make_env(stage, seed):
    import clamp_wide as W
    lvl, yaw, bias = W.SCHEDULE[stage]
    if PUSH_BAND:
        import push_clamp_wide as P
        env = W.make_env_cls()(seed=seed, level=0.0, yaw_max=yaw, episode_seconds=P.EPISODE_S)
    else:
        env = W.make_env_cls()(seed=seed, level=lvl, yaw_max=yaw)
    env.set_bias(bias)                     # real-servo error of that stage (applied from the next reset)
    return env


def push_first(env, teacher, seed):
    """Push-then-clamp mode: place the cube, run the scripted push + re-home; returns (teacher obs, info, ended).
    The push runs under the RIG's guard margins: the training-only margin scaling (WIDE_GUARD_SCALE*, applied to
    safety.MARGIN in-process by clamp_wide) is for the learned clamp; the scripted pusher was never built for it
    (first try with the scaled margins during the push: teacher 12 % vs 73 % at the rig's margins)."""
    import clamp_teacher as C
    import clamp_wide as W
    import push_clamp_wide as P
    import safety as SF
    rng = np.random.default_rng(int(seed))
    start = (rng.uniform(-0.12, 0.12), C.CUBE_Y + rng.uniform(-PUSH_BAND, PUSH_BAND) / 100)
    scaled = {w: dict(SF.MARGIN[w]) for w in ("meas", "cmd")}
    if getattr(SF, "_wide_scaled", False):
        for k in SF.MARGIN["meas"]:
            SF.MARGIN["meas"][k] = scaled["meas"][k] / (W.GUARD_SCALE_BOX if k == "arm-box" else W.GUARD_SCALE)
        for k in SF.MARGIN["cmd"]:
            SF.MARGIN["cmd"][k] = scaled["cmd"][k] / W.GUARD_SCALE_CMD
    try:
        pc = P.PushClamp(env, teacher, detect=True)
        pc.start(P._set_start(env, start))
        info = {}
        while pc.phase != "clamp":
            _, _, term, trunc, info = pc.step()
            if term or trunc:
                return pc.obs, info, True
        return pc.obs, info, False
    finally:
        for w in ("meas", "cmd"):
            SF.MARGIN[w].update(scaled[w])


def _rollouts(job):
    stage, seeds, teacher_path, net_path, beta, keep = job
    import torch
    torch.set_num_threads(1)
    teacher = Teacher(teacher_path)
    net = None
    if net_path:
        net = build_net()
        net.load_state_dict(torch.load(net_path, map_location="cpu"))
        net.eval()
    env = make_env(stage, int(seeds[0]))
    O, Y, res = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) + 31)
        o_t, _ = env.reset(seed=int(seed))
        sens, ens = Sensors(env, rng), ChunkEnsembler()
        prev = np.zeros((2, 3))
        drive_teacher = net is None or rng.random() < beta          # per-episode mixing (DAgger beta)
        info, tgt_hist, obs_hist = {}, [], []
        if PUSH_BAND:                                                # scripted push first; the clamp is learned
            o_t, info, ended = push_first(env, teacher, seed)
            if ended:                                                # push phase ended the episode (guard / time)
                res.append((False, int(info.get("bad_contact_steps", 0)), bool(info.get("can_tipped")),
                            bool(drive_teacher)))
                continue
        for t in range(env.base.max_steps):
            o_s = student_obs(env, sens, prev)
            if drive_teacher:
                ee = env.sim._ee_targets.copy()
                o_t, _, term, trunc, info = teacher_step(env, teacher, o_t)
                prev = env.sim._ee_targets - ee
                if keep:
                    obs_hist.append(o_s)
                    tgt_hist.append((ee, env.sim._ee_targets.copy()))
            else:
                if keep:
                    O.append(o_s)
                    Y.append(lookahead(env, teacher, o_t))
                # shadow the teacher's residual offset (part of its observation) as if it were driving
                import clamp_wide as W
                a_t = teacher.act(o_t).reshape(2, 3)
                env.offset += np.clip(W.OFFSET_MAX * a_t - env.offset, -W.OFFSET_RATE, W.OFFSET_RATE)
                with torch.no_grad():
                    chunk = net(torch.from_numpy(o_s[None])).numpy()[0].reshape(K, 2, 3) / SCALE
                ens.add(env.sim._ee_targets + chunk)
                (o_t, _, term, trunc, info), prev = execute_target(env, ens.next())
            if term or trunc:
                break
        if drive_teacher and keep:                                   # BC labels from the teacher's own future
            after = [a for _, a in tgt_hist]
            for i, (o_s, (ee0, _)) in enumerate(zip(obs_hist, tgt_hist)):
                fut = after[i:i + K]
                fut += [fut[-1]] * (K - len(fut))
                O.append(o_s)
                Y.append((np.array(fut) - ee0) * SCALE)
        res.append((bool(info.get("is_success")), int(info.get("bad_contact_steps", 0)), bool(info.get("can_tipped")),
                    bool(drive_teacher)))
    env.close()
    return (np.array(O, np.float32).reshape(-1, OBS_DIM), np.array(Y, np.float32).reshape(-1, K * 6), res)


def run(pool, stage, seeds, teacher, net_path, beta, keep, workers):
    jobs = [(stage, [int(x) for x in c], teacher, net_path, beta, keep) for c in np.array_split(seeds, workers) if len(c)]
    parts = pool.map(_rollouts, jobs, chunksize=1)
    O = np.concatenate([p[0] for p in parts]) if keep else None
    Y = np.concatenate([p[1] for p in parts]) if keep else None
    return O, Y, sum((p[2] for p in parts), [])


def summary(res, who=None):
    r = [x for x in res if who is None or x[3] == who]
    n = max(len(r), 1)
    succ = sum(x[0] for x in r) / n
    clean = sum(x[0] and x[1] == 0 for x in r) / n
    return succ, clean, f"success {succ:.2f} clean {clean:.2f} box tipped {sum(x[2] for x in r) / n:.2f} ({len(r)} episodes)"


def fit(net, O, Y, steps):
    import torch
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    Ot, Yt = torch.from_numpy(O), torch.from_numpy(Y)
    for k in range(steps):
        idx = torch.randint(0, len(O), (1024,))
        loss = (net(Ot[idx] + torch.randn(1024, OBS_DIM) * 0.01) - Yt[idx]).abs().mean()     # L1, as ACT
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs_clamp_wide/rotation_done.zip")
    ap.add_argument("--stage", type=int, default=3, help="clamp_wide.SCHEDULE index for layouts (3 = +-1.5 cm, any yaw)")
    ap.add_argument("--out", default="runs_chunk/rotation")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=36)
    ap.add_argument("--eval-episodes", type=int, default=36)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--init", default=None, help="warm-start the student from this student.pt (same architecture)")
    a = ap.parse_args()
    import torch
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    teacher = str(HERE / a.teacher)
    log = open(out / "log.txt", "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        if a.eval:
            _, _, res = run(pool, a.stage, np.arange(90000, 90000 + a.eval_episodes), teacher, str(HERE / a.eval), 0.0, False, a.workers)
            say(f"eval {a.eval} stage {a.stage}: {summary(res)[2]}")
            return
        t0 = time.time()
        O, Y, res = run(pool, a.stage, np.arange(1000, 1000 + a.episodes * 2), teacher, None, 1.0, True, a.workers)
        say(f"round 0 (teacher rollouts): teacher {summary(res)[2]} | {len(O)} chunk labels [{time.time() - t0:.0f}s]")
        net = build_net()
        if a.init:
            net.load_state_dict(torch.load(str(HERE / a.init), map_location="cpu"))
            say(f"warm start from {a.init}")
        if PUSH_BAND:
            say(f"PUSH-THEN-CLAMP mode: cube x +-12 cm, y +-{PUSH_BAND:g} cm, scripted push (cameras) then the clamp")
        best = -1.0
        for r in range(1, a.rounds + 1):
            loss = fit(net, O, Y, 6000 if r == 1 else 3000)
            torch.save(net.state_dict(), out / "student_last.pt")
            beta = max(0.0, 0.5 - 0.2 * (r - 1))
            O2, Y2, res = run(pool, a.stage, np.arange(5000 * r, 5000 * r + a.episodes), teacher, str(out / "student_last.pt"), beta, True, a.workers)
            O, Y = np.concatenate([O, O2]), np.concatenate([Y, Y2])
            _, _, ev = run(pool, a.stage, np.arange(90000, 90000 + a.eval_episodes), teacher, str(out / "student_last.pt"), 0.0, False, a.workers)
            succ, clean, txt = summary(ev)
            say(f"round {r} (beta {beta:.1f}, L1 {loss:.4f}, {len(O)} labels): student alone {txt} [{time.time() - t0:.0f}s]")
            if succ * clean > best:
                best = succ * clean
                torch.save(net.state_dict(), out / "student.pt")
                (out / "student_meta.json").write_text(json.dumps({
                    "K": K, "scale": SCALE, "ensemble_m": ENSEMBLE_M, "obs_dim": OBS_DIM, "stage": a.stage,
                    "teacher": a.teacher, "round": r, "success": succ, "clean": clean,
                    "output": "(K, 2 arms, 3) tool-point targets relative to the commanded tool point at prediction time, m x 20"}))
                say(f"  new best -> {out / 'student.pt'}")


if __name__ == "__main__":
    main()
