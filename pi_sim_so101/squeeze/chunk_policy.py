#!/usr/bin/env python3
"""Chunked RL policy (ACT-style action chunking) for the clamp, with the real rig's quirks in training.

Student: MLP, inputs = what the rig measures (same 34 numbers as detector_policy.py: joints, gripper, tool point by
forward kinematics, commanded tool point, the overhead-equivalent cube reading + seen flag, previous action, clock),
output = the next K=8 tool-point targets per arm (rig_tools/chunking.py format), executed with temporal ensembling.
Teacher: the trained RL policy (state-based). DAgger: chunk labels are the teacher's actions at the next 8 states.

Real-rig behaviour put into training (2026-11-12 live runs):
  - SERVO SHORTFALL: every episode each arm joint settles a random 0-3 deg off its command (either sign), via the
    position actuators' constant bias term -- the real servos (P 16) stop 1-3 deg short.
  - SQUEEZE LIMITER: like rig_tools/real_policy.py, the commanded grippers may not come closer than 1 mm; the inward
    part of the step is dropped. (On the real arms the old policy kept asking for a tighter squeeze and stalled ~6 s.)

    python squeeze/chunk_policy.py --rounds 8 --workers 3
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "rig_tools")]
os.environ.setdefault("SQUEEZE_TASK", "clamp")
os.environ.setdefault("CLAMP_RECOVERY", "1")

import chunking as CH                  # noqa: E402
import detector_policy as DP           # noqa: E402

OUT = HERE / "runs_clamp_v3" / "chunk_policy"
SHORTFALL_DEG = 3.0
SQUEEZE_MIN_GAP = 0.001


def build_net():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(DP.OBS_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, CH.K * 6))


def _rollouts(job):
    seeds, net_path, beta, keep = job

    import torch
    import train as T
    import safety as SF
    import clamp_teacher as C
    from clamp_rl import ClampRLEnv
    torch.set_num_threads(1)
    teacher = T.BCPPO.load(DP.TEACHER_DIR / "best.zip", device="cpu")
    ts = np.load(DP.TEACHER_DIR / "obs_stats.npz")
    tnorm = lambda o: np.clip((o - ts["mean"]) / np.sqrt(ts["var"] + 1e-8), -10, 10).astype(np.float32)
    net = None
    if net_path:
        net = build_net()
        net.load_state_dict(torch.load(net_path, map_location="cpu"))
        net.eval()
    env = ClampRLEnv(seed=int(seeds[0]))
    m, d, sim = env.model, env.data, env.base.sim
    w = np.full(5, 400.0)
    guard = SF.Guard(m, {s: (-w, w) for s in SF.SIDES}, log=lambda *a: None)
    acts = [m.actuator(f"{s}_{j}").id for s in C.SIDES
            for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]
    step_m = 0.5 * sim.delta_scale
    O, Y, res = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) + 4242)
        # servo shortfall: each joint settles a random 0-3 deg off its command this episode
        bias = np.radians(rng.uniform(0, SHORTFALL_DEG, len(acts)) * rng.choice([-1, 1], len(acts)))
        m.actuator_biasprm[acts, 0] = -m.actuator_gainprm[acts, 0] * bias
        o_t, _ = env.reset(seed=int(seed))
        det = DP.Detector(rng)
        ens = CH.ChunkEnsembler()
        prev = np.zeros(6, np.float32)
        obs_ep, lab_ep, k, limited = [], [], 0, 0
        while True:
            o_s = DP.student_obs(env, det, prev)
            label = np.asarray(teacher.predict(tnorm(o_t), deterministic=True)[0], np.float32)
            obs_ep.append(o_s)
            lab_ep.append(label)
            if net is not None:
                with torch.no_grad():
                    chunk = net(torch.from_numpy(o_s[None])).numpy()[0]
                ens.add(k, sim._ee_targets, chunk)
            if net is None or rng.random() < beta:
                act = label.copy()
            else:
                act = CH.action_toward(ens.target(k), sim._ee_targets, step_m)
            # squeeze limiter (as on the real runner): commanded grippers never closer than 1 mm
            q = {}
            for i, s in enumerate(C.SIDES):
                arm = sim.robot[s]
                tgt = np.clip(sim._ee_targets[i] + act.reshape(2, 3)[i] * step_m, sim._workspace_low[i], sim._workspace_high[i])
                jt, _ = sim._solve_ik(arm, tgt, None)
                cur = d.ctrl[arm.actuator_ids]
                q[s] = np.r_[np.rad2deg(cur + np.clip(jt - cur, -sim._rate_limit, sim._rate_limit)), 0.0]
            if guard.distances(q)["arm-arm"][0] < SQUEEZE_MIN_GAP:
                act[0], act[3] = min(act[0], 0.0), max(act[3], 0.0)
                limited += 1
            o_t, _, term, trunc, info = env.step(act)
            prev = np.asarray(act, np.float32)
            k += 1
            if term or trunc:
                break
        if keep:
            O.append(np.asarray(obs_ep, np.float32))
            Y.append(CH.chunk_labels(lab_ep, step_m))
        res.append((info["perturbation"], bool(info["is_success"]), int(info["bad_free_steps"]), bool(info["can_tipped"]), limited))
    return (np.concatenate(O) if keep else None, np.concatenate(Y) if keep else None, res)


def run(pool, seeds, net_path, beta, keep, workers):
    jobs = [([int(x) for x in c], net_path, beta, keep) for c in np.array_split(seeds, workers) if len(c)]
    parts = pool.map(_rollouts, jobs, chunksize=1)
    return (np.concatenate([p[0] for p in parts]) if keep else None, np.concatenate([p[1] for p in parts]) if keep else None,
            sum((p[2] for p in parts), []))


def summary(res):
    n = len(res)
    ok = sum(r[1] for r in res)
    clean = sum(r[1] and r[2] == 0 for r in res)
    lim = np.mean([r[4] for r in res])
    return clean / n, f"success {ok / n:.2f} clean {clean / n:.2f} box_tipped {sum(r[3] for r in res) / n:.2f} limiter {lim:.0f} ticks/ep ({n} episodes)"


def train(net, O, Y, steps=5000):
    import torch
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    Ot, Yt = torch.from_numpy(O), torch.from_numpy(Y)
    for _ in range(steps):
        idx = torch.randint(0, len(O), (1024,))
        loss = ((net(Ot[idx] + torch.randn(1024, DP.OBS_DIM) * 0.01) - Yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--eval", type=int, default=32)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "log.txt", "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    net, best, O, Y = build_net(), -1.0, None, None
    betas = [1.0, 0.5, 0.3, 0.1] + [0.0] * max(0, a.rounds - 3)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for rnd, beta in enumerate(betas):
            o, y, _ = run(pool, np.arange(400_000 + rnd * 10_000, 400_000 + rnd * 10_000 + a.episodes),
                          str(OUT / "policy.pt") if rnd > 0 else None, beta, True, a.workers)
            O = o if O is None else np.concatenate([O, o])
            Y = y if Y is None else np.concatenate([Y, y])
            loss = train(net, O, Y)
            torch.save(net.state_dict(), OUT / "policy.pt")
            score, txt = summary(run(pool, np.arange(950_000, 950_000 + a.eval), str(OUT / "policy.pt"), 0.0, False, a.workers)[2])
            say(f"round {rnd} (beta {beta}, {len(O)} labelled steps, mse {loss:.4f}) chunked policy alone, servo "
                f"shortfall 0-3 deg + squeeze limiter: {txt}")
            if score > best:
                best = score
                torch.save(net.state_dict(), OUT / "best.pt")
                say(f"  new best ({best:.2f}) -> best.pt")
    say("done")


if __name__ == "__main__":
    main()
