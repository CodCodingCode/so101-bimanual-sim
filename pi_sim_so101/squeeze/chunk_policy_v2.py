#!/usr/bin/env python3
"""Chunked policy (ACT-style action chunking), v2: labels are the TEACHER'S OWN next 8 steps from the student's state.

Student: MLP, inputs = what the rig measures (the 34 numbers of detector_policy.py), output = the next K=8 commanded
tool points per arm relative to the current one (rig_tools/chunking.py format), executed with temporal ensembling.

Labels (the fix): every LABEL_EVERY-th tick the full sim is saved (sim_snapshot.py, bit-exact), the teacher drives
K steps from there (with the same squeeze limiter as the real runner), the commanded tool points it reaches are the
label, and the sim is restored so the student's episode continues untouched. v1 summed the teacher's one-step answers
along the STUDENT's path instead, which piled up corrections (overshoot).

Training conditions (v1's extra +-3 deg servo error is gone: the sim's fitted servo sag already matches the real arms):
  teacher  runs_clamp_v3/grip_teacher (100 % delivered + clean under grip randomisation)
  grip randomisation (CLAMP_GRIP_DR=1), recovery disturbances, the squeeze limiter (commanded grippers >= 1 mm apart)

    python squeeze/chunk_policy_v2.py --rounds 8 --workers 3
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
os.environ.setdefault("CLAMP_GRIP_DR", "1")
os.environ.setdefault("DETECTOR_TEACHER", "runs_clamp_v3/grip_teacher")

import chunking as CH                  # noqa: E402
import detector_policy as DP           # noqa: E402

OUT = HERE / os.environ.get("CHUNK_OUT", "runs_clamp_v3/chunk_policy_v2")
SQUEEZE_MIN_GAP = 0.001
LABEL_EVERY = 2


def build_net():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(DP.OBS_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, CH.K * 6))


def _make_limiter(env, guard, C):
    sim, d = env.base.sim, env.data
    step_m = 0.5 * sim.delta_scale

    def limit(_env, act):
        """The real runner's squeeze limiter: drop the inward x part if the commanded grippers would be < 1 mm apart."""
        q = {}
        for i, s in enumerate(C.SIDES):
            arm = sim.robot[s]
            tgt = np.clip(sim._ee_targets[i] + act.reshape(2, 3)[i] * step_m, sim._workspace_low[i], sim._workspace_high[i])
            jt, _ = sim._solve_ik(arm, tgt, None)
            cur = d.ctrl[arm.actuator_ids]
            q[s] = np.r_[np.rad2deg(cur + np.clip(jt - cur, -sim._rate_limit, sim._rate_limit)), 0.0]
        if guard.distances(q)["arm-arm"][0] < SQUEEZE_MIN_GAP:
            act = act.copy()
            act[0], act[3] = min(act[0], 0.0), max(act[3], 0.0)
            limit.count += 1
        return act
    limit.count = 0
    return limit


def _rollouts(job):
    seeds, net_path, beta, keep = job
    import torch
    import train as T
    import safety as SF
    import clamp_teacher as C
    import sim_snapshot as SS
    from clamp_rl import ClampRLEnv
    torch.set_num_threads(1)
    teacher = T.BCPPO.load(DP.TEACHER_DIR / "best.zip", device="cpu")
    ts = np.load(DP.TEACHER_DIR / "obs_stats.npz")
    tact = lambda env, o: np.asarray(teacher.predict(np.clip((o - ts["mean"]) / np.sqrt(ts["var"] + 1e-8), -10, 10)
                                                     .astype(np.float32), deterministic=True)[0], np.float32)
    net = None
    if net_path:
        net = build_net()
        net.load_state_dict(torch.load(net_path, map_location="cpu"))
        net.eval()
    env = ClampRLEnv(seed=int(seeds[0]))
    sim = env.base.sim
    w = np.full(5, 400.0)
    guard = SF.Guard(env.model, {s: (-w, w) for s in SF.SIDES}, log=lambda *a: None)
    limit = _make_limiter(env, guard, C)
    step_m = 0.5 * sim.delta_scale
    O, Y, res = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) + 4242)
        o_t, _ = env.reset(seed=int(seed))
        det = DP.Detector(rng)
        ens = CH.ChunkEnsembler()
        prev = np.zeros(6, np.float32)
        k, limit.count = 0, 0
        while True:
            o_s = DP.student_obs(env, det, prev)
            if keep and k % LABEL_EVERY == 0:
                ee_now = sim._ee_targets.copy()
                ee, _ = SS.teacher_rollout(env, tact, CH.K, limiter=limit)
                O.append(o_s)
                Y.append(((ee - ee_now) * CH.SCALE).reshape(-1).astype(np.float32))
            if net is not None:
                with torch.no_grad():
                    chunk = net(torch.from_numpy(o_s[None])).numpy()[0]
                ens.add(k, sim._ee_targets, chunk)
            if net is None or rng.random() < beta:
                act = tact(env, o_t)
            else:
                act = CH.action_toward(ens.target(k), sim._ee_targets, step_m)
            act = limit(env, act)
            o_t, _, term, trunc, info = env.step(act)
            prev = np.asarray(act, np.float32)
            k += 1
            if term or trunc:
                break
        res.append((info["perturbation"], bool(info["is_success"]), int(info["bad_free_steps"]), bool(info["can_tipped"]),
                    limit.count))
    return (np.asarray(O, np.float32).reshape(-1, DP.OBS_DIM) if keep else None,
            np.asarray(Y, np.float32).reshape(-1, CH.K * 6) if keep else None, res)


def run(pool, seeds, net_path, beta, keep, workers):
    jobs = [([int(x) for x in c], net_path, beta, keep) for c in np.array_split(seeds, max(workers, len(seeds) // 4)) if len(c)]
    parts = pool.map(_rollouts, jobs, chunksize=1)
    return (np.concatenate([p[0] for p in parts]) if keep else None, np.concatenate([p[1] for p in parts]) if keep else None,
            sum((p[2] for p in parts), []))


def summary(res):
    n = len(res)
    ok = sum(r[1] for r in res)
    clean = sum(r[1] and r[2] == 0 for r in res)
    return clean / n, (f"delivered {ok / n:.2f} clean {clean / n:.2f} box_tipped {sum(r[3] for r in res) / n:.2f} "
                       f"limiter {np.mean([r[4] for r in res]):.0f} ticks/ep ({n} episodes)")


def train(net, O, Y, steps=6000):
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

    say(f"teacher {DP.TEACHER_DIR.name}, grip DR {os.environ.get('CLAMP_GRIP_DR')}, labels = teacher's own {CH.K}-step "
        f"rollouts every {LABEL_EVERY} ticks")
    net, best, O, Y = build_net(), -1.0, None, None
    betas = [1.0, 0.5, 0.3, 0.1] + [0.0] * max(0, a.rounds - 4)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for rnd, beta in enumerate(betas):
            o, y, _ = run(pool, np.arange(500_000 + rnd * 10_000, 500_000 + rnd * 10_000 + a.episodes),
                          str(OUT / "policy.pt") if rnd > 0 else None, beta, True, a.workers)
            O = o if O is None else np.concatenate([O, o])
            Y = y if Y is None else np.concatenate([Y, y])
            loss = train(net, O, Y)
            torch.save(net.state_dict(), OUT / "policy.pt")
            score, txt = summary(run(pool, np.arange(960_000, 960_000 + a.eval), str(OUT / "policy.pt"), 0.0, False, a.workers)[2])
            say(f"round {rnd} (beta {beta}, {len(O)} labelled ticks, mse {loss:.4f}) chunked policy alone: {txt}")
            if score > best:
                best = score
                torch.save(net.state_dict(), OUT / "best.pt")
                say(f"  new best ({best:.2f}) -> best.pt")
            np.savez(OUT / "data.npz", O=O, Y=Y)
    say("done")


if __name__ == "__main__":
    main()
