#!/usr/bin/env python3
"""Recovery training by DAgger (instead of PPO): the policy drives, the re-planning teacher labels every state it
reaches, the policy is retrained on everything collected so far. Starts from the 100 %-clean base policy and keeps
its observation normalisation, so the clean behaviour is the starting point and the clean demos stay in the data.

    SQUEEZE_TASK=clamp CLAMP_RECOVERY=1 python squeeze/dagger.py --rounds 6

Writes squeeze/runs_clamp_recovery/dagger/{model.zip,best.zip,obs_stats.npz,log.txt}. "clean" = success with no
arm contact outside the short window right after a disturbance.
"""
from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
os.environ.setdefault("SQUEEZE_TASK", "clamp")
os.environ.setdefault("CLAMP_RECOVERY", "1")

import train as T                      # noqa: E402

BASE = HERE / os.environ.get("DAGGER_BASE", "runs_clamp/ppo")
OUT = HERE / os.environ.get("DAGGER_OUT", "runs_clamp_recovery/dagger")
EVAL_SEED = 90_000


def _norm(obs, stats):
    return np.clip((obs - stats[0]) / np.sqrt(stats[1] + 1e-8), -10, 10).astype(np.float32)


def _rollouts(args):
    """Run episodes with policy/teacher mixing beta; return (normalised obs, teacher labels, per-episode results)."""
    model_path, seeds, beta, keep = args
    import torch
    torch.set_num_threads(1)
    from clamp_rl import ClampRLEnv
    s = np.load(OUT / "obs_stats.npz")
    stats = (s["mean"], s["var"])
    model = T.BCPPO.load(model_path, device="cpu")
    env = ClampRLEnv(seed=int(seeds[0]))
    rng = np.random.default_rng(int(seeds[0]) + 7)
    O, A, res = [], [], []
    for seed in seeds:
        o, _ = env.reset(seed=int(seed))
        while True:
            label = env.expert_action()
            on = _norm(o, stats)
            if keep:
                O.append(on)
                A.append(label)
            if beta > 0 and rng.random() < beta:
                act = label
            else:
                act, _ = model.predict(on, deterministic=True)
            o, r, term, trunc, info = env.step(act)
            if term or trunc:
                break
        res.append((info["perturbation"], bool(info["is_success"]), int(info["bad_free_steps"]), bool(info["can_tipped"])))
    return (np.array(O, np.float32).reshape(-1, len(o)), np.array(A, np.float32).reshape(-1, 6), res)


def _run(pool, model_path, seeds, beta, keep, workers):
    chunks = [(str(model_path), [int(x) for x in c], beta, keep) for c in np.array_split(seeds, workers) if len(c)]
    parts = pool.map(_rollouts, chunks)
    O = np.concatenate([p[0] for p in parts]) if keep else None
    A = np.concatenate([p[1] for p in parts]) if keep else None
    return O, A, sum((p[2] for p in parts), [])


def _summary(res):
    n, ok, clean, tip = collections.Counter(), collections.Counter(), collections.Counter(), 0
    for pert, s, bad, t in res:
        n[pert] += 1
        ok[pert] += s
        clean[pert] += s and bad == 0
        tip += t
    tot = len(res)
    score = sum(clean.values()) / tot
    txt = (f"success {sum(ok.values()) / tot:.2f} clean {score:.2f} box_tipped {tip / tot:.2f} | clean by disturbance "
           + " ".join(f"{k}={clean[k]}/{n[k]}" for k in ("none", "nudge", "slip", "push") if n[k]))
    return score, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--eval", type=int, default=60)
    ap.add_argument("--betas", default="0.5,0.3,0.2,0.1,0,0")
    ap.add_argument("--bc-steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "log.txt", "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    shutil.copy(BASE / "obs_stats.npz", OUT / "obs_stats.npz")
    s = np.load(OUT / "obs_stats.npz")
    stats = (s["mean"], s["var"])
    model = T.BCPPO.load(BASE / ("best.zip" if (BASE / "best.zip").exists() else "model.zip"), device="cpu")
    model.policy.optimizer.param_groups[0]["lr"] = a.lr
    model.save(OUT / "model.zip")
    d = np.load(T.DEMOS)
    O, A = _norm(d["obs"].astype(np.float32), stats), d["act"].astype(np.float32)
    say(f"start: {len(O)} clean demo steps from {T.DEMOS.name}")
    betas = [float(x) for x in a.betas.split(",")]
    eval_seeds = np.arange(EVAL_SEED, EVAL_SEED + a.eval)
    with mp.get_context("spawn").Pool(T.WORKERS) as pool:     # spawn: fork after torch use deadlocks
        best, txt = _summary(_run(pool, OUT / "model.zip", eval_seeds, 0.0, False, T.WORKERS)[2])
        say(f"round 0 (base policy) greedy: {txt}")
        shutil.copy(OUT / "model.zip", OUT / "best.zip")
        for rnd in range(1, a.rounds + 1):
            beta = betas[min(rnd - 1, len(betas) - 1)]
            seeds = np.arange(200_000 + rnd * 10_000, 200_000 + rnd * 10_000 + a.episodes)
            o, ac, res = _run(pool, OUT / "model.zip", seeds, beta, True, T.WORKERS)
            O, A = np.concatenate([O, o]), np.concatenate([A, ac])
            loss = T._bc_steps(model.policy, O, A, np.zeros(len(O), np.float32), a.bc_steps, critic=False)
            model.save(OUT / "model.zip")
            score, txt = _summary(_run(pool, OUT / "model.zip", eval_seeds, 0.0, False, T.WORKERS)[2])
            say(f"round {rnd} (beta {beta}, +{len(o)} labelled steps, {len(O)} total, bc mse {loss:.4f}) greedy: {txt}")
            if score > best:
                best = score
                shutil.copy(OUT / "model.zip", OUT / "best.zip")
                say(f"  new best ({best:.2f}) -> best.zip")
    say("done")


if __name__ == "__main__":
    main()
