#!/usr/bin/env python3
"""Train the two-sided squeeze with RL: scripted demos -> behaviour-cloning warm start -> PPO with a BC term.

Why not PPO from scratch: on the one-arm pinch earlier today, pure PPO never learned the grasp
(0/200 after 580k steps) while a demo warm start plus PPO with a BC term worked. The squeeze is harder
(two arms must close on the cube together), so this starts from demos and lets PPO improve on them.

    python squeeze/train.py demos  --episodes 240        # scripted plan with action noise (DART), 12 workers
    python squeeze/train.py bc                             # warm start the PPO actor + critic on the demos
    python squeeze/train.py ppo    --steps 3000000         # PPO fine-tuning with a decaying BC term
    python squeeze/train.py eval   runs/ppo/best.zip --n 100

Everything lands in squeeze/runs/. progress.csv has the task-stage rates, which matter more than reward.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import os
TASK = os.environ.get("SQUEEZE_TASK", "clamp")      # "clamp": two-arm tip clamp into the box (clamp_rl.py)
RUNS = HERE / (("runs_clamp_recovery" if os.environ.get("CLAMP_RECOVERY") == "1" else "runs_clamp") if TASK == "clamp" else "runs")
if os.environ.get("CLAMP_RUNS"):                       # e.g. runs_clamp_v3 (clamp v3: real joint ranges, gentle squeeze)
    RUNS = HERE / os.environ["CLAMP_RUNS"]
DEMOS = RUNS / "demos.npz"
STAGES = ("squeezed", "lifted", "cleared", "placed", "is_success", "can_tipped")
REWARD_SCALE = 0.01   # keeps returns O(10) for the critic; success is ~25 scaled
GAMMA = 0.99
WORKERS = int(os.environ.get("TRAIN_WORKERS", "12"))   # Mac: keep <= 4 (memory)
CLEAN_DEMOS = os.environ.get("CLEAN_DEMOS", "1") == "1"


# ---------------------------------------------------------------- envs
def env_cls():
    from clamp_rl import ClampRLEnv          # the old side-squeeze env (rl_env.py) is archived in _legacy_squeeze/
    return ClampRLEnv


def make_env(rank=0):
    def _init():
        import gymnasium as gym
        env = env_cls()(seed=rank)
        return gym.wrappers.TransformReward(env, lambda r: r * REWARD_SCALE)
    return _init


def vec_env(n, stats=None, start=0):
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
    venv = VecMonitor(SubprocVecEnv([make_env(start + i) for i in range(n)], start_method="fork"),
                      info_keywords=STAGES)
    norm = VecNormalize(venv, norm_obs=True, norm_reward=False, gamma=GAMMA, training=False)
    if stats is not None:
        norm.obs_rms.mean, norm.obs_rms.var = stats
    return norm


# ---------------------------------------------------------------- demos
def _demo_worker(args):
    seeds, noise_levels = args
    env = env_cls()(seed=int(seeds[0]))
    rng = np.random.default_rng(int(seeds[0]) + 12345)
    obs_l, act_l, rew_l, done_l, succ = [], [], [], [], 0
    for k, seed in enumerate(seeds):
        sigma = noise_levels[k % len(noise_levels)]
        o, _ = env.reset(seed=int(seed))
        eo, ea, er, ed = [], [], [], []
        for _ in range(env.base.max_steps):
            label = env.expert_action()
            executed = np.clip(label + rng.normal(0, sigma, label.shape), -1, 1) if sigma else label
            eo.append(o)
            ea.append(label)
            o, r, term, trunc, info = env.step(executed)
            er.append(r * REWARD_SCALE)
            ed.append(term or trunc)
            if term or trunc:
                break
        ed[-1] = True
        clean = bool(info["is_success"]) and not info.get("bad_free_steps", info.get("bad_contact_steps", 0))
        succ += clean
        if clean or not CLEAN_DEMOS:          # only contact-free successes are imitated
            obs_l += eo; act_l += ea; rew_l += er; done_l += ed
    return np.array(obs_l), np.array(act_l), np.array(rew_l), np.array(done_l), succ


def cmd_demos(args):
    RUNS.mkdir(parents=True, exist_ok=True)
    seeds = np.arange(100_000, 100_000 + args.episodes)
    chunks = [([int(x) for x in c], (0.0, 0.1, 0.2, 0.3)) for c in np.array_split(seeds, WORKERS)]
    t0 = time.time()
    with mp.get_context("fork").Pool(WORKERS) as pool:
        parts = pool.map(_demo_worker, chunks)
    obs, act, rew, done = (np.concatenate([p[i] for p in parts]) for i in range(4))
    succ = sum(p[4] for p in parts)
    ret = np.zeros_like(rew)
    running = 0.0
    for i in range(len(rew) - 1, -1, -1):   # Monte Carlo returns, for pretraining the critic
        running = rew[i] + GAMMA * running * (0.0 if done[i] else 1.0)
        ret[i] = running
    np.savez_compressed(DEMOS, obs=obs, act=act, ret=ret, done=done)
    print(f"{args.episodes} demo episodes ({succ} clean successes kept), {len(obs)} steps, {time.time() - t0:.0f}s -> {DEMOS}")


# ---------------------------------------------------------------- policy helpers
def new_model(env, lr=1e-4):
    from stable_baselines3 import PPO
    return BCPPO("MlpPolicy", env, learning_rate=lr, n_steps=256, batch_size=512, n_epochs=5, gamma=GAMMA,
                 gae_lambda=0.95, clip_range=0.1, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.6),
                 device="cpu", verbose=0)


def _bc_steps(policy, obs_n, act, ret, steps, batch=1024, critic=True, weight=1.0):
    import torch
    opt = policy.optimizer
    n = len(obs_n)
    losses = []
    for _ in range(steps):
        idx = np.random.randint(0, n, batch)
        o = torch.as_tensor(obs_n[idx])
        mean = policy.get_distribution(o).distribution.mean
        loss = weight * ((mean - torch.as_tensor(act[idx])) ** 2).mean()
        if critic:
            v = policy.predict_values(o).squeeze(-1)
            loss = loss + 0.5 * ((v - torch.as_tensor(ret[idx])) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    return float(np.mean(losses[-50:]))


def _load_demos():
    d = np.load(DEMOS)
    obs = d["obs"].astype(np.float32)
    mean, var = obs.mean(0), obs.var(0) + 1e-6
    obs_n = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
    return obs_n, d["act"].astype(np.float32), d["ret"].astype(np.float32), (mean.astype(np.float64), var.astype(np.float64))


try:
    from stable_baselines3 import PPO as _PPO

    class BCPPO(_PPO):
        """PPO that also takes a few behaviour-cloning steps on the demos after every update (DAPG-style)."""

        bc_data = None
        bc_weight = 0.0
        bc_steps = 20

        def train(self):
            super().train()
            if self.bc_data is not None and self.bc_weight > 1e-3:
                obs_n, act, ret = self.bc_data
                self.logger.record("train/bc_loss", _bc_steps(self.policy, obs_n, act, ret, self.bc_steps,
                                                              critic=False, weight=self.bc_weight))
except ImportError:   # demos-only machines
    BCPPO = None


# ---------------------------------------------------------------- evaluation
def evaluate(model, stats, n=48, deterministic=True, start=50_000):
    env = vec_env(WORKERS, stats, start=start)
    env.seed(start)
    obs = env.reset()
    counts, done_eps = {k: 0 for k in STAGES}, 0
    while done_eps < n:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, infos = env.step(action)
        for d, info in zip(dones, infos):
            if d and done_eps < n:
                done_eps += 1
                for k in STAGES:
                    counts[k] += bool(info.get(k))
    env.close()
    return {k: v / n for k, v in counts.items()}


def cmd_bc(args):
    obs_n, act, ret, stats = _load_demos()
    env = vec_env(WORKERS, stats)
    model = new_model(env, lr=3e-4)
    model.policy.optimizer.param_groups[0]["lr"] = 3e-4
    for epoch in range(args.epochs):
        loss = _bc_steps(model.policy, obs_n, act, ret, steps=500)
        print(f"bc epoch {epoch}: loss {loss:.4f}")
    env.close()
    out = RUNS / "bc"
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.zip")
    np.savez(out / "obs_stats.npz", mean=stats[0], var=stats[1])
    res = evaluate(model, stats, n=args.n)
    print("BC greedy:", " ".join(f"{k}={v:.2f}" for k, v in res.items()))


def cmd_ppo(args):
    from stable_baselines3.common.callbacks import BaseCallback
    obs_n, act, ret, _ = _load_demos()
    src = Path(args.init_dir) if getattr(args, "init_dir", None) else RUNS / "bc"
    s = np.load(src / "obs_stats.npz")
    stats = (s["mean"], s["var"])
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    env = vec_env(WORKERS, stats)
    model = BCPPO.load(src / ("best.zip" if (src / "best.zip").exists() else "model.zip"), env=env, device="cpu", learning_rate=args.lr)
    model.bc_data = (obs_n, act, ret)
    model.bc_weight = args.bc_weight
    best = {"success": -1.0}

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.f = open(out / "progress.csv", "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(["steps", "ep_rew_mean", *STAGES, "bc_weight", "greedy_success"])
            self.eps = []

        def _on_step(self):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.eps.append({**{k: bool(info.get(k)) for k in STAGES}, "r": info["episode"]["r"]})
            return True

        def _on_rollout_end(self):
            t = self.num_timesteps
            frac = t / args.steps
            self.model.bc_weight = args.bc_weight * max(0.1, 1.0 - frac)   # decays to 10% of its start
            recent = self.eps[-120:]
            greedy = ""
            if t // args.eval_every != (t - self.model.n_steps * WORKERS) // args.eval_every:
                res = evaluate(self.model, stats, n=48)
                greedy = res["is_success"]
                self.model.save(out / f"model_{t}.zip")
                if greedy > best["success"]:
                    best["success"] = greedy
                    self.model.save(out / "best.zip")
                print(f"[{t:>9,}] greedy: " + " ".join(f"{k}={v:.2f}" for k, v in res.items()), flush=True)
            if recent:
                row = [t, np.mean([e["r"] for e in recent])] + [np.mean([e[k] for e in recent]) for k in STAGES]
                self.w.writerow(row + [self.model.bc_weight, greedy])
                self.f.flush()
                if not greedy:
                    print(f"[{t:>9,}] train (stochastic) " + " ".join(f"{k}={v:.2f}" for k, v in zip(STAGES, row[2:])), flush=True)

    res = evaluate(model, stats, n=48)
    print("start (BC) greedy:", " ".join(f"{k}={v:.2f}" for k, v in res.items()), flush=True)
    best["success"] = res["is_success"]
    model.save(out / "best.zip")
    np.savez(out / "obs_stats.npz", mean=stats[0], var=stats[1])
    model.learn(total_timesteps=args.steps, callback=Progress(), reset_num_timesteps=True)
    model.save(out / "final.zip")
    print("best greedy success", best["success"])


def cmd_eval(args):
    path = Path(args.model)
    s = np.load(path.parent / "obs_stats.npz")
    stats = (s["mean"], s["var"])
    model = BCPPO.load(path, device="cpu")
    for det in (True, False):
        res = evaluate(model, stats, n=args.n, deterministic=det, start=900_000)
        print("greedy" if det else "stochastic", " ".join(f"{k}={v:.2f}" for k, v in res.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demos")
    p.add_argument("--episodes", type=int, default=240)
    p = sub.add_parser("bc")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n", type=int, default=48)
    p = sub.add_parser("ppo")
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bc-weight", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250_000)
    p.add_argument("--name", default="ppo")
    p.add_argument("--init-dir", default=None, help="start from this run's best.zip/model.zip + obs_stats (fine-tune)")
    p = sub.add_parser("eval")
    p.add_argument("model")
    p.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    {"demos": cmd_demos, "bc": cmd_bc, "ppo": cmd_ppo, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
