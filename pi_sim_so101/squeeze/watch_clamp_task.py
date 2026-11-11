#!/usr/bin/env python3
"""Watch the clamp-into-box RL task live in MuJoCo (exactly what PPO trains on): scripted teacher by default, or a
trained policy. Random cube / box placements every episode; prints success and any bad contacts.

    ~/.pyenv/versions/3.10.9/bin/mjpython squeeze/watch_clamp_task.py                 # teacher
    ~/.pyenv/versions/3.10.9/bin/mjpython squeeze/watch_clamp_task.py --policy squeeze/runs_clamp/ppo/best.zip
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
os.environ.setdefault("SQUEEZE_TASK", "clamp")

import mujoco.viewer           # noqa: E402
from clamp_rl import ClampRLEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()
    model = stats = None
    if a.policy:
        import train as T
        model = T.BCPPO.load(a.policy, device="cpu")
        s = np.load(Path(a.policy).parent / "obs_stats.npz")
        stats = (s["mean"], s["var"])
    env = ClampRLEnv(seed=a.seed)
    seed = a.seed
    obs, _ = env.reset(seed=seed)
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.05, 0.05]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.65, -65, -25
        print(f"episode {seed} ({'policy' if model else 'teacher'})", flush=True)
        while v.is_running():
            t0 = time.perf_counter()
            if model is not None:
                o = np.clip((obs - stats[0]) / np.sqrt(stats[1] + 1e-8), -10, 10).astype(np.float32)
                act, _ = model.predict(o, deterministic=True)
            else:
                act = env.expert_action()
            obs, r, term, trunc, info = env.step(act)
            v.sync()
            if term or trunc:
                bad = {k: info[k] for k in info if k.startswith("bad_") and k != "bad_contact_steps" and info[k]}
                print(f"  -> {'SUCCESS (cube in box)' if info['is_success'] else 'failed'} | bad contacts {bad or 'none'}", flush=True)
                time.sleep(1.0)
                seed += 1
                obs, _ = env.reset(seed=seed)
                print(f"episode {seed}", flush=True)
            time.sleep(max(0.0, 0.04 - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
