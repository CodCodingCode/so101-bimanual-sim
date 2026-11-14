#!/usr/bin/env python3
"""Evaluate the SmolVLA policy in the sim: the sim renders the three calibrated cameras exactly as the dataset was made
(native 640x480 -> 320x240 -> JPEG), sends them + the measured joints to the laptop policy server (SSH tunnel
localhost:8790), and executes the returned joint targets (first --exec of each 50-step chunk) through the sim's own
rate limiter. Success / clean exactly as for the other policies (ClampRLEnv info).

    ssh -N -L 8790:127.0.0.1:8790 dzkchen &      # tunnel
    python squeeze/vla_sim_eval.py --episodes 10 --exec 10
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
os.environ.setdefault("SQUEEZE_TASK", "clamp")
os.environ.setdefault("CLAMP_RECOVERY", "0")
os.environ.setdefault("CLAMP_GRIP_DR", "1")

URL = "http://127.0.0.1:8790"


def ask(imgs, state):
    buf = io.BytesIO()
    np.savez(buf, camera1=imgs[0], camera2=imgs[1], camera3=imgs[2], state=state.astype(np.float32))
    req = urllib.request.Request(URL + "/chunk", data=buf.getvalue(), method="POST")
    z = np.load(io.BytesIO(urllib.request.urlopen(req, timeout=60).read()))
    return z["chunk"], float(z["ms"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--exec", type=int, default=10, help="steps of each 50-step chunk executed before re-planning")
    ap.add_argument("--seed0", type=int, default=990_000)
    ap.add_argument("--randomize", action="store_true", help="appearance randomisation as in the training data")
    ap.add_argument("--video", default=None, help="save an mp4 of the overhead + wrist views of the first episode")
    ap.add_argument("--look", default=None, choices=("ours", "alt"),
                    help="scene look: 'ours' = the training-data randomisation (same as --randomize), 'alt' = gen_vla_sim's "
                         "yellow-arm scene (scene-transfer experiment)")
    ap.add_argument("--url", default=None, help="policy server (default http://127.0.0.1:8790)")
    a = ap.parse_args()
    global URL
    URL = a.url or URL
    import mujoco
    from PIL import Image
    import distill as D
    env = D.make_env(a.seed0)
    if a.look == "alt":
        import gen_vla_sim as G
        rnd = G._alt_randomizer(D)(env)
    else:
        rnd = D.Randomizer(env) if (a.randomize or a.look == "ours") else None
    sim, d = env.base.sim, env.base.data
    q_ids = [sim.robot[s].qpos_ids for s in ("left", "right")]
    arms = [sim.robot["left"], sim.robot["right"]]
    override = {"q": None}
    solve = sim._solve_ik

    def solve_or_override(arm, pos, quat, downward=None):
        if override["q"] is not None:
            return override["q"][0 if arm is arms[0] else 1], 0.0
        return solve(arm, pos, quat, downward)
    sim._solve_ik = solve_or_override

    def render():
        mujoco.mj_forward(sim.model, d)
        out = []
        for c in D.CAMS:
            im = Image.fromarray(sim.render_camera(c, native=True)).resize((320, 240), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            out.append(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
        return out

    res, frames = [], []
    for ep in range(a.episodes):
        seed = a.seed0 + ep
        env.reset(seed=seed)
        if rnd is not None:                          # after reset (reset restores the nominal look)
            rnd.apply(np.random.default_rng(seed + 5))
        plan, k, ms, after = None, 0, [], 0
        t0 = time.time()
        for t in range(env.base.max_steps):
            if plan is None or k >= a.exec:
                imgs = render()
                state = np.rad2deg(np.r_[d.qpos[q_ids[0]], d.qpos[q_ids[1]]])
                plan, m = ask(imgs, state)
                ms.append(m)
                k = 0
                if a.video and ep == 0:
                    frames.append(np.concatenate(imgs, 1))
            override["q"] = np.deg2rad(plan[k].reshape(2, 5))
            k += 1
            _, _, term, trunc, info = env.step(np.zeros(6, np.float32))
            if info["is_success"]:
                after += 1
            if term or trunc or after > 12:
                break
        override["q"] = None
        clean = info["is_success"] and info["bad_free_steps"] == 0
        res.append((info["is_success"], clean, info["squeezed"], info["lifted"]))
        print(f"episode {ep} seed {seed}: delivered {info['is_success']} clean {clean} | squeezed {info['squeezed']} "
              f"lifted {info['lifted']} placed {info['placed']} | contacts table {info['bad_arm_table']} box "
              f"{info['bad_arm_box']} arm-arm {info['bad_arm_arm']} camera {info['bad_arm_camera']} | {t + 1} ticks, "
              f"inference {np.mean(ms):.0f} ms, {time.time() - t0:.0f} s", flush=True)
    n = len(res)
    print(f"SmolVLA in sim, exec {a.exec}/50: delivered {sum(r[0] for r in res)}/{n} clean {sum(r[1] for r in res)}/{n} "
          f"squeezed {sum(r[2] for r in res)}/{n} lifted {sum(r[3] for r in res)}/{n}", flush=True)
    if a.video and frames:
        import imageio
        imageio.mimsave(a.video, frames, fps=25 / a.exec * 2)


if __name__ == "__main__":
    main()
