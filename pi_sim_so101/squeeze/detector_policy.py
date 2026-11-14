#!/usr/bin/env python3
"""RL policy that uses the camera through the overhead red-cube DETECTOR (the one that was accurate on the real rig),
instead of raw pixels (an end-to-end pixel student misjudged the real cube by ~4.7 cm on real frames).

Student inputs (everything the rig can measure): per arm joint angles, gripper, tool point by forward kinematics, its
own commanded tool point; the detector's cube (x, y) + a "seen" flag; previous action; the grab-attempt clock.
The detector is modelled like the real one: the cube centre projected through the calibrated overhead camera and
back-projected onto the table plane at cube height (so a lifted cube reads off by the parallax, as on the rig), with
~3 mm noise, and lost while the grippers cover it (the policy then gets the last position it saw).
Teacher: the trained RL policy (state-based, sim-privileged inputs); trained by DAgger.

    python squeeze/detector_policy.py --rounds 6 --workers 3
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
sys.path[:0] = [str(HERE), str(HERE.parent)]
os.environ.setdefault("SQUEEZE_TASK", "clamp")
os.environ.setdefault("CLAMP_RECOVERY", "1")

OUT = HERE / os.environ.get("DETECTOR_OUT", "runs_clamp_v3/detector_policy")
TEACHER_DIR = HERE / os.environ.get("DETECTOR_TEACHER", "runs_clamp_v3/dagger")
# SO101_RIG: one switch for every sim's cameras (the rig built from robots/so101/calibration by
# rig_tools/build_rig_from_calibration.py once the wrist refit is validated)
_ACTIVE_RIG = HERE.parents[1] / "rig_tools" / "data" / "so101_camera_rig.active.json"   # apply_calibration.py local
RIG = (Path(os.environ["SO101_RIG"]) if os.environ.get("SO101_RIG") else
       _ACTIVE_RIG if _ACTIVE_RIG.exists() else HERE.parents[1] / "so101_actuation" / "cameras" / "so101_camera_rig.json")
NOISE_M = 0.003
OBS_DIM = 34


class Detector:
    """The real overhead red-cube detector, modelled: project the cube centre through the calibrated camera (with
    lens distortion), back-project the pixel onto the plane z = cube half height."""

    def __init__(self, rng):
        r = json.loads(RIG.read_text())["cameras"]["overhead"]
        self.fx, self.fy, self.cx, self.cy = (r["intrinsics"][k] for k in ("fx", "fy", "cx", "cy"))
        self.k1, self.k2 = r["distortion"][:2]
        self.pos = np.array(r["pose"]["pos"])
        w, x, y, z = r["pose"]["quat_wxyz"]
        self.R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                           [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                           [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        self.rng = rng
        self.last, self.seen = None, 0.0

    def _project(self, p):
        c = self.R.T @ (p - self.pos)
        xn, yn = c[0] / -c[2], -c[1] / -c[2]
        r2 = xn * xn + yn * yn
        dd = 1 + self.k1 * r2 + self.k2 * r2 * r2
        return self.cx + self.fx * xn * dd, self.cy + self.fy * yn * dd

    def _backproject(self, u, v, zp):
        xd, yd = (u - self.cx) / self.fx, (v - self.cy) / self.fy
        xn, yn = xd, yd
        for _ in range(30):
            r2 = xn * xn + yn * yn
            dd = 1 + self.k1 * r2 + self.k2 * r2 * r2
            xn, yn = xd / dd, yd / dd
        ray = self.R @ np.array([xn, -yn, -1.0])
        return self.pos + (zp - self.pos[2]) / ray[2] * ray

    def read(self, cube_xyz, tips_xy, half):
        """Detector reading (x, y) and the 'seen' flag; holds the last reading while the cube is covered."""
        covered = min(np.linalg.norm(t - cube_xyz[:2]) for t in tips_xy) < 0.025 and self.rng.random() < 0.6
        if not covered:
            u, v = self._project(np.asarray(cube_xyz, float))
            xy = self._backproject(u, v, half)[:2] + self.rng.normal(0, NOISE_M, 2)
            self.last, self.seen = xy, 1.0
        else:
            self.seen = 0.0
        return self.last, self.seen


def student_obs(env, det, prev_action):
    import clamp_teacher as C
    b, sim = env.base, env.base.sim
    tips = [sim.robot[s].position[:2] for s in C.SIDES]
    xy, seen = det.read(b.cube_pos(), tips, C.CUBE_HALF)
    parts = []
    for i, s in enumerate(C.SIDES):
        arm = sim.robot[s]
        parts += [np.rad2deg(b.data.qpos[arm.qpos_ids]) / 90.0, [arm.gripper_opening], arm.position * 5.0,
                  sim._ee_targets[i] * 5.0]
    parts += [[xy[0] / 0.05, (xy[1] - C.CUBE_Y) / 0.05, seen], prev_action,
              [(b._step - env._attempt_start) / 450.0]]
    return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])


def build_net():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(OBS_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 6), nn.Tanh())


def _rollouts(job):
    seeds, net_path, beta, keep = job
    import torch
    import train as T
    from clamp_rl import ClampRLEnv
    torch.set_num_threads(1)
    teacher = T.BCPPO.load(TEACHER_DIR / "best.zip", device="cpu")
    ts = np.load(TEACHER_DIR / "obs_stats.npz")
    tnorm = lambda o: np.clip((o - ts["mean"]) / np.sqrt(ts["var"] + 1e-8), -10, 10).astype(np.float32)
    net = None
    if net_path:
        net = build_net()
        net.load_state_dict(torch.load(net_path, map_location="cpu"))
        net.eval()
    env = ClampRLEnv(seed=int(seeds[0]))
    O, A, res = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) + 99)
        o_t, _ = env.reset(seed=int(seed))
        det = Detector(rng)
        prev = np.zeros(6, np.float32)
        while True:
            o_s = student_obs(env, det, prev)
            label = np.asarray(teacher.predict(tnorm(o_t), deterministic=True)[0], np.float32)
            if keep:
                O.append(o_s)
                A.append(label)
            if net is None or rng.random() < beta:
                act = label
            else:
                with torch.no_grad():
                    act = net(torch.from_numpy(o_s[None])).numpy()[0]
            o_t, _, term, trunc, info = env.step(act)
            prev = np.asarray(act, np.float32)
            if term or trunc:
                break
        res.append((info["perturbation"], bool(info["is_success"]), int(info["bad_free_steps"]), bool(info["can_tipped"])))
    return np.array(O, np.float32).reshape(-1, OBS_DIM), np.array(A, np.float32).reshape(-1, 6), res


def run(pool, seeds, net_path, beta, keep, workers):
    jobs = [([int(x) for x in c], net_path, beta, keep) for c in np.array_split(seeds, workers) if len(c)]
    parts = pool.map(_rollouts, jobs, chunksize=1)
    return (np.concatenate([p[0] for p in parts]) if keep else None, np.concatenate([p[1] for p in parts]) if keep else None,
            sum((p[2] for p in parts), []))


def summary(res):
    n = len(res)
    ok = sum(r[1] for r in res)
    clean = sum(r[1] and r[2] == 0 for r in res)
    return clean / n, f"success {ok / n:.2f} clean {clean / n:.2f} box_tipped {sum(r[3] for r in res) / n:.2f} ({n} episodes)"


def train(net, O, A, steps=4000):
    import torch
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    Ot, At = torch.from_numpy(O), torch.from_numpy(A)
    for k in range(steps):
        idx = torch.randint(0, len(O), (1024,))
        x = Ot[idx] + torch.randn(1024, OBS_DIM) * 0.01
        loss = ((net(x) - At[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--eval", type=int, default=48)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--resume", action="store_true", help="continue from policy.pt + data.npz with policy-only rounds")
    ap.add_argument("--init", default=None, help="start the student from these weights (fresh data, new teacher)")
    a = ap.parse_args()
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "log.txt", "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    net = build_net()
    best = -1.0
    betas = [1.0, 0.5, 0.3, 0.1] + [0.0] * max(0, a.rounds - 3)
    start_round = 0
    if a.init:
        net.load_state_dict(torch.load(a.init))
        betas = [0.5, 0.3, 0.1] + [0.0] * max(0, a.rounds - 3)
        say(f"init from {a.init}; teacher {TEACHER_DIR}; grip DR {os.environ.get('CLAMP_GRIP_DR') == '1'}")
    if a.resume:
        net.load_state_dict(torch.load(OUT / "policy.pt"))
        z = np.load(OUT / "data.npz")
        best = float(z["best"])
        start_round = int(z["rounds"])
        betas = [0.0] * a.rounds
        say(f"resume: {len(z['O'])} labelled steps, best clean {best:.2f}, {a.rounds} more policy-only rounds")
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        O = z["O"] if a.resume else None
        A = z["A"] if a.resume else None
        for rnd, beta in enumerate(betas, start=start_round):
            path = str(OUT / "policy.pt") if (rnd > 0 or a.init) else None
            if a.init and rnd == 0:
                torch.save(net.state_dict(), OUT / "policy.pt")
            o, ac, res = run(pool, np.arange(300_000 + rnd * 10_000, 300_000 + rnd * 10_000 + a.episodes), path, beta, True,
                             a.workers)
            O = o if O is None else np.concatenate([O, o])
            A = ac if A is None else np.concatenate([A, ac])
            loss = train(net, O, A)
            torch.save(net.state_dict(), OUT / "policy.pt")
            score, txt = summary(run(pool, np.arange(900_000, 900_000 + a.eval), str(OUT / "policy.pt"), 0.0, False, a.workers)[2])
            say(f"round {rnd} (beta {beta}, {len(O)} labelled steps, mse {loss:.4f}) policy alone: {txt}")
            if score > best:
                best = score
                torch.save(net.state_dict(), OUT / "best.pt")
                say(f"  new best ({best:.2f}) -> best.pt")
            np.savez(OUT / "data.npz", O=O, A=A, best=best, rounds=rnd + 1)
    say("done")


if __name__ == "__main__":
    main()
