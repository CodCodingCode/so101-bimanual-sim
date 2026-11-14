#!/usr/bin/env python3
"""Sim episodes for VLA (SmolVLA) co-training, recorded exactly like a real run (rig_tools/real_policy.py logs):

  images  overhead / left_wrist / right_wrist, rendered through the CALIBRATED camera rig at the native 640x480,
          resized to 320x240 and JPEG-encoded (the real runs save 320x240 per camera); rendered every 2nd tick and
          held in between (the real runs keep a frame every 5th tick)
  state   measured joint angles, 5 per arm (sim degrees = real units, offsets included)
  action  commanded joint targets after IK + rate limit, 5 per arm (the real log's "cmd")

Teacher: runs_clamp_v3/grip_teacher (state RL policy; 100 % delivered + clean under grip randomisation).
Per episode: appearance randomisation around the real rig's measured colours (distill.Randomizer), grip
randomisation (CLAMP_GRIP_DR=1), the recovery disturbances (nudge / slip / push). Recording stops ~0.5 s after the cube
settles in the box. Only delivered + clean episodes are kept.

    python squeeze/gen_vla_sim.py --episodes 240 --workers 3
"""
from __future__ import annotations

import argparse
import io
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
os.environ.setdefault("CLAMP_GRIP_DR", "1")

OUT = HERE / "runs_clamp_v3" / "vla_sim"
TEACHER_DIR = HERE / "runs_clamp_v3" / "grip_teacher"
RENDER_EVERY = 2
W, H = 320, 240
AFTER_SUCCESS = 12
TASK = "squeeze the red cube between both grippers, lift it and drop it into the grey box"
# --look alt: a scene that looks NOTHING like ours (scene-transfer / LoRA-connector experiment, user 2026-11-13). Same
# bimanual layout, calibrated cameras, task, teacher; different appearance: stock-yellow arms and printed mounts, flat
# blue-grey table, beige room, warmer brighter light with shadows. Cube and box keep their colours (the instruction
# names them). Mild jitter around this look (a consistent source domain).
ALT = {"arm": (0.95, 0.74, 0.10), "table": (0.28, 0.40, 0.52), "room": (0.82, 0.78, 0.66)}


def _alt_randomizer(D):
    class AltLook(D.Randomizer):
        def apply(self, rng):
            m = self.m
            self.restore_materials()
            self._set(self.arm_light, np.array(ALT["arm"]) * rng.uniform(0.9, 1.08) + rng.normal(0, 0.015, 3))
            m.light_castshadow[:] = 1
            if self.table_mat >= 0:
                for g in self.table:
                    m.geom_matid[g] = -1                       # no wood texture
            self._set(self.table, np.array(ALT["table"]) * rng.uniform(0.9, 1.1) + rng.normal(0, 0.02, 3))
            self._set(self.cube, self.E.CUBE_RGB * rng.uniform(0.9, 1.1) + rng.normal(0, 0.02, 3))
            self._set(self.box, self.E.BOX_RGB * rng.uniform(0.9, 1.2) + rng.normal(0, 0.01, 3))
            for g in self.room:
                m.geom_matid[g] = -1
            self._set(self.room, np.array(ALT["room"]) * rng.uniform(0.9, 1.08))
            d0, a0, p0 = self.light0
            m.light_diffuse[:] = d0 * rng.uniform(1.1, 1.35) * np.array([1.0, 0.95, 0.85])
            m.light_ambient[:] = a0 * rng.uniform(0.9, 1.2)
            m.light_pos[:] = p0 + rng.normal(0, 0.1, p0.shape)
            cp, cq = self.cam0
            m.cam_pos[:] = cp + rng.normal(0, 0.002, cp.shape)
            cq2 = cq + rng.normal(0, 0.006, cq.shape)
            m.cam_quat[:] = cq2 / np.linalg.norm(cq2, axis=1, keepdims=True)
    return AltLook


def _episodes(job):
    seeds, out, look = job
    import torch
    from PIL import Image
    import train as T
    import distill as D
    torch.set_num_threads(1)
    teacher = T.BCPPO.load(TEACHER_DIR / "best.zip", device="cpu")
    ts = np.load(TEACHER_DIR / "obs_stats.npz")
    tnorm = lambda o: np.clip((o - ts["mean"]) / np.sqrt(ts["var"] + 1e-8), -10, 10).astype(np.float32)
    env = D.make_env(int(seeds[0]))
    rnd = (_alt_randomizer(D) if look == "alt" else D.Randomizer)(env)
    sim, d = env.base.sim, env.base.data
    q_ids = [sim.robot[s].qpos_ids for s in ("left", "right")]
    a_ids = [sim.robot[s].actuator_ids for s in ("left", "right")]
    res = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) + 31337)
        obs, _ = env.reset(seed=int(seed))
        rnd.apply(rng)          # AFTER reset: the sim's reset restores the nominal colours / lights / camera poses
        frames = {c: [] for c in D.CAMS}
        S, A, F, CUBE = [], [], [], []
        after = 0
        for t in range(env.base.max_steps):
            if t % RENDER_EVERY == 0:
                import mujoco
                mujoco.mj_forward(sim.model, d)
                for c in D.CAMS:
                    im = Image.fromarray(sim.render_camera(c, native=True)).resize((W, H), Image.Resampling.BILINEAR)
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=90)
                    frames[c].append(buf.getvalue())
            F.append(len(frames[D.CAMS[0]]) - 1)
            S.append(np.rad2deg(np.r_[d.qpos[q_ids[0]], d.qpos[q_ids[1]]]))
            CUBE.append(env.base.cube_pos().copy())
            act = np.asarray(teacher.predict(tnorm(obs), deterministic=True)[0], np.float32)
            obs, _, term, trunc, info = env.step(act)
            A.append(np.rad2deg(np.r_[d.ctrl[a_ids[0]], d.ctrl[a_ids[1]]]))
            if info["is_success"]:
                after += 1
            if term or trunc or after > AFTER_SUCCESS:
                break
        ok, clean = bool(info["is_success"]), bool(info["is_success"]) and info["bad_free_steps"] == 0
        if clean:
            np.savez(out / f"ep_{seed}.npz", state=np.asarray(S, np.float32), action=np.asarray(A, np.float32),
                     frame_idx=np.asarray(F, np.int32), cube=np.asarray(CUBE, np.float32),
                     **{f"jpg_{c}": np.array(frames[c], dtype=object) for c in D.CAMS},
                     task=TASK, fps=25, grip=np.array([info["grip_mu"], info["grip_stiffness"]]),
                     perturbation=info["perturbation"])
        res.append((int(seed), ok, clean, len(S), info["perturbation"]))
    return res


def preview(look, path):
    from PIL import Image
    import mujoco
    import distill as D
    import clamp_teacher as C
    env = D.make_env(1)
    rnd = (_alt_randomizer(D) if look == "alt" else D.Randomizer)(env)
    env.reset(seed=1)
    rnd.apply(np.random.default_rng(3))      # after reset (reset restores the nominal look)
    sim = env.base.sim
    plan = C.ClampPlan((*env.base.cube_pos()[:2], 0.0), tuple(env.base.can_pos()[:2]))
    for _ in range(120):                                   # into the clamp, so the grippers are in view
        tg = plan.next_targets(sim._ee_targets.copy())
        u = np.zeros((2, 4))
        u[:, :3] = (tg - sim._ee_targets) / sim.delta_scale
        env.base.step(np.clip(u, -1, 1).ravel())
    mujoco.mj_forward(sim.model, sim.data)
    ims = [Image.fromarray(sim.render_camera(c, native=True)).resize((W, H)) for c in D.CAMS]
    Image.fromarray(np.concatenate([np.asarray(i) for i in ims], 1)).save(path, quality=88)
    print("preview ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=1_200_000)
    ap.add_argument("--look", default="ours", choices=("ours", "alt"))
    ap.add_argument("--preview", default=None, help="just render one start frame per camera to this jpg")
    ap.add_argument("--out", default=None, help="output folder (default runs_clamp_v3/vla_sim or vla_sim_alt)")
    a = ap.parse_args()
    out_dir = Path(a.out) if a.out else (OUT if a.look == "ours" else OUT.parent / "vla_sim_alt")
    if a.preview:
        return preview(a.look, a.preview)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = np.arange(a.seed0, a.seed0 + a.episodes)
    jobs = [([int(x) for x in c], out_dir, a.look) for c in np.array_split(seeds, a.episodes // 4)]
    t0, done = time.time(), []
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_episodes, jobs):
            done += r
            n, ok, cl = len(done), sum(x[1] for x in done), sum(x[2] for x in done)
            print(f"{time.strftime('%H:%M:%S')} {n}/{a.episodes} episodes, delivered {ok}, kept (clean) {cl}, "
                  f"{(time.time() - t0) / n * a.episodes / 60 - (time.time() - t0) / 60:.0f} min left", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
