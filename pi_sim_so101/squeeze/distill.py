#!/usr/bin/env python3
"""Camera student for the two-arm tip clamp into the box (with recovery): scripted teacher -> vision policy via DAgger.

The student sees only what the real rig has:
  - overhead + left wrist + right wrist RGB (64x64), rendered with the CALIBRATED camera rig
    (sim/rig_tools/data/so101_camera_rig.calibrated_full.json: fitted poses, focal lengths, lens distortion)
  - per arm: joint angles (5), gripper opening, tool point from forward kinematics (3), its own commanded tool point (3)
  - its previous action (6), and the grab-attempt clock (1; on the rig: time since the overhead camera last saw the
    cube settle somewhere new, see clamp_rl._update_attempt)
No contact flags, no cube/box state. The re-planning scripted teacher (clamp_rl.expert_action, state-based) labels every
frame -- the same teacher the state DAgger policy (runs_clamp_recovery/dagger) learned from at 95 % success.
Episodes include the recovery disturbances (nudge / slip / push).

Domain randomisation per episode, centred on the real rig's measured colours (env.py: grey-blue arms, maroon cube,
dark box, wood table tint, grey room), light, small camera pose jitter; image augmentation at train time (shift, brightness, contrast,
noise).

    python squeeze/distill.py                      # full run: teacher data, 3 DAgger rounds, eval
    python squeeze/distill.py --eval runs/student/student.pt --episodes 50

Everything lands in squeeze/runs_clamp_recovery/student/.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

os.environ.setdefault("SQUEEZE_TASK", "clamp")
os.environ.setdefault("CLAMP_RECOVERY", "1")
OUT = HERE / "runs_clamp_recovery" / "student"
RIG = HERE.parents[1] / "rig_tools" / "data" / "so101_camera_rig.calibrated_full.json"
TEACHER = "scripted re-planning clamp teacher (clamp_rl.ClampRLEnv.expert_action)"
CAMS = ("overhead", "left_wrist", "right_wrist")
IMG = 64
PROP = 31
STAGES = ("squeezed", "lifted", "placed", "is_success", "clean", "can_tipped")


# ------------------------------------------------------------------ environment with real-camera rendering
def make_env(seed):
    import env as E
    base_sim = E._Sim
    if not getattr(base_sim, "_cam_patched", False):
        class _CamSim(base_sim):
            _cam_patched = True

            def __init__(self, *a, **k):
                k.setdefault("camera_rig", str(RIG))
                k.setdefault("image_size", IMG)
                super().__init__(*a, **k)
        E._Sim = _CamSim
    from clamp_rl import ClampRLEnv
    return ClampRLEnv(seed=seed)


class Randomizer:
    """Per-episode appearance randomisation; stores nominal values once."""

    def __init__(self, env):
        import mujoco
        m = env.base.model
        self.m, self.mj = m, mujoco
        self.rgba0 = m.geom_rgba.copy()
        self.mat0 = m.mat_rgba.copy()
        self.light0 = (m.light_diffuse.copy(), m.light_ambient.copy(), m.light_pos.copy())
        self.head0 = (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())
        self.cam0 = (m.cam_pos.copy(), m.cam_quat.copy())
        self.body0 = (m.body_pos.copy(), m.body_quat.copy())
        names = [m.geom(g).name for g in range(m.ngeom)]
        bodies = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        arm_parts = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1", "camera_mount")
        def col(g):
            return self.rgba0[g] if m.geom_matid[g] < 0 else self.mat0[m.geom_matid[g]]
        import env as E
        self.E = E
        on_arm = [any(bodies[g] == f"{s}_{p}" for s in ("left", "right") for p in arm_parts) for g in range(m.ngeom)]
        # arm plastic: the matched grey-blue (env.ARM_RGB) on arm bodies, plus the light printed camera mounts
        self.arm_light = [g for g in range(m.ngeom) if m.geom_group[g] <= 2
                          and ((on_arm[g] and np.allclose(col(g)[:3], E.ARM_RGB, atol=0.02))
                               or (names[g].startswith("mount_") and col(g)[0] > 0.7 and col(g)[1] > 0.6))]
        self.table = [g for g in range(m.ngeom) if names[g] == "table"]
        self.table_mat = m.geom_matid[self.table[0]]   # wood texture material (or -1 for a flat table)
        self.cube = [g for g in range(m.ngeom) if names[g] == "cube_geom"]
        self.box = [g for g in range(m.ngeom) if names[g] == "can_geom" or names[g].startswith("box_wall_")]
        self.room = [g for g in range(m.ngeom) if names[g] == "floor" or names[g].startswith("room_wall_")]
        self.mount = m.body("overhead_camera_mount").id

    def _set(self, gs, rgb):
        m = self.m
        for g in gs:
            if m.geom_matid[g] >= 0:
                m.geom_matid[g] = -1
            m.geom_rgba[g, :3] = np.clip(rgb, 0, 1)
            m.geom_rgba[g, 3] = 1.0

    def apply(self, rng):
        m = self.m
        E = self.E   # every colour jitters around the real rig's measured colours (env.py, matched_sim)
        self._set(self.arm_light, E.ARM_RGB * rng.uniform(0.85, 1.2) + rng.normal(0, 0.02, 3))
        if self.table_mat >= 0:   # keep the wood grain; jitter its tint / brightness
            m.mat_rgba[self.table_mat, :3] = np.clip(self.mat0[self.table_mat, :3] * rng.uniform(0.75, 1.25)
                                                     + rng.normal(0, 0.04, 3), 0, 1)
        else:
            self._set(self.table, E.TABLE_RGB * rng.uniform(0.75, 1.25) + rng.normal(0, 0.04, 3))
        self._set(self.cube, E.CUBE_RGB * rng.uniform(0.8, 1.2) + rng.normal(0, 0.03, 3))
        self._set(self.box, E.BOX_RGB * rng.uniform(0.75, 1.5) + rng.normal(0, 0.02, 3))
        for g in self.room:
            m.geom_rgba[g, :3] = np.clip(self.rgba0[g, :3] * rng.uniform(0.8, 1.2), 0, 1)
        d0, a0, p0 = self.light0
        m.light_diffuse[:] = d0 * rng.uniform(0.6, 1.3)
        m.light_ambient[:] = a0 * rng.uniform(0.6, 1.4)
        m.light_pos[:] = p0 + rng.normal(0, 0.15, p0.shape)
        m.vis.headlight.diffuse[:] = self.head0[0] * rng.uniform(0.6, 1.3)
        m.vis.headlight.ambient[:] = self.head0[1] * rng.uniform(0.6, 1.4)
        bp, bq = self.body0
        m.body_pos[self.mount] = bp[self.mount] + rng.normal(0, 0.004, 3)
        q = bq[self.mount] + rng.normal(0, 0.006, 4)
        m.body_quat[self.mount] = q / np.linalg.norm(q)
        cp, cq = self.cam0
        m.cam_pos[:] = cp + rng.normal(0, 0.002, cp.shape)
        cq2 = cq + rng.normal(0, 0.006, cq.shape)
        m.cam_quat[:] = cq2 / np.linalg.norm(cq2, axis=1, keepdims=True)


def observe(env, prev_action):
    """(images uint8 [3,IMG,IMG,3], proprio float32 [PROP]) from real-observable quantities only."""
    import mujoco
    sim = env.base.sim
    mujoco.mj_forward(sim.model, sim.data)
    imgs = np.stack([sim.render_camera(c) for c in CAMS])
    parts = []
    for i, side in enumerate(("left", "right")):
        arm = sim.robot[side]
        parts += [np.rad2deg(sim.data.qpos[arm.qpos_ids]) / 90.0, [arm.gripper_opening],
                  arm.position * 5.0, sim._ee_targets[i] * 5.0]
    parts += [prev_action, [(env.base._step - env._attempt_start) / 450.0]]
    return imgs.astype(np.uint8), np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])


# ------------------------------------------------------------------ student network
def build_student():
    import torch
    import torch.nn as nn

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            def enc():
                return nn.Sequential(
                    nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                    nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU())
            self.encs = nn.ModuleList([enc() for _ in CAMS])
            self.prop = nn.Sequential(nn.Linear(PROP, 64), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(128 * len(CAMS) + 64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                      nn.Linear(256, 6), nn.Tanh())

        def forward(self, imgs, prop):
            # imgs: [B, 3cams, 3, H, W] float in [0, 1]
            feats = [e(imgs[:, i]) for i, e in enumerate(self.encs)]
            return self.head(torch.cat(feats + [self.prop(prop)], 1))

    return Student()


def to_tensor(imgs_u8, device):
    import torch
    x = torch.from_numpy(imgs_u8).to(device).float() / 255.0          # [B, 3, H, W, 3]
    return x.permute(0, 1, 4, 2, 3).contiguous()


def augment(x, gen):
    """Shift up to 4 px, brightness / contrast jitter, noise (per sample, per camera)."""
    import torch
    B, C, ch, H, W = x.shape
    pad = torch.nn.functional.pad(x.view(B * C, ch, H, W), (4, 4, 4, 4), mode="replicate")
    ox = torch.randint(0, 9, (B * C,), generator=gen)
    oy = torch.randint(0, 9, (B * C,), generator=gen)
    idx_y = (oy[:, None] + torch.arange(H)[None]).to(x.device)
    idx_x = (ox[:, None] + torch.arange(W)[None]).to(x.device)
    pad = pad[torch.arange(B * C, device=x.device)[:, None, None, None], torch.arange(ch, device=x.device)[None, :, None, None],
              idx_y[:, None, :, None], idx_x[:, None, None, :]]
    x = pad.view(B, C, ch, H, W)
    b = (torch.rand(B, C, 1, 1, 1, generator=gen) * 0.4 - 0.2).to(x.device)
    c = (torch.rand(B, C, 1, 1, 1, generator=gen) * 0.5 + 0.75).to(x.device)
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    x = (x - mean) * c + mean + b
    x = x + torch.randn(x.shape, generator=gen).to(x.device) * 0.02
    return x.clamp(0, 1)


# ------------------------------------------------------------------ data collection (workers)
def _collect(job):
    """Run episodes; execute teacher (prob beta) or student; label every frame with the teacher's action."""
    seeds, student_path, beta, noise, out_file = job
    import torch
    torch.set_num_threads(1)
    student = None
    if student_path:
        student = build_student()
        student.load_state_dict(torch.load(student_path, map_location="cpu"))
        student.eval()
    env = make_env(int(seeds[0]))
    rnd = Randomizer(env)
    I, P, A, results = [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) + 777)
        rnd.apply(rng)
        env.reset(seed=int(seed))
        prev = np.zeros(6, np.float32)
        ever, after_success = set(), 0
        for t in range(env.base.max_steps):
            imgs, prop = observe(env, prev)
            ta = np.asarray(env.expert_action(), np.float32)      # must run every step: advances the plan
            I.append(imgs); P.append(prop); A.append(ta)
            if student is not None and rng.random() >= beta:
                with torch.no_grad():
                    act = student(to_tensor(imgs[None], "cpu"), torch.from_numpy(prop[None])).numpy()[0]
            else:
                act = ta + rng.normal(0, noise, 6).astype(np.float32) if noise > 0 else ta
            act = np.clip(act, -1, 1).astype(np.float32)
            _, _, term, trunc, info = env.step(act)
            prev = act
            for st in STAGES:
                if info.get(st):
                    ever.add(st)
            if info["is_success"] and not info["bad_free_steps"]:
                ever.add("clean")
            if "is_success" in ever:
                after_success += 1
            if term or trunc or after_success > 25:
                break
        ever.discard("clean")       # judged over the whole episode
        if "is_success" in ever and not info["bad_free_steps"]:
            ever.add("clean")
        results.append({st: st in ever for st in STAGES})
    np.savez(out_file, imgs=np.stack(I), prop=np.stack(P), act=np.stack(A))
    return results


def collect(round_name, n_episodes, seed0, student_path, beta, noise, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = np.arange(seed0, seed0 + n_episodes)
    chunks = np.array_split(seeds, workers)
    jobs = [(c, student_path, beta, noise, str(OUT / f"data_{round_name}_{i}.npz")) for i, c in enumerate(chunks) if len(c)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = [r for rs in pool.map(_collect, jobs) for r in rs]
    rates = {st: float(np.mean([r[st] for r in res])) for st in STAGES}
    print(f"[{round_name}] {n_episodes} episodes, beta {beta}, noise {noise}, {time.time() - t0:.0f}s: "
          + " ".join(f"{k}={v:.2f}" for k, v in rates.items()), flush=True)
    return rates


# ------------------------------------------------------------------ training
def load_all():
    I, P, A = [], [], []
    for f in sorted(OUT.glob("data_*.npz")):
        z = np.load(f)
        I.append(z["imgs"]); P.append(z["prop"]); A.append(z["act"])
    return np.concatenate(I), np.concatenate(P), np.concatenate(A)


def train(epochs, init=None):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    I, P, A = load_all()
    n = len(A)
    net = build_student().to(dev)
    if init is not None and Path(init).exists():
        net.load_state_dict(torch.load(init, map_location=dev))
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(0)
    Pt, At = torch.from_numpy(P), torch.from_numpy(A)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen).numpy()
        tot = 0.0
        for k in range(0, n, 256):
            idx = perm[k:k + 256]
            x = augment(to_tensor(I[idx], dev), gen)
            prop = (Pt[idx] + torch.randn(len(idx), PROP, generator=gen) * 0.01).to(dev)
            loss = ((net(x, prop) - At[idx].to(dev)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"  train epoch {ep}: mse {tot / n:.4f} ({n} frames, {time.time() - t0:.0f}s)", flush=True)
    net = net.cpu()
    torch.save(net.state_dict(), OUT / "student.pt")
    return OUT / "student.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval", default=None, help="evaluate a student checkpoint only")
    ap.add_argument("--episodes", type=int, default=50)
    a = ap.parse_args()
    if a.eval:
        collect("eval", a.episodes, 900_000, a.eval, 0.0, 0.0, a.workers)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    log = {"teacher": str(TEACHER), "rig": str(RIG), "rounds": []}
    log["rounds"].append({"round": "teacher", **collect("r0_teacher", 160, 100_000, None, 1.0, 0.15, a.workers)})
    ckpt = train(15)
    for r, beta in enumerate((0.5, 0.2, 0.0), start=1):
        rates = collect(f"r{r}_dagger", 100, 100_000 + 1000 * r, str(ckpt), beta, 0.0, a.workers)
        log["rounds"].append({"round": f"dagger{r}", "beta": beta, **rates})
        ckpt = train(8, init=ckpt)
    final = collect("final_eval", 60, 900_000, str(ckpt), 0.0, 0.0, a.workers)
    log["final_student_eval"] = final
    for f in OUT.glob("data_final_eval_*.npz"):
        f.unlink()
    (OUT / "log.json").write_text(json.dumps(log, indent=1))
    print("final student (cameras + joints only):", " ".join(f"{k}={v:.2f}" for k, v in final.items()), flush=True)


if __name__ == "__main__":
    main()
