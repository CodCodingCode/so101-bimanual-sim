"""Real-rig adapter for the ACTION-CHUNKED clamp student (runs_chunk/rotation/student.pt): builds its 41-number input
from what the rig measures and turns its 8-tick chunks into one tool-point target per tick (ACT temporal ensembling).
No MuJoCo needed except for the forward kinematics the runner already computes.

Trained in sim (chunk_student.py, DAgger from runs_clamp_wide/rotation_servo_done.zip): cube within +-1.5 cm of the
nominal spot, ANY cube rotation, +-1 deg servo spread, weak-grip randomisation. Sim result, student alone:
86 % success / 75 % clean (36 held-out episodes; teacher 83 % / 72 %). NOT yet run on the real arms.

Chunk format (agreed with sim-e5 for real_policy.py): net output (K=8, 2 arms, 3) = tool-point targets relative to the
COMMANDED tool point at prediction time, metres x 20; linear head. Execution: each tick average every chunk covering
the tick, weights exp(-0.1 * age index) oldest first, normalised; clip the step to 0.5 * delta_scale; IK.

    from chunk_student_real import StudentRunner
    run = StudentRunner(path_to_student_pt)
    run.reset(box=(x, y, yaw))                         # vision.detect_box on the overhead frame before moving
    target = run.step(joints_deg, gripper_open, tool_fk, tool_cmd, cube_xyz, cube_seen, cube_yaw, attempt_clock)
        # joints_deg: {"left": 5 angles (deg, sim units incl. joint offsets), "right": ...}
        # gripper_open: {"left": 0..1, "right": 0..1}
        # tool_fk / tool_cmd: (2, 3) metres, world frame (left row 0)   -- forward kinematics / commanded tool points
        # cube_xyz: triangulated cube centre (cube_locator), held while hidden; cube_seen: 1 / 0
        # cube_yaw: radians from vision.detect_cube_pose (any of the 90-deg-equivalent values), held while hidden
        # attempt_clock: ticks since the current grab attempt began / 450
    -> (2, 3) commanded tool-point target for this tick (feed to the runner's IK + safety guard)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CUBE_Y = 0.0372
CUBE_HALF = 0.016
BOX_HOME_Y = -0.077


class ChunkEnsembler:
    """ACT temporal ensembling. Time is in TRAINING ticks (0.04 s) and may advance fractionally: next(adv) with
    adv = real tick / 0.04 s (rt_control.TickClock), sampling each chunk linearly between its steps. adv = 1 every
    call is exactly the original integer-tick ensembler (sim tests)."""

    def __init__(self, K, m=0.1):
        self.K, self.m, self.chunks, self.t = K, m, [], 0.0

    def add(self, chunk_abs):
        self.chunks.append((self.t, np.asarray(chunk_abs, float)))
        self.chunks = [(t0, c) for t0, c in self.chunks if self.t - t0 <= self.K - 1]

    def _at(self, c, a):
        i = int(np.floor(a))
        j = min(i + 1, self.K - 1)
        f = a - i
        return c[i] if f <= 1e-9 or i == j else c[i] * (1 - f) + c[j] * f

    def next(self, adv=1.0):
        preds = [self._at(c, self.t - t0) for t0, c in self.chunks if 0 <= self.t - t0 <= self.K - 1]
        if not preds:                                  # a very long tick ran past every chunk: use the newest end
            preds = [self.chunks[-1][1][-1]]
        w = np.exp(-self.m * np.arange(len(preds)))
        self.t += adv
        return np.tensordot(w / w.sum(), np.array(preds), axes=1)


class StudentRunner:
    def __init__(self, student_pt, delta_scale=0.016, action_scale=0.5):
        import torch
        import torch.nn as nn
        meta = json.loads(Path(str(student_pt).replace("student.pt", "student_meta.json")).read_text())
        self.K, self.scale, self.m, self.obs_dim = meta["K"], meta["scale"], meta["ensemble_m"], meta["obs_dim"]
        self.net = nn.Sequential(nn.Linear(self.obs_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, self.K * 6))
        self.net.load_state_dict(torch.load(student_pt, map_location="cpu"))
        self.net.eval()
        self.torch = torch
        self.max_step = action_scale * delta_scale          # same per-tick limit as training
        self.reset()

    def reset(self, box=(0.0, BOX_HOME_Y, 0.0)):
        self.box = np.asarray(box, float)
        self.ens = ChunkEnsembler(self.K, self.m)
        self.prev_step = np.zeros((2, 3))

    def obs(self, joints_deg, gripper_open, tool_fk, tool_cmd, cube_xyz, cube_seen, cube_yaw, attempt_clock):
        parts = []
        for i, s in enumerate(("left", "right")):
            parts += [np.asarray(joints_deg[s], float)[:5] / 90.0, [gripper_open[s]], np.asarray(tool_fk[i]) * 5.0,
                      np.asarray(tool_cmd[i]) * 5.0]
        c = np.asarray(cube_xyz, float)
        b = self.box
        parts += [[c[0] / 0.05, (c[1] - CUBE_Y) / 0.05, (c[2] - CUBE_HALF) / 0.05, float(cube_seen),
                   np.sin(4 * cube_yaw), np.cos(4 * cube_yaw)],
                  [b[0] / 0.05, (b[1] - BOX_HOME_Y) / 0.05, np.sin(4 * b[2]), np.cos(4 * b[2])],
                  [attempt_clock], self.prev_step.ravel() * self.scale]
        o = np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])
        assert o.shape == (self.obs_dim,), o.shape
        return o

    def step(self, joints_deg, gripper_open, tool_fk, tool_cmd, cube_xyz, cube_seen, cube_yaw, attempt_clock, adv=1.0):
        """adv: training ticks this call stands for (real tick / 0.04 s). The per-tick step limit scales with it, and
        prev_step is stored per training tick, as the student saw it in training."""
        o = self.obs(joints_deg, gripper_open, tool_fk, tool_cmd, cube_xyz, cube_seen, cube_yaw, attempt_clock)
        with self.torch.no_grad():
            chunk = self.net(self.torch.from_numpy(o[None])).numpy()[0].reshape(self.K, 2, 3) / self.scale
        cmd = np.asarray(tool_cmd, float)
        self.ens.add(cmd + chunk)
        lim = self.max_step * adv
        target = cmd + np.clip(self.ens.next(adv) - cmd, -lim, lim)
        self.prev_step = (target - cmd) / adv
        return target
