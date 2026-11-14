"""Rig adapter for the camera-detector CHUNKED clamp policy v2 (squeeze/chunk_policy_v2.py, retrained on the bb1b28
calibration: runs_clamp_v3/chunk_policy_v2_bb1b28/best.pt; sim, policy alone: 1.00 delivered / 0.94 clean, 32 eps).

It plugs into student_ctl.StudentCtl (the rig controller sim-74's chunked student already runs through: cube sensing,
grab-attempt clock, gripper orientation, IK, rate limit, squeeze limiter, release gap) and differs from sim-74's student
only in what the network sees:

  policy input (34 numbers, squeeze/detector_policy.student_obs), per arm then:
      measured joints / 90 (5) | gripper opening (1) | tool point by FK x 5 (3) | COMMANDED tool point x 5 (3)
      cube from the OVERHEAD detector only: x / 0.05, (y - CUBE_Y) / 0.05, seen (3)
          = the overhead red blob's ray on the plane z = cube half height (cube_locator.locate with one ray), exactly the
            detector modelled in training (a lifted cube reads off by the parallax, as it did there); held while unseen
      previous executed step in action units (6) | attempt clock = ticks since the attempt began / 450 (1)
  policy output: 8 future tool-point targets per arm, relative to the commanded tool point, metres x 20
      (rig_tools/chunking.py format) -> ACT temporal ensembling (chunk_student_real.ChunkEnsembler, which also handles
      the real loop's fractional ticks) -> step capped at 0.5 x delta_scale per training tick, as in training

    runner = ChunkV2Runner(best_pt, delta_scale=sim.delta_scale)
    ctl = ChunkV2Ctl(m, d, sim, C, V, CL, runner, guard, squeeze_limit=True, filters=cfg["perception"])
"""
from __future__ import annotations

import numpy as np

import student_ctl as SC
from chunk_student_real import ChunkEnsembler

CUBE_Y = 0.0372
K, SCALE, M = 8, 20.0, 0.1            # chunking.py: chunk length, output scale, ensembling weight
OBS_DIM = 34
SQUEEZE_TIP_GAP = 0.05                # see ChunkV2Ctl._holding_now
TABLE_X, TABLE_Y = (-0.12, 0.12), (CUBE_Y - 0.17, CUBE_Y + 0.10)   # plausible cube area (as StudentCtl.sense)


def build_net():
    import torch.nn as nn                # = squeeze/chunk_policy_v2.build_net
    return nn.Sequential(nn.Linear(OBS_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, K * 6))


class ChunkV2Runner:
    """Same interface as chunk_student_real.StudentRunner (reset / step / prev_step / max_step), so StudentCtl can
    drive it unchanged. The cube input comes from detector(), which ChunkV2Ctl calls on every fresh camera result."""

    def __init__(self, policy_pt, delta_scale=0.016, action_scale=0.5):
        import torch
        self.torch = torch
        self.net = build_net()
        self.net.load_state_dict(torch.load(policy_pt, map_location="cpu"))
        self.net.eval()
        self.step_m = action_scale * delta_scale      # one training action unit, metres per tick
        self.max_step = self.step_m                   # StudentCtl uses this for the release-gap step
        self.det_xy, self.det_seen = None, 0.0
        self.reset()

    def reset(self, box=None):
        self.box = None if box is None else np.asarray(box, float)   # not a policy input: the box sits at a fixed
        self.ens = ChunkEnsembler(K, M)                               # offset from the cube in training (checked by
        self.prev_step = np.zeros((2, 3))                             # the runner before it starts)

    def detector(self, xy):
        """Overhead detector reading (x, y) on the cube-half-height plane, or None (not seen: hold the last)."""
        if xy is not None:
            self.det_xy, self.det_seen = np.asarray(xy, float)[:2].copy(), 1.0
        else:
            self.det_seen = 0.0

    def obs(self, joints_deg, gripper_open, tool_fk, tool_cmd, attempt_clock):
        if self.det_xy is None:
            raise RuntimeError("the overhead detector has not seen the cube yet")
        parts = []
        for i, s in enumerate(("left", "right")):
            parts += [np.asarray(joints_deg[s], float)[:5] / 90.0, [gripper_open[s]], np.asarray(tool_fk[i]) * 5.0,
                      np.asarray(tool_cmd[i]) * 5.0]
        x, y = self.det_xy
        prev_action = np.clip(self.prev_step / self.step_m, -1.0, 1.0)
        parts += [[x / 0.05, (y - CUBE_Y) / 0.05, self.det_seen], prev_action.ravel(), [attempt_clock]]
        o = np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])
        assert o.shape == (OBS_DIM,), o.shape
        return o

    def step(self, joints_deg, gripper_open, tool_fk, tool_cmd, cube_xyz, cube_seen, cube_yaw, attempt_clock, adv=1.0):
        """StudentRunner.step's signature. cube_xyz / cube_seen / cube_yaw (StudentCtl's triangulation and overhead yaw)
        are NOT inputs of this policy -- it was trained on the overhead detector only; StudentCtl still uses them for the
        attempt clock and the release logic. adv: training ticks this call stands for (real period / 0.04 s)."""
        o = self.obs(joints_deg, gripper_open, tool_fk, tool_cmd, attempt_clock)
        with self.torch.no_grad():
            chunk = self.net(self.torch.from_numpy(o[None])).numpy()[0].reshape(K, 2, 3) / SCALE
        cmd = np.asarray(tool_cmd, float)
        self.ens.add(cmd + chunk)
        lim = self.max_step * adv
        target = cmd + np.clip(self.ens.next(adv) - cmd, -lim, lim)
        self.prev_step = (target - cmd) / adv       # StudentCtl overwrites it with the step actually executed
        return target


def filter_cmd(filt, ctl, prev_cmd, cmd, ee_prev, adv):
    """Run the command filter (cmd_filter.CmdFilter: never send a pose inside the guard's commanded-pose margins; send
    the farthest point toward it that passes) on the controller's command, and keep the controller's own state in step
    with what is actually sent: shadow ctrl (next tick's rate-limit base), commanded tool points, the policy's previous
    step. Shared by the live runner and the sim test. Returns (cmd to send, kept 0..1)."""
    cmd_f, kept = filt(prev_cmd, cmd)
    if kept >= 1.0:
        return cmd, 1.0
    sim, d = ctl.sim, ctl.d
    for s in SC.SIDES:
        d.ctrl[sim.robot[s].actuator_ids] = np.deg2rad(np.asarray(cmd_f[s], float)[:5])
    sim._ee_targets[:] = ee_prev + kept * (sim._ee_targets - ee_prev)
    ctl.run.prev_step = (sim._ee_targets - ee_prev) / adv
    return {s: np.asarray(cmd_f[s], float) for s in SC.SIDES}, kept


class ChunkV2Ctl(SC.StudentCtl):
    """StudentCtl + the overhead-only detector reading this policy was trained on (computed on every fresh camera
    result, from the same red blob the triangulation uses)."""

    def _triangulate(self, frames, dets=None):
        out = super()._triangulate(frames, dets)
        xy = None
        if self._overhead_ray is not None:
            p, _ = self.CL.locate([self._overhead_ray])      # one ray -> its point on the plane z = cube half height
            if p is not None and TABLE_X[0] <= p[0] <= TABLE_X[1] and TABLE_Y[0] <= p[1] <= TABLE_Y[1]:
                xy = p[:2]
        self.run.detector(xy)
        return out

    def _holding_now(self):
        """'The cube is squeezed' for the attempt clock, as training decides it (clamp_rl._update_attempt: the cube only
        counts as lying on the table -- and so as dropped / missed -- when it is NOT squeezed by both arms). The rig
        cannot sense touch; for this design the fingertips (FK) are 4.3-4.6 cm apart while both arms touch the cube and
        5.0-5.8 cm while approaching (sim trace, 2026-11-13). StudentCtl's rule (< 5.5 cm AND tips > 1.5 cm high, tuned
        for sim-74's twist-offset design) flickered off during the on-table squeeze: 8 false "dropped" restarts of the
        attempt clock in one episode, the policy started its grab over each time."""
        tl, tr = self.sim.robot["left"].position, self.sim.robot["right"].position
        mid = 0.5 * (tl + tr)
        return np.linalg.norm(tl[:2] - tr[:2]) < SQUEEZE_TIP_GAP and np.linalg.norm(mid[:2] - self.xyz[:2]) < 0.025

    def start(self, frames, dets=None):
        out = super().start(frames, dets)
        if self.run.det_xy is None:
            raise RuntimeError("the overhead camera does not see the cube (this policy reads the cube from it alone)")
        return out
