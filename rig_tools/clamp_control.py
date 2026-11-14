"""Clamp control shared by the REAL runner (real_clamp.py) and its sim test, so the sim exercises exactly the code that
drives the arms. Added 2026-11-11 after a real grab test: the tips stopped ~1 cm short of their commands on each side
(servo P gain 16, no integral: joints settle 1-3 deg short, ~1 cm at 25 cm reach), so the 1.5 mm press gave no force.

  SIM PREVIEW    before anything moves, the whole plan is simulated with the calibrated physics (fitted servos, real
                 joint ranges, the cube where the overhead camera saw it, the box where it will be). If the simulated
                 grab does not lift cleanly, the run is refused. The preview's joint trajectory is the reference.
  REPLAY + JOINT FEEDBACK  the real arms get the preview's joint commands, and while the plan approaches (hover,
                 descend, close: before any contact) each joint's command is nudged until the MEASURED joint matches
                 the preview's measured joint (capped at +-4 deg), then frozen from the squeeze on. The real arm then
                 follows the exact pose sequence that worked in sim (sag included), so the press becomes real force.
                 (Tried first and rejected in sim: a Cartesian x/y correction, which fought the arm's sag and pushed
                 the forearm into the table, and correcting to the pure IK pose, which lifted the tips 7 mm above where
                 the design grips.)
  STIFF SERVOS   arm joints 1-5 run with position gain P = 32 (factory default) instead of 16 for the run, written with
                 the EEPROM lock ON so it lives in RAM only; set back to 16 at the end (and lost on power-off).
The safety guard checks the preview's measured pose (where the arm is expected to be) plus the real measured pose;
the command itself stays inside each servo's real range and the 10 deg lead limit.
"""
from __future__ import annotations

import mujoco
import numpy as np

SIDES = ("left", "right")
REG_P, REG_LOCK = 21, 55
P_STIFF, P_SOFT = 32, 16
FEEDBACK_PHASES = ("hover", "descend", "close")
GAIN = 0.2                       # fraction of the remaining joint error removed per tick while the reference is still
MAX_JOINT_CORR = np.radians(4.0)
STILL_TICKS = 4
STILL_EPS = np.radians(0.05)     # rad per tick: the reference counts as still below this


def set_p_gain(arms, value, joints=(1, 2, 3, 4, 5)):
    """Position gain for the arm joints, RAM only (EEPROM lock on). Returns the values read back."""
    out = {}
    for s, a in arms.items():
        a.bus.sync_write(REG_LOCK, 1, [(i, [1]) for i in joints])
        a.bus.sync_write(REG_P, 1, [(i, [int(value)]) for i in joints])
        out[s] = [a.bus.read(i, REG_P, 1)[0] for i in joints]
    return out


def preview(env, C, plan, cube_xy, box_xy, home_units, bad_contacts):
    """Simulate the plan from the real start pose with the calibrated physics. Returns per tick: ctrl and qpos (rad,
    5 per arm), phase; plus how high the cube was lifted, contact count, and where the cube ended."""
    m, d, sim = env.model, env.data, env.sim
    cx, cy = cube_xy
    for s in SIDES:
        sim.pad_quat[s] = C.clamp_quat(s, (cx, cy))
    cq, bq = m.jnt_qposadr[m.joint("cube_joint").id], m.jnt_qposadr[m.joint("can_joint").id]
    d.qpos[cq:cq + 7] = [cx, cy, C.CUBE_HALF + 0.001, 1, 0, 0, 0]
    d.qpos[bq:bq + 7] = [box_xy[0], box_xy[1], 0.015, 1, 0, 0, 0] if box_xy is not None else [2.5, 2, -1, 1, 0, 0, 0]
    glo = m.actuator_ctrlrange[sim.robot["left"].gripper_actuator][0]
    for i, s in enumerate(SIDES):
        arm = sim.robot[s]
        d.qpos[arm.qpos_ids] = np.deg2rad(home_units[i * 6:i * 6 + 5])
        d.ctrl[arm.actuator_ids] = d.qpos[arm.qpos_ids]
        d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = glo
        d.ctrl[arm.gripper_actuator] = glo
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(40):                               # settle at the start pose
        mujoco.mj_step(m, d)
    for i, s in enumerate(SIDES):
        sim._ee_targets[i] = sim.robot[s].position.copy()
    out = {"ctrl": [], "qpos": [], "phase": []}
    bad, z_hold = 0, 0.0
    while not plan.done:
        out["phase"].append(plan.phase)
        tg = plan.next_targets(sim._ee_targets.copy())
        a = np.zeros((2, 4))
        a[:, :3] = (tg - sim._ee_targets) / sim.delta_scale
        env.step(np.clip(a, -1, 1).ravel())
        out["ctrl"].append({s: d.ctrl[sim.robot[s].actuator_ids].copy() for s in SIDES})
        out["qpos"].append({s: d.qpos[sim.robot[s].qpos_ids].copy() for s in SIDES})
        bad += len(bad_contacts(m, d))
        if plan.phase == "hold":
            z_hold = max(z_hold, d.qpos[cq + 2] - C.CUBE_HALF)
    out.update(lifted_mm=z_hold * 1000, contacts=bad, cube_end=d.qpos[cq:cq + 3].copy())
    return out


class ReplayController:
    def __init__(self, ref, feedback=True):
        self.ref, self.feedback = ref, feedback
        self.joff = {s: np.zeros(5) for s in SIDES}
        self.frozen, self._still = False, 0

    def tick(self, k, meas_rad):
        """k: tick index; meas_rad: {side: 5 measured joint angles, rad}. Returns (cmd, expect, info) in degrees
        (gripper 0 = shut appended)."""
        ref = self.ref
        k = min(k, len(ref["ctrl"]) - 1)
        phase = ref["phase"][k]
        if k > 0:
            moving = max(np.abs(ref["qpos"][k][s] - ref["qpos"][k - 1][s]).max() for s in SIDES)
            self._still = self._still + 1 if moving < STILL_EPS else 0
        if phase not in FEEDBACK_PHASES:
            self.frozen = True
        cmd, expect, err = {}, {}, {}
        for s in SIDES:
            e_q = ref["qpos"][k][s] - meas_rad[s]
            err[s] = e_q
            if self.feedback and not self.frozen and self._still >= STILL_TICKS:
                self.joff[s] = np.clip(self.joff[s] + GAIN * e_q, -MAX_JOINT_CORR, MAX_JOINT_CORR)
            cmd[s] = np.r_[np.rad2deg(ref["ctrl"][k][s] + self.joff[s]), 0.0]
            expect[s] = np.r_[np.rad2deg(ref["qpos"][k][s]), 0.0]
        info = {"phase": phase, "frozen": self.frozen,
                "joint_corr_deg": {s: np.rad2deg(v).round(2).tolist() for s, v in self.joff.items()},
                "joint_err_deg": {s: np.rad2deg(v).round(2).tolist() for s, v in err.items()}}
        return cmd, expect, info
