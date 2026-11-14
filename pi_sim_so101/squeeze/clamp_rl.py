"""RL version of the two-arm tip clamp into the grey box (no jaws), for train.py (SQUEEZE_TASK=clamp).

Same interface as rl_env.SqueezeRLEnv: `.base` (SqueezeEnv), 6-D Cartesian actions (per arm dx, dy, dz), jaws shut,
`expert_action()` from the scripted ClampPlan, info stages STAGES.

Per episode (~70 % of the area where the scripted clamp lands the cube in the box):
  cube   x +-2.5 cm, y +-2.5 cm around (0, CUBE_Y), yaw +-15 deg, mass 10-20 g (SqueezeEnv)
  box    just past the cube (+BOX_DY), +-0.5 cm / +0..1 cm jitter
  grip friction (jaw CoACD pieces) 0.35-0.60
  gripper orientations set from the cube position (clamp_quat; a 5-DoF arm cannot change it on the fly)
Reward (staged; the in-box rate out-earns every pre-goal term):
  tips to the cube's sides -> both tips touching (squeeze) -> lift -> clear the rim -> over the box -> cube in box
Penalties, every step: arm-arm contact (incl. wrist camera mounts), arm-camera post/head, arm-box, forearm/hand on the
table, cube tilted > 20 deg while held; dropping a lifted cube away from the box.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import clamp_teacher as C          # noqa: E402
import design_squeeze as DS        # noqa: E402
import env as E                    # noqa: E402

E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
C.BOX_DY = C.BOX_DY - 0.005    # 0.5 cm more clearance (box wall vs grippers while squeezing)

ACTION_SCALE = 0.5
CUBE_JITTER = 0.015 if C._D.name == "design_clamp3.json" else 0.025   # v3 is verified clean within +-1.5 cm
YAW_DEG = 15.0
FRICTION = (0.35, 0.60)
ON_BOX = 12.0
SQUEEZE_BONUS, CLEAR_BONUS, PLACE_BONUS, SUCCESS_BONUS = 10.0, 8.0, 10.0, 50.0
BAD_PER_KIND = 2.0          # per step per kind (user: punish anything the arms hit); the teacher is contact-free
TILT_PENALTY = 0.5
DROP_PENALTY = 10.0
TIP_PENALTY = 60.0           # knocking the box over ends the episode; must cost more than giving up
BOX_SHOVE = 1.0             # per step, scaled by how far the box was shoved (beyond 1 cm, full at 3 cm)
RIM = 0.03
BAD_KINDS = ("arm-arm", "arm-camera", "arm-box", "arm-table")
# Recovery training (CLAMP_RECOVERY=1): things go wrong on purpose, longer episodes to retry.
RECOVERY = os.environ.get("CLAMP_RECOVERY") == "1"
CLOCK_STEPS = 450                  # clock = steps since the current grab ATTEMPT began / 450
EARLY_STEPS = 135                  # hover + descend + close: cube moved in this window -> new attempt
QUAT_STEPS = 30                    # re-grab: blend the gripper tilt to the new cube spot over this many steps
MISSED_STEPS = 210                 # by now the cube should be up; still on the table -> grab missed, new attempt
PERTURB = {"none": 0.2, "nudge": 0.3, "slip": 0.3, "push": 0.2}
# GRIP training (2026-11-12: the real arms dropped the cube while braking over the box; the sim grip was far too strong).
# CLAMP_GRIP_DR=1: per episode the fingertip-cube friction 0.4-1.0 (pads AND the cube: MuJoCo takes the larger of the two,
#   so randomising only the pads, as FRICTION did, never changed anything: the cube is 1.0) and the squeeze joints
#   (shoulder pans) 0.25-1x as stiff. Sim with friction 0.5 + 1/4 stiffness reproduces the real drop.
# CLAMP_SLIP=1: penalties while carrying (lifted, not yet over the box):
#   slip risk -- how much of the available friction the grip is using, |tangential| / (mu * normal) per fingertip; the
#     drop is preceded by this saturating at 1 (clean carries stay < ~0.8), so it warns before the cube moves;
#   slip -- the cube moving / rotating relative to the fingertips beyond the normal wobble of a good grip
#     (p95 0.6 mm and 2 deg per step).
GRIP_DR = os.environ.get("CLAMP_GRIP_DR") == "1"
SLIP = os.environ.get("CLAMP_SLIP") == "1"
GRIP_FRICTION = (0.4, 1.0)
SQUEEZE_STIFFNESS = (0.25, 1.0)
SLIP_RISK_FROM, SLIP_RISK_W = 0.7, 1.0      # per step, 0 at 70 % of the friction used, full at 100 %
SLIP_MM, SLIP_DEG, SLIP_MOVE_W = 1.0, 3.0, 0.5
# CLAMP_GUARD=1: the REAL arms' safety guard in the loop (rig_tools/safety.py, same convex hulls, same measured-pose
# margins: arm-arm 3 mm, camera mount 6 mm, camera post 8 mm, box 2 mm). The contact penalty above only counts actual
# touching and is cheap next to the in-box reward (2026-11-12: sim-74's student, through the rig controller, bumped the
# grippers together while releasing over the box in 5/15 episodes; the real guard would have stopped 7/15). Here:
#   inside a margin -> the episode ENDS (as on the rig: the arms fold to rest) with -GUARD_PENALTY
#   within 2x a margin -> -GUARD_NEAR_W per step, scaled, so the policy learns to keep the clearance
GUARD_TRAIN = os.environ.get("CLAMP_GUARD") == "1"
# CLAMP_GUARD_CMD=1 (with CLAMP_GUARD=1): also the guard's COMMANDED-pose check (the joint targets sent to the servos:
# arm-arm 0.5 mm, camera mount 12 mm, camera post 15 mm, box 6 mm), which the rig runs too (2026-11-13: a student tripped
# it at 0.3 mm commanded arm-arm in the rig sim test). Same termination + penalty.
GUARD_CMD = os.environ.get("CLAMP_GUARD_CMD") == "1"
# Table friction: PERMANENT in the base sim (pi_sim scene.py / env.py, user 2026-11-12): the table has contact priority 1
# and a friction of 0.2-0.3 drawn every episode (the real cube slides easily on the real table). Nothing to do here.
GUARD_PENALTY, GUARD_NEAR_W = 60.0, 0.5


class ClampRLEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, episode_seconds: float | None = None, seed: int | None = None):
        super().__init__()
        episode_seconds = episode_seconds or (30.0 if RECOVERY else 18.0)
        self.base = E.SqueezeEnv(episode_seconds=episode_seconds, seed=seed)
        self.sim, self.model, self.data = self.base.sim, self.base.model, self.base.data
        m = self.model
        self._rng = np.random.default_rng(seed)
        self._pads = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                      and "coacd" in m.mesh(m.geom_dataid[g]).name]
        self._cq = m.jnt_qposadr[m.joint("cube_joint").id]
        self._bq = m.jnt_qposadr[m.joint("can_joint").id]
        self._cube_geom, self._cube_body = m.geom("cube_geom").id, m.body("cube").id
        self._grip_body = m.body("left_gripper").id
        self._side_of = {g: s for g in range(m.ngeom) for s in C.SIDES if m.body(m.geom_bodyid[g]).name.startswith(s)}
        self._pans = [m.actuator(f"{s}_shoulder_pan").id for s in C.SIDES]
        self._pan_kp, self._pan_bias = m.actuator_gainprm[self._pans, 0].copy(), m.actuator_biasprm[self._pans, 1].copy()
        self._cube_mu = float(m.geom_friction[self._cube_geom, 0])
        self._f6 = np.zeros(6)
        self._guard = None
        if GUARD_TRAIN:
            sys.path.insert(0, str(HERE.parents[1] / "rig_tools"))
            import safety as SF
            self._SF = SF
            w = np.full(5, 400.0)                       # joint ranges are already the real ones (env.apply_real_ranges)
            self._guard = SF.Guard(m, {s: (-w, w) for s in SF.SIDES}, box_xy=None, log=lambda *a: None)
            self._gq = {s: [m.jnt_qposadr[m.joint(f"{s}_{j}").id] for j in
                            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")] for s in C.SIDES}
            self._gg = {s: (m.jnt_qposadr[m.joint(f"{s}_gripper").id], *m.actuator_ctrlrange[m.actuator(f"{s}_gripper").id])
                        for s in C.SIDES}
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.reset(seed=seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, self._observe().shape, np.float32)

    # ------------------------------------------------------------ helpers
    def _cube_bottom(self):
        return self.base.cube_pos()[2] - C.CUBE_HALF

    def _tip_targets(self):
        cp = self.base.cube_pos()
        z = cp[2] - C.CUBE_HALF + C.TIP_Z
        return {"left": np.array([cp[0] - C.CUBE_HALF - 0.003, cp[1] + C.TIP_DY["left"], z]),
                "right": np.array([cp[0] + C.CUBE_HALF + 0.003, cp[1] + C.TIP_DY["right"], z])}

    def _tilt(self):
        R = self.data.xmat[self.model.body("cube").id].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    def _observe(self):
        b, tt = self.base, self._tip_targets()
        extra = [self.sim._ee_targets.ravel()]
        for i, s in enumerate(C.SIDES):
            extra += [tt[s] - b.tcp(s), tt[s] - self.sim._ee_targets[i]]
        extra += [b.can_pos()[:2] - b.cube_pos()[:2], [self._cube_bottom() - b.lid_z()], [self._tilt() / 45.0],
                  np.array([k in self._bad_now for k in BAD_KINDS], float),
                  [(self.base._step - self._attempt_start) / CLOCK_STEPS]]   # attempt clock: the teacher's schedule is timed
        return np.concatenate([b._observe(), *[np.asarray(x, np.float32).ravel() for x in extra]]).astype(np.float32)

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        cx, cy = rng.uniform(-CUBE_JITTER, CUBE_JITTER), C.CUBE_Y + rng.uniform(-CUBE_JITTER, CUBE_JITTER)
        bx, by = cx + rng.uniform(-0.005, 0.005), cy + C.BOX_DY - rng.uniform(0.0, 0.01)
        for s in C.SIDES:
            self.sim.pad_quat[s] = C.clamp_quat(s, (cx, cy))
        self.base.reset(seed=int(rng.integers(2 ** 31)))
        m, d = self.model, self.data
        m.geom_friction[self._pads, 0] = rng.uniform(*FRICTION)
        m.geom_friction[self._cube_geom, 0] = self._cube_mu
        m.actuator_gainprm[self._pans, 0], m.actuator_biasprm[self._pans, 1] = self._pan_kp, self._pan_bias
        self.grip = (float(max(m.geom_friction[self._pads[0], 0], self._cube_mu)), 1.0)
        if GRIP_DR:
            mu, k = rng.uniform(*GRIP_FRICTION), rng.uniform(*SQUEEZE_STIFFNESS)
            m.geom_friction[self._pads, 0] = mu
            m.geom_friction[self._cube_geom, 0] = mu
            m.actuator_gainprm[self._pans, 0], m.actuator_biasprm[self._pans, 1] = self._pan_kp * k, self._pan_bias * k
            self.grip = (float(mu), float(k))
        self.table_mu = float(m.geom_friction[m.geom("table").id, 0])     # drawn by the base sim's reset (0.2-0.3)
        self._slip_ref, self.slip_risk_sum, self.slip_sum = None, 0.0, 0.0
        self._guard_box, self.guard_stop, self.guard_near_sum = None, None, 0.0
        yaw = np.radians(rng.uniform(-YAW_DEG, YAW_DEG))
        d.qpos[self._cq:self._cq + 7] = [cx, cy, C.CUBE_HALF + 0.001, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[self._bq:self._bq + 7] = [bx, by, E.CAN_HEIGHT / 2, 1, 0, 0, 0]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        for _ in range(20):
            mujoco.mj_step(m, d)
        self.plan = C.ClampPlan((cx, cy, 0.0), (bx, by))
        self._attempt_cube, self._box_xy, self.replans, self._need_replan = np.array([cx, cy]), (bx, by), 0, False
        self._attempt_start = 0
        self._quat_t = QUAT_STEPS
        self._ever = {k: False for k in ("squeezed", "lifted", "cleared", "placed", "success", "dropped")}
        self._box_steps = 0
        self._box_start = self.base.can_pos()[:2].copy()
        self._bad_now = set()
        self.bad_counts = {k: 0 for k in BAD_KINDS}
        self.bad_free = 0
        self._bonus = {k: False for k in ("squeeze", "clear", "place", "success")}
        self._shape_reach, self._holding, self._lift_steps, self._grace = True, False, 0, 0
        self._pert = str(rng.choice(list(PERTURB), p=list(PERTURB.values()))) if RECOVERY else "none"
        self._pert_at = int(rng.integers(20, 110)) if self._pert == "nudge" else int(rng.integers(5, 60))
        self._pert_left, self._pert_body, self._pert_force, self.perturbed = 0, None, None, False
        return self._observe(), {}

    def _perturb(self):
        """Apply this episode's disturbance (training only)."""
        m, d, rng = self.model, self.data, self._rng
        if self._pert_left > 0:
            self._pert_left -= 1
            if self._pert_left == 0:
                d.xfrc_applied[:] = 0
            return
        if self.perturbed or self._pert == "none":
            return
        if self._pert == "nudge" and self.base._step == self._pert_at and not self.base.squeezed():
            # move the cube to another spot the clamp can still do cleanly (cube-box layout map): 0-3 cm left /
            # 0.8 cm right of the box, 9.7-11.8 cm in front of it, inside the normal cube area, >= 1.2 cm away
            p, box = d.qpos[self._cq:self._cq + 3].copy(), self.base.can_pos()[:2]
            for _ in range(30):
                if C._D.name == "design_clamp3.json":   # v3: same cube-box layout as the design (box BOX_DY+0.5..1.5 cm away)
                    q = box + [rng.uniform(-0.005, 0.005), rng.uniform(-C.BOX_DY, -C.BOX_DY + 0.01)]
                else:
                    q = box + [rng.uniform(-0.025, 0.008), rng.uniform(0.097, 0.118)]
                if abs(q[0]) <= CUBE_JITTER and abs(q[1] - C.CUBE_Y) <= CUBE_JITTER and np.linalg.norm(q - p[:2]) >= 0.012:
                    break
            else:
                self.perturbed = True          # nowhere feasible to move it: no nudge this episode
                return
            p[:2] = q
            yaw = np.radians(rng.uniform(-20, 20))
            d.qpos[self._cq:self._cq + 7] = [p[0], p[1], C.CUBE_HALF + 0.001, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            d.qvel[m.jnt_dofadr[m.joint("cube_joint").id]:m.jnt_dofadr[m.joint("cube_joint").id] + 6] = 0
            mujoco.mj_forward(m, d)
            self.perturbed = True
        elif self._pert == "slip" and self._lift_steps == self._pert_at:
            self._pert_body = m.body("cube").id
            ang = rng.uniform(0, 2 * np.pi)
            self._pert_force = np.array([0.3 * np.cos(ang), 0.3 * np.sin(ang), -4.0])     # grip slip: mostly straight down
            d.xfrc_applied[self._pert_body, :3] = self._pert_force
            self._pert_left, self.perturbed, self._grace = 4, True, 40
        elif self._pert == "push" and self._lift_steps == self._pert_at:
            self._pert_body = m.body(f"{rng.choice(C.SIDES)}_gripper").id
            ang = rng.uniform(0, 2 * np.pi)
            d.xfrc_applied[self._pert_body, :3] = [6.0 * np.cos(ang), 6.0 * np.sin(ang), 0.0]
            self._pert_left, self.perturbed, self._grace = 5, True, 40

    def friction_use(self):
        """Largest share of the available friction either fingertip uses on the cube: |tangential| / (mu * normal),
        summed over that fingertip's contacts. 1 = sliding."""
        m, d = self.model, self.data
        N, Tg, mu = {s: 0.0 for s in C.SIDES}, {s: np.zeros(3) for s in C.SIDES}, {s: 1.0 for s in C.SIDES}
        for i in range(d.ncon):
            c = d.contact[i]
            o = c.geom2 if c.geom1 == self._cube_geom else c.geom1 if c.geom2 == self._cube_geom else None
            if o is None or o not in self._side_of:
                continue
            s = self._side_of[o]
            mujoco.mj_contactForce(m, d, i, self._f6)
            fr = c.frame.reshape(3, 3)
            N[s] += self._f6[0]
            Tg[s] += self._f6[1] * fr[1] + self._f6[2] * fr[2]
            mu[s] = c.friction[0]
        return max(float(np.linalg.norm(Tg[s]) / (mu[s] * N[s])) if N[s] > 1e-6 else 1.0 for s in C.SIDES)

    def _cube_in_grip(self):
        """Cube position / orientation in the fingertip frame (midpoint of the tool points, left gripper axes)."""
        b, d = self.base, self.data
        Rg = d.xmat[self._grip_body].reshape(3, 3)
        mid = 0.5 * (b.tcp("left") + b.tcp("right"))
        return Rg.T @ (b.cube_pos() - mid), Rg.T @ d.xmat[self._cube_body].reshape(3, 3)

    def _slip_penalty(self, carrying):
        if not carrying:
            self._slip_ref = None
            return 0.0
        risk = float(np.clip((self.friction_use() - SLIP_RISK_FROM) / (1.0 - SLIP_RISK_FROM), 0.0, 1.0))
        p, R = self._cube_in_grip()
        move = 0.0
        if self._slip_ref is not None:
            mm = 1000 * np.linalg.norm(p - self._slip_ref[0])
            deg = np.degrees(np.arccos(np.clip((np.trace(self._slip_ref[1].T @ R) - 1) / 2, -1, 1)))
            move = min(max(0.0, mm - SLIP_MM) / SLIP_MM + max(0.0, deg - SLIP_DEG) / SLIP_DEG, 1.0)
        self._slip_ref = (p, R)
        self.slip_risk_sum += risk
        self.slip_sum += move
        return SLIP_RISK_W * risk + SLIP_MOVE_W * move

    def _guard_check(self):
        """(penalty, stop kinds) from the real guard's distances in the current (measured) pose."""
        g, d = self._guard, self.data
        if self._guard_box is None:                   # the rig detects the box once, before moving
            det = getattr(self, "box_det", None)      # envs that DETECT the box (clamp_wide): the guard uses the
            if det is not None:                       # detection (x, y, yaw) as on the rig, not the true box
                self._guard_box = (float(det[0]), float(det[1]), float(det[2]))
            else:
                self._guard_box = (*self.base.can_pos()[:2], 0.0)
            g.set_box(self._guard_box[:2], self._guard_box[2])
        q = {}
        for s in C.SIDES:
            adr, lo, hi = self._gg[s]
            q[s] = np.r_[np.rad2deg(d.qpos[self._gq[s]]), 100.0 * (d.qpos[adr] - lo) / (hi - lo)]
        checks = [("meas", q)]
        if GUARD_CMD:                                  # the commanded pose: the IK joint targets (as sent to the servos)
            qc = {}
            for s in C.SIDES:
                arm = self.sim.robot[s]
                lo, hi = self.model.actuator_ctrlrange[arm.gripper_actuator]
                qc[s] = np.r_[np.rad2deg(d.ctrl[arm.actuator_ids]), 100.0 * (d.ctrl[arm.gripper_actuator] - lo) / (hi - lo)]
            checks.append(("cmd", qc))
        stop, near = [], 0.0
        for which, pose in checks:
            dist = g.distances(pose)
            for kind, margin in self._SF.MARGIN[which].items():
                if kind in dist:
                    dd = dist[kind][0]
                    if dd < margin:
                        stop.append(f"{kind} {1000 * dd:.1f} mm ({which})")
                    near = max(near, float(np.clip((2 * margin - dd) / margin, 0.0, 1.0)))
        return GUARD_NEAR_W * near, stop

    def _update_attempt(self, squeezed, in_box):
        """Start a new grab attempt (and a fresh teacher plan) when the cube sits on the table somewhere this attempt
        cannot finish: moved before the grab, dropped after a slip, or a grab that missed. Uses only the cube pose and
        contacts, so a deployed policy can compute the same attempt clock."""
        b = self.base
        cube = b.cube_pos()
        on_table = self._cube_bottom() < 0.004 and not squeezed
        t = b._step - self._attempt_start
        moved = t < EARLY_STEPS and np.linalg.norm(cube[:2] - self._attempt_cube) > 0.012
        missed = t > MISSED_STEPS and not self._holding
        near_box = np.linalg.norm(b.can_pos()[:2] - cube[:2]) < 0.05
        if on_table and not in_box and not near_box and (moved or missed or self._need_replan):
            self._need_replan = False
            self.plan = C.ClampPlan((cube[0], cube[1], 0.0), self._box_xy)
            # first back the arms apart and up (they may still be together where the cube slipped), same total time
            ee = self.sim._ee_targets.copy()
            mx = 0.5 * (ee[0, 0] + ee[1, 0])
            up = max(ee[:, 2].max() + 0.03, C.CARRY_Z)
            _, hl, hr, k = self.plan.waypoints[0]
            self.plan.waypoints[:1] = [("backoff", np.array([mx - 0.07, ee[0, 1], up]), np.array([mx + 0.07, ee[1, 1], up]), 20),
                                       ("hover", hl, hr, k - 20)]
            self._quat_from = {sd: self.sim.pad_quat[sd].copy() for sd in C.SIDES}
            self._quat_to = {sd: C.clamp_quat(sd, cube[:2]) for sd in C.SIDES}
            self._quat_t = 0
            self._attempt_cube, self._attempt_start = cube[:2].copy(), b._step
            self._shape_reach = True
            self.replans += 1

    def expert_action(self):
        """Scripted teacher: follows the current attempt's ClampPlan (a fresh plan per attempt, see _update_attempt)."""
        tg = self.plan.next_targets(self.sim._ee_targets.copy())
        a = (tg - self.sim._ee_targets) / (self.sim.delta_scale * ACTION_SCALE)
        return np.clip(a.ravel(), -1, 1).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1, 1).reshape(2, 3)
        full = np.zeros((2, 4))
        full[:, :3] = a * ACTION_SCALE
        if RECOVERY:
            self._perturb()
        if self._quat_t < QUAT_STEPS:
            self._quat_t += 1
            for sd in C.SIDES:
                q0, q1, w = self._quat_from[sd], self._quat_to[sd], self._quat_t / QUAT_STEPS
                q = (1 - w) * q0 + w * (q1 if q0 @ q1 >= 0 else -q1)          # nlerp (small angle)
                self.sim.pad_quat[sd] = q / np.linalg.norm(q)
        _, _, term, trunc, info = self.base.step(full.ravel())
        b, ev = self.base, self._ever
        bad = DS.bad_contacts(self.model, self.data)
        self._bad_now = set(bad)
        for k in self._bad_now:
            self.bad_counts[k] += 1
        self.bad_free += bool(self._bad_now) and self._grace == 0
        squeezed = b.squeezed()
        bottom = self._cube_bottom()
        lifted = squeezed and bottom > 0.01
        cleared = lifted and bottom > RIM + 0.005
        box_dist = float(np.linalg.norm(b.can_pos()[:2] - b.cube_pos()[:2]))
        in_box = b.on_can()
        self._box_steps = self._box_steps + 1 if in_box else 0
        self._lift_steps = self._lift_steps + 1 if lifted else self._lift_steps
        self._grace = max(0, self._grace - 1)
        r = 0.0
        if self._shape_reach:
            tt = self._tip_targets()
            dsum = sum(np.linalg.norm(tt[s] - b.tcp(s)) for s in C.SIDES) / 2
            r += 1.0 - np.tanh(10 * dsum)
        if squeezed:
            r += 0.5
            if not self._bonus["squeeze"]:
                r += SQUEEZE_BONUS
                self._bonus["squeeze"] = True
            ev["squeezed"] = True
            self._shape_reach = False
        if lifted:
            r += 2.0 * min(bottom / 0.06, 1.0)
            ev["lifted"] = True
            self._holding = True
        if cleared:
            r += 1.5 * (1.0 - np.tanh(10 * box_dist))
            if not self._bonus["clear"]:
                r += CLEAR_BONUS
                self._bonus["clear"] = True
            ev["cleared"] = True
        if in_box:
            r += ON_BOX
            if not self._bonus["place"]:
                r += PLACE_BONUS
                self._bonus["place"] = True
            ev["placed"] = True
            if self._box_steps >= b.settle_steps and not ev["success"]:
                r += SUCCESS_BONUS
                ev["success"] = True
            self._holding = False
        r -= BAD_PER_KIND * len(self._bad_now)
        shove = float(np.linalg.norm(b.can_pos()[:2] - self._box_start))
        if shove > 0.01:
            r -= BOX_SHOVE * min((shove - 0.01) / 0.02, 1.0)
        if info.get("can_tipped"):
            r -= TIP_PENALTY
        if lifted and self._tilt() > 20:
            r -= TILT_PENALTY
        if SLIP and self._grace == 0:          # not while a training disturbance shoves the cube / a gripper
            r -= self._slip_penalty(lifted and box_dist > 0.04)
        guard_end = False
        if self._guard is not None and self._grace == 0:
            near, stop = self._guard_check()
            r -= near
            self.guard_near_sum += near
            if stop:
                r -= GUARD_PENALTY
                self.guard_stop, guard_end = stop, True
        if self._holding and not squeezed and not in_box and bottom < 0.005 and box_dist > 0.04:
            if self._grace == 0:                     # a drop caused by the training disturbance is not the policy's fault
                r -= DROP_PENALTY
            ev["dropped"] = True
            self._holding, self._shape_reach = False, True       # guide it back to the cube to re-clamp
            self._need_replan = True                             # the teacher starts a fresh clamp from here
        self._update_attempt(squeezed, in_box)
        info = {"is_success": ev["success"], "squeezed": ev["squeezed"], "lifted": ev["lifted"], "cleared": ev["cleared"],
                "placed": ev["placed"], "can_tipped": bool(info.get("can_tipped")), "box_shove_mm": round(1000 * shove, 1),
                "bad_contact_steps": int(sum(self.bad_counts.values())), "bad_free_steps": int(self.bad_free),
                "perturbation": self._pert, "perturbed": self.perturbed, "attempts": 1 + self.replans,
                "recovered": bool(ev["success"] and self.perturbed),
                "grip_mu": self.grip[0], "grip_stiffness": self.grip[1], "table_mu": self.table_mu,
                "slip_risk": round(self.slip_risk_sum, 2), "slip": round(self.slip_sum, 2),
                "guard_stop": self.guard_stop, "guard_near": round(self.guard_near_sum, 2),
                **{f"bad_{k.replace('-', '_')}": self.bad_counts[k] for k in BAD_KINDS}}
        return self._observe(), float(r), bool(term or guard_end), bool(trunc), info

    def render(self):
        return self.base.render()
