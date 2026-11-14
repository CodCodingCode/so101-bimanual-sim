#!/usr/bin/env python3
"""Wide-area clamp: cube anywhere within +-12 cm (was +-1.5 cm), box wherever the overhead camera finds it.

User, 2026-11-12: "train to move the cube from +-12 cm instead of +-1.5 cm; use the IK script as a start, but use
classical ML to detect where the box is".

  * Base motion = the scripted IK clamp (clamp_teacher.ClampPlan, design v3), planned from the cube position and the
    box position the CLASSICAL detector reads off the rendered overhead frame (vision.detect_box: dark-pixel blobs +
    a fitted 10 x 10 x 3 cm box silhouette through the calibrated camera; no learning). Detected once at reset, as
    the real runner would before moving.
  * Learned part = a RESIDUAL in tool-point space: per arm an offset (dx, dy, dz) added to the plan's targets. The
    action is the offset target (x OFFSET_MAX); the offset follows it at <= OFFSET_RATE per step. Off-centre the open-loop
    plan misses the cube by ~1 cm because the 5-DoF arms cannot hold the requested gripper angle exactly (measured:
    tip error 10-20 mm, direction changes with position), which a correction policy can learn from the tip / target
    errors in its observation.
  * Curriculum (SCHEDULE): A) cube area fixed at +-1.5 cm, cube rotation 15 -> 45 deg (the policy learns to adjust the
    pinch to the cube's angle); B) any rotation, cube area +-1.5 -> +-12 cm and box spot fixed -> anywhere in its
    region. A stage is passed when the last EVAL_WINDOW episodes succeed >= ADVANCE_A / ADVANCE_B.
  * Reward, contacts, stages: clamp_rl.ClampRLEnv (unchanged).

    python squeeze/clamp_wide.py check --level 0 --yaw 45 --n 12    # plan only (zero residual)
    python squeeze/clamp_wide.py train --steps 12000000              # residual PPO + curriculum -> squeeze/runs_clamp_wide/
    python squeeze/clamp_wide.py eval runs_clamp_wide/rotation_done.zip --level 0 --yaw 45 --n 48
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
# twist-tolerant clamp (sim-e5): plan alone 12/12 at yaw 30/40/45 deg vs 6/12, 10/12, 12/12 for v3 (2026-11-12)
os.environ.setdefault("CLAMP_DESIGN", "design_clamp3c_twistoff.json")

RUNS = HERE / "runs_clamp_wide"
# Camera rig (sim-e5, 2026-11-13): SO101_RIG overrides; else the validated real calibration once it is adopted
# (rig_tools/data/so101_camera_rig.active.json, written only after the cube check passes); else calibrated_full.
# Same rule as distill.py / detector_policy.py, so envs, students, the cube sensor and sim-e5's harness switch together.
_RIG_DIR = HERE.parents[1] / "rig_tools" / "data"
RIG = Path(os.environ["SO101_RIG"]) if os.environ.get("SO101_RIG") else (
    _RIG_DIR / "so101_camera_rig.active.json" if (_RIG_DIR / "so101_camera_rig.active.json").exists()
    else _RIG_DIR / "so101_camera_rig.calibrated_full.json")
CUBE_Y = 0.0372
R0, R1 = 0.015, 0.12                 # cube half-range (m): curriculum start / goal
BOX_X, BOX_Y, BOX_YAW = 0.07, (-0.10, -0.06), 15.0   # box region at level 1 (centre x +-7 cm, y, yaw +-deg)
BOX_HOME = (0.0, -0.077)             # level-0 box spot (where the real place-in-box run worked)
SEPARATION = 0.075                   # cube centre outside the box footprint + margin (max-norm, m)
OFFSET_RATE, OFFSET_MAX = 0.002, 0.04
EVAL_WINDOW = 120
ADVANCE_A = float(os.environ.get("WIDE_ADVANCE_A", "0.85"))   # success needed to leave a rotation (A) stage
ADVANCE_B = float(os.environ.get("WIDE_ADVANCE_B", "0.75"))   # ... a distance (B) stage
# (area level 0..1, cube yaw range +-deg): A = rotation in the +-1.5 cm area first (user), then B = distance
# third entry = servo bias (deg, per joint, +-): real servos stop 1-3 deg short. Measured 2026-11-12, plan alone at
# +-1.5 cm, any yaw: 96 % success with ideal servos, 23 % with +-3 deg -> A2 is where the policy has to learn to adjust.
SCHEDULE = ([(0.0, 15.0, 0.0), (0.0, 25.0, 0.0), (0.0, 35.0, 0.0), (0.0, 45.0, 0.0)]          # A: rotation, ideal servos
            + [(0.0, 45.0, 1.0)]                                                               # A2: rotation + servo spread
            + [(round(0.1 * i, 1), 45.0, 1.0) for i in range(1, 11)])                          # B: distance
# servo spread +-1 deg (not 3): the sim sag already matches the real arms to ~0.5 deg (sim-e5, 2026-11-12), so the bias
# is only the joint-to-joint / day-to-day spread on top, not the whole 1-3 deg shortfall.
# Real servos stop 1-3 deg short (sim-e5, real rig 2026-11-12): per episode a constant per-joint bias of up to this many
# degrees through the position actuators' bias term (force = kp * (ctrl - bias - q)). 0 = off.
SERVO_BIAS_DEG = float(os.environ.get("WIDE_SERVO_BIAS", "0"))
# The real runner's joint tracking correction (see _CamSim). OFF by default since 2026-11-12 ~18:45: on the real arms it
# never engaged during policy runs (joint_corr_deg 0 on every tick: a policy never holds a target still), sim-e5 made it
# opt-in (real_policy.py --correct), and the sim's sag matches the real arms within ~0.5 deg (lift -1.2 real vs -1.1..
# -1.3 sim, elbow -1.8 vs -2.3..-2.5). WIDE_TRACKING=1 turns it on (then GRASP_DZ applies).
TRACKING = os.environ.get("WIDE_TRACKING", "0") == "1"
PROFILE = HERE.parents[1] / "so101_actuation" / "so101_profile.json"
NEAR_CUBE = 0.03
GRASP_DZ = float(os.environ.get("WIDE_GRASP_DZ_MM", "-3")) / 1000      # grip height change with tracking on
PLACE_PHASES, PLACE_GAIN, PLACE_MAX = ("carry", "lower"), 0.15, 0.04   # closed-loop drop over the detected box
PLACE_FB = os.environ.get("WIDE_PLACE_FB", "0") == "1"
RIG_RULES = os.environ.get("WIDE_RIG_RULES", "1") == "1"
GUARD_SCALE = float(os.environ.get("WIDE_GUARD_SCALE", "1.0"))   # training-only multiplier on the guard's measured margins
# box margin separately: on the rig the guard's box is the DETECTED one (0.5-2.6 mm off, no yaw), so it needs more buffer
GUARD_SCALE_BOX = float(os.environ.get("WIDE_GUARD_SCALE_BOX", str(GUARD_SCALE)))
GUARD_SCALE_CMD = float(os.environ.get("WIDE_GUARD_SCALE_CMD", "1.0"))   # same for the COMMANDED-pose margins (CLAMP_GUARD_CMD)
# the rig controller's SQUEEZE LIMITER (rig_tools/student_ctl.py): commanded grippers (IK of the target, jaws shut) never
# closer than SQUEEZE_MIN_GAP: first the inward x step is dropped, then everything but the vertical
RIG_LIMITER = os.environ.get("WIDE_RIG_LIMITER", "1") == "1"
SQUEEZE_MIN_GAP = 0.001
# Cube-on-table friction (user, 2026-11-12: the real cube slides easily). Since ~23:10 the BASE sim does it for every
# env (pi_sim scene: table priority 1; env reset: table friction U(0.2, 0.3) per episode, user's instruction), so by
# default this env leaves the table alone. WIDE_TABLE_MU="lo,hi" overrides the range (experiments only).
_TMU = os.environ.get("WIDE_TABLE_MU", "")
TABLE_MU = tuple(float(v) for v in _TMU.split(",")) if _TMU else None   # the real controller's preload cap over the box      # off: it pushed the arms into the box (clean 85 -> 48 %)
sys.path.insert(0, str(HERE.parents[1] / "so101_actuation"))
WORKERS = int(os.environ.get("WIDE_WORKERS", "5"))  # user's Mac: <= 4-6 sim workers (another session also trains)


def make_env_cls():
    import gymnasium as gym
    import mujoco
    from gymnasium import spaces

    import clamp_rl as CR
    import clamp_teacher as C
    import env as E
    import vision as V

    if GUARD_SCALE != 1.0 or GUARD_SCALE_BOX != 1.0 or GUARD_SCALE_CMD != 1.0:
        # TRAIN with wider guard margins than the rig enforces (this process only; the rig's safety.py is untouched):
        # the guard-trained student still tripped the real 3.0 mm arm-arm margin at 2.7-2.9 mm in 4 of 60 rig-harness
        # episodes, so it learns to keep GUARD_SCALE x the margin (and the near-margin penalty up to 2 x that)
        import safety as _SF
        if not getattr(_SF, "_wide_scaled", False):
            for k in _SF.MARGIN["meas"]:
                _SF.MARGIN["meas"][k] *= GUARD_SCALE_BOX if k == "arm-box" else GUARD_SCALE
            for k in _SF.MARGIN["cmd"]:
                _SF.MARGIN["cmd"][k] *= GUARD_SCALE_CMD
            _SF._wide_scaled = True

    if TRACKING and not getattr(C, "_wide_grasp_dz", False):
        # The tracking correction removes the arm's sag, which lifts the grip ~ to where the design (tuned on a sagging
        # arm) would slip: plan alone, tracking on, +-3 deg servo error, any yaw: 54 % success; tips -2 / -3 / -4 / -7
        # mm -> 90 / 94 / 96 / 94 % (clean 79 / 75 / 62 / 0 %: deeper = arm-table). Every plan (also re-grab plans
        # made by ClampRLEnv) reads TIP_Z / DROP_Z from clamp_teacher, so shift them there.
        C.TIP_Z += GRASP_DZ
        C.DROP_Z += GRASP_DZ
        C._wide_grasp_dz = True

    class _CamSim(E._Sim):
        """The squeeze sim with the calibrated cameras and the REAL runner's joint tracking correction
        (so101_actuation/tracking.TrackingCorrector, as rig_tools/real_policy.py uses it): per joint, up to +-4 deg,
        integrated while the IK target is still and both tips are > 3 cm from the cube, frozen from then on. It is
        added to the IK joint targets (after IK, as on the rig), so it survives the sim's rate limiter."""

        def __init__(self, *a, **k):
            k.setdefault("camera_rig", str(RIG))
            self.tracker, self._corr = None, None
            super().__init__(*a, **k)
            if TRACKING:
                import tracking as TR
                self.tracker = TR.TrackingCorrector(TR.load_profile(PROFILE))
                self._corr = {s: np.zeros(5) for s in C.SIDES}

        def reset_tracking(self):
            if self.tracker is not None:
                self.tracker.reset()
                self._corr = {s: np.zeros(5) for s in C.SIDES}

        def _solve_ik(self, arm, position, quaternion, downward=None):
            target, err = super()._solve_ik(arm, position, quaternion, downward)
            if self._corr is not None:
                target = np.array(target, float).copy()
                target[:5] += np.radians(self._corr[arm.name])
            return target, err

        def _track(self):
            d = self.data
            cube = d.body("cube").xpos
            near = min(np.linalg.norm(self.robot[s].position[:2] - cube[:2]) for s in C.SIDES) < NEAR_CUBE
            if near:
                self.tracker.freeze()
            tgt = {s: np.rad2deg(self._joint_targets[i][:5]) - self._corr[s] for i, s in enumerate(C.SIDES)}
            meas = {s: np.rad2deg(d.qpos[self.robot[s].qpos_ids][:5]) for s in C.SIDES}
            self.tracker.step(tgt, meas, active=not near)
            self._corr = {s: self.tracker.corr[s].copy() for s in C.SIDES}

        def step(self, action):
            if self.tracker is not None and self._ready and not self._done:
                self._track()
            return super().step(action)

    class ClampWideEnv(CR.ClampRLEnv):
        def __init__(self, seed=None, level=0.0, yaw_max=None, episode_seconds=None):
            self.level = float(level)
            self.yaw_max = float(CR.YAW_DEG if yaw_max is None else yaw_max)   # cube spawn yaw range (deg)
            self.detect_fail = 0
            E._Sim, old = _CamSim, E._Sim            # this env's sim renders the calibrated overhead camera
            try:
                super().__init__(episode_seconds=episode_seconds, seed=seed)
            finally:
                E._Sim = old
            self.observation_space = spaces.Box(-np.inf, np.inf, self._observe().shape, np.float32)

        def set_bias(self, bias_max):
            self.bias_max = float(bias_max)

        def set_yaw(self, yaw_max):
            self.yaw_max = float(yaw_max)

        def set_level(self, level):
            self.level = float(np.clip(level, 0.0, 1.0))

        # -------------------------------------------------------- layout + detection
        def _sample_layout(self, rng):
            """Cube anywhere in +-r; box with probability (1 - level) at the clamp design's spacing from the cube
            (ClampRLEnv's rule: box_dy from the design, the box can otherwise touch the left gripper while it
            closes), else anywhere in its region (detected by the camera either way)."""
            r = R0 + self.level * (R1 - R0)
            for _ in range(100):
                cx, cy = rng.uniform(-r, r), CUBE_Y + rng.uniform(-r, r)
                if rng.random() >= self.level:
                    bx, by, byaw = cx + rng.uniform(-0.005, 0.005), cy + C.BOX_DY - rng.uniform(0.0, 0.01), 0.0
                else:
                    bx, by = rng.uniform(-BOX_X, BOX_X), rng.uniform(*BOX_Y)
                    byaw = np.radians(rng.uniform(-BOX_YAW, BOX_YAW))
                if max(abs(cx - bx), abs(cy - by)) >= SEPARATION:
                    return cx, cy, bx, by, byaw
            return 0.0, CUBE_Y, 0.0, CUBE_Y + C.BOX_DY, 0.0

        def reset(self, *, seed=None, options=None):
            """ClampRLEnv.reset first (all of its per-episode randomisation and bookkeeping, e.g. sim-e5's grip
            friction / squeeze stiffness with CLAMP_GRIP_DR=1, so this env follows base-class changes), then this
            env's layout: wide cube / box spots, cube yaw range, servo bias, tracking reset, CLASSICAL box detection
            on the rendered overhead frame, and the plan to the DETECTED box."""
            m, d = self.model, self.data
            if not hasattr(self, "cam"):                 # first reset runs inside ClampRLEnv.__init__
                self.cam = V.OverheadCamera.from_env(self.sim)
            for attempt in range(5):
                CR.ClampRLEnv.reset(self, seed=seed if attempt == 0 else None)
                rng = self._rng
                self.sim.reset_tracking()
                self._servo_bias(rng)
                self._table_friction(rng)
                cx, cy, bx, by, byaw = self._sample_layout(rng)
                for s in C.SIDES:
                    self.sim.pad_quat[s] = C.clamp_quat(s, (cx, cy))
                yaw = np.radians(rng.uniform(-self.yaw_max, self.yaw_max))
                d.qpos[self._cq:self._cq + 7] = [cx, cy, C.CUBE_HALF + 0.001, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                d.qpos[self._bq:self._bq + 7] = [bx, by, E.CAN_HEIGHT / 2, np.cos(byaw / 2), 0, 0, np.sin(byaw / 2)]
                d.qvel[:] = 0
                mujoco.mj_forward(m, d)
                for _ in range(20):
                    mujoco.mj_step(m, d)
                det = V.detect_box(self._render_overhead(), self.cam)
                if det is not None:
                    break
                self.detect_fail += 1
            if det is None:                           # never seen (rare): fall back to the true spot, flagged
                det = (*self.base.can_pos()[:2], byaw, 0.0)
            self.box_det = np.array(det, float)
            box_xy = (float(det[0]), float(det[1]))
            self.plan = C.ClampPlan((cx, cy, 0.0), box_xy)
            self._nom = self.sim._ee_targets.copy()
            self._plan_id = id(self.plan)
            self.offset = np.zeros((2, 3))
            self.place_corr = np.zeros((2, 3))
            self._attempt_cube, self._box_xy = np.array([cx, cy]), box_xy
            self._box_start = self.base.can_pos()[:2].copy()
            self._pert, self._pert_at = "none", 0
            self._pert_left, self._pert_body, self._pert_force, self.perturbed = 0, None, None, False
            self.layout = dict(cube=(cx, cy), box_true=(bx, by, byaw), box_det=tuple(det), level=self.level)
            return self._observe(), {}

        def _render_overhead(self):
            """The overhead frame. The FIRST render creates this process's OpenGL context; on macOS several processes
            doing that at once deadlocked (workers stuck in mujoco.Renderer.__init__, 0 % CPU, 2026-11-12), so that
            first call is serialised across processes with a file lock."""
            if getattr(self, "_gl_ready", False):
                return self.sim.render_camera("overhead", native=True)
            import fcntl
            import tempfile
            with open(Path(tempfile.gettempdir()) / "clamp_wide_gl.lock", "w") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    rgb = self.sim.render_camera("overhead", native=True)
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
            self._gl_ready = True
            return rgb

        def _table_friction(self, rng):
            m = self.model
            t = m.geom("table").id
            if TABLE_MU is not None:                   # experiment override of the base sim's per-episode draw
                m.geom_priority[t] = 1
                m.geom_friction[t, 0] = rng.uniform(*TABLE_MU)
            self.table_mu = float(m.geom_friction[t, 0])

        def _servo_bias(self, rng):
            m = self.model
            ids = np.array([a for s in C.SIDES for a in self.sim.robot[s].actuator_ids[:5]])
            if not hasattr(self, "_bias0"):
                self._bias0 = m.actuator_biasprm[ids, 0].copy()
            kp = -m.actuator_biasprm[ids, 1]
            b = getattr(self, "bias_max", SERVO_BIAS_DEG)
            self.servo_bias_deg = rng.uniform(-b, b, len(ids)) if b else np.zeros(len(ids))
            m.actuator_biasprm[ids, 0] = self._bias0 - kp * np.radians(self.servo_bias_deg)

        def _observe(self):
            base = super()._observe()
            det = getattr(self, "box_det", np.zeros(4))
            off = getattr(self, "offset", np.zeros((2, 3)))
            extra = np.r_[off.ravel() / OFFSET_MAX, det[0], det[1], np.sin(4 * det[2]), np.cos(4 * det[2])]
            return np.concatenate([base, extra]).astype(np.float32)

        def _place_feedback(self):
            """Closed-loop drop (classical): while the lifted cube is carried / lowered, shift both arms so the
            MEASURED cube (x, y) ends over the DETECTED box centre. Needed because the joint tracking correction
            removes the arm's sag, which the open-loop drop point was tuned on (plan alone, ideal servos, any yaw:
            96 % in the box without tracking, 44 % with it). Frozen at release; reset on a new grab attempt."""
            if PLACE_FB and self.plan.phase in PLACE_PHASES and self._cube_bottom() > 0.01:
                err = np.asarray(self._box_xy) - self.base.cube_pos()[:2]
                self.place_corr[:, :2] = np.clip(self.place_corr[:, :2] + PLACE_GAIN * err, -PLACE_MAX, PLACE_MAX)
            elif self.plan.phase in ("hover", "backoff"):
                self.place_corr[:] = 0.0

        def rig_rules(self, target):
            """The real controller's PRELOAD CAP over the box (rig_tools/student_ctl.py, sim-e5, 2026-11-12): once the
            cube was held in this attempt and the fingertip midpoint is within 3 cm of the box, each tip's inward
            command stays within 5 mm of its measured position. Without it the arms were commanded ~2.6 cm past the
            cube while squeezing and snapped into each other when the cube dropped into the box (real guard: 7/15
            stops). Applied here so training sees what the rig enforces. WIDE_RIG_RULES=0 turns it off."""
            if not RIG_RULES:
                return target
            if id(self.plan) != getattr(self, "_rr_plan", None):
                self._rr_plan, self._rr_held, self._rr_over = id(self.plan), False, False
            tl, tr = self.base.tcp("left"), self.base.tcp("right")
            mid = 0.5 * (tl + tr)
            c = self.base.cube_pos()
            self._rr_held |= bool(np.linalg.norm(tl[:2] - tr[:2]) < 0.055 and np.linalg.norm(mid[:2] - c[:2]) < 0.025
                                  and min(tl[2], tr[2]) > 0.015)
            self._rr_over |= bool(self._rr_held and np.linalg.norm(mid[:2] - np.asarray(self._box_xy)) < 0.03)
            if self._rr_over:
                target = np.array(target, float).copy()
                target[0, 0] = min(target[0, 0], tl[0] + 0.005)
                target[1, 0] = max(target[1, 0], tr[0] - 0.005)
            return self._squeeze_limit(np.asarray(target, float), self.sim._ee_targets.copy())

        def _squeeze_limit(self, target, ee_before):
            g = getattr(self, "_guard", None)
            if not RIG_LIMITER or g is None:                 # needs the real guard (CLAMP_GUARD=1)
                return target

            def cmd_gap(t):
                q = {}
                for i, s in enumerate(C.SIDES):
                    jt, _ = self.sim._solve_ik(self.sim.robot[s], t[i], self.sim.pad_quat[s])
                    q[s] = np.r_[np.rad2deg(np.asarray(jt)[:5]), 0.0]
                return g.distances(q)["arm-arm"][0]

            if cmd_gap(target) >= SQUEEZE_MIN_GAP:
                return target
            t2 = target.copy()
            t2[0, 0] = min(t2[0, 0], ee_before[0, 0])        # left: no further +x (toward the right arm)
            t2[1, 0] = max(t2[1, 0], ee_before[1, 0])        # right: no further -x
            if cmd_gap(t2) >= SQUEEZE_MIN_GAP:
                return t2
            t3 = ee_before.copy()
            t3[:, 2] = target[:, 2]                          # only vertical
            if cmd_gap(t3) >= SQUEEZE_MIN_GAP:
                return t3
            return ee_before.copy()                          # still too close: hold (sim-e5's controller fix, 2026-11-13)

        # -------------------------------------------------------- residual control
        def step(self, action):
            a = np.clip(np.asarray(action, np.float64), -1, 1).reshape(2, 3)
            if id(self.plan) != self._plan_id:            # clamp_rl started a fresh attempt: restart from here
                self._plan_id = id(self.plan)
                self._nom = self.sim._ee_targets - self.offset - self.place_corr
            self._nom = self.plan.next_targets(self._nom)
            self._place_feedback()
            # the action is the offset TARGET (x OFFSET_MAX); the offset follows it at <= OFFSET_RATE per step. (v1
            # integrated the action: exploration noise random-walked the tips off the cube, 27 % success at level 0.)
            self.offset += np.clip(OFFSET_MAX * a - self.offset, -OFFSET_RATE, OFFSET_RATE)
            cmd = self.rig_rules(self._nom + self.offset + self.place_corr)
            base_a = (cmd - self.sim._ee_targets) / (self.sim.delta_scale * CR.ACTION_SCALE)
            obs, r, term, trunc, info = super().step(np.clip(base_a, -1, 1).ravel())
            info["level"] = self.level
            info["yaw_max"] = self.yaw_max
            info["bias_max"] = getattr(self, "bias_max", SERVO_BIAS_DEG)
            info["box_err_mm"] = float(1000 * np.linalg.norm(self.box_det[:2] - self.layout["box_true"][:2]))
            info["detect_fail"] = self.detect_fail
            return obs, r, term, trunc, info

    return ClampWideEnv


STAGES = ("squeezed", "lifted", "cleared", "placed", "is_success", "can_tipped")


# ------------------------------------------------------------------ checks without learning
def _check_worker(args):
    level, yaw, seeds, policy = args
    if os.environ.get("WIDE_HANG_DUMP_S"):                # debug: dump this worker's stack if it stalls
        import faulthandler
        faulthandler.dump_traceback_later(int(os.environ["WIDE_HANG_DUMP_S"]), exit=True)
    import torch
    Env = make_env_cls()
    env = Env(seed=int(seeds[0]), level=level, yaw_max=yaw)      # servo bias: WIDE_SERVO_BIAS
    net = None
    if policy:
        from stable_baselines3 import PPO
        net = PPO.load(policy, device="cpu")
        stats = json.loads(Path(str(policy).replace(".zip", "_obs_stats.json")).read_text())
        mu, var = np.array(stats["mean"]), np.array(stats["var"])
    out = []
    for s in seeds:
        o, _ = env.reset(seed=int(s))
        info = {}
        for _ in range(env.base.max_steps):
            if net is None:
                a = np.zeros(6)
            else:
                a, _ = net.predict(np.clip((o - mu) / np.sqrt(var + 1e-8), -10, 10), deterministic=True)
            o, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        out.append({**{k: bool(info.get(k)) for k in STAGES}, "clean": info.get("bad_contact_steps", 0) == 0,
                    "box_err_mm": info.get("box_err_mm"), "layout": env.layout})
    env.close()
    return out


def check(level, n, policy=None, workers=WORKERS, yaw=15.0):
    import multiprocessing as mp
    seeds = np.arange(1000, 1000 + n)
    chunks = [(level, yaw, c, policy) for c in np.array_split(seeds, workers) if len(c)]
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        res = [r for part in pool.map(_check_worker, chunks) for r in part]
    rate = {k: np.mean([r[k] for r in res]) for k in (*STAGES, "clean")}
    errs = [r["box_err_mm"] for r in res if r["box_err_mm"] is not None]
    print(f"level {level:.2f} yaw +-{yaw:.0f} ({'policy' if policy else 'plan only'}, n={n}): "
          + " ".join(f"{k} {v:.2f}" for k, v in rate.items())
          + f" | box detection error median {np.median(errs):.1f} mm, max {np.max(errs):.1f} mm", flush=True)
    return rate, res


# ------------------------------------------------------------------ training
def train(steps, workers, start_stage=0, resume=None):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    RUNS.mkdir(exist_ok=True)
    Env = make_env_cls()
    REWARD_SCALE = 0.01

    def mk(rank):
        def _init():
            import gymnasium as gym
            return gym.wrappers.TransformReward(Env(seed=rank, level=SCHEDULE[start_stage][0], yaw_max=SCHEDULE[start_stage][1]), lambda r: r * REWARD_SCALE)
        return _init

    venv = VecMonitor(SubprocVecEnv([mk(i) for i in range(workers)], start_method="spawn"),
                      info_keywords=(*STAGES, "bad_contact_steps", "level", "yaw_max", "bias_max", "box_err_mm"))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, gamma=0.99)
    if resume:
        vn = Path(str(resume).replace(".zip", "_vecnorm.pkl"))
        if vn.exists():                               # keep the policy's input normalisation
            venv = VecNormalize.load(str(vn), venv.venv)
            venv.training = True
        model = PPO.load(resume, env=venv, device="cpu")
    else:
        model = PPO("MlpPolicy", venv, learning_rate=1e-4, n_steps=512, batch_size=512, n_epochs=5, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.1, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, device="cpu",
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-3.5), verbose=0)
        # exploration noise sigma 0.03 (1.2 mm offset target): plan + noise 20/20 at level 0; 0.06 -> 80 %, 0.1 -> 45 %
        with torch.no_grad():                         # start as the plan alone: zero mean residual
            model.policy.action_net.weight.mul_(0.0)
            model.policy.action_net.bias.zero_()

    class Curriculum(BaseCallback):
        """Stage A (user, 2026-11-12): cube area fixed at +-1.5 cm, cube rotation widened 15 -> 45 deg (45 = any
        rotation of a square cube), so the policy learns to adjust the pinch to the cube's angle. Stage B: rotation
        stays random (+-45) while the cube area widens to +-12 cm. Each stage needs ADVANCE[A|B] success over the
        last EVAL_WINDOW episodes of that stage; a checkpoint is kept per finished stage."""

        def __init__(self):
            super().__init__()
            self.k, self.hist, self.t0, self.best = start_stage, [], time.time(), -1.0
            self.log = open(RUNS / "train_log.txt", "a")

        def _say(self, s):
            print(s, flush=True)
            self.log.write(s + "\n")
            self.log.flush()

        def _desc(self, k):
            lvl, yaw, bias = SCHEDULE[k]
            name = "B distance" if lvl > 0 else ("A2 rotation+servo error" if bias else "A rotation")
            return f"stage {k} [{name}] cube +-{100 * (R0 + lvl * (R1 - R0)):.1f} cm, yaw +-{yaw:.0f} deg, servo +-{bias:.0f} deg"

        def _save(self, stem, succ, clean):
            self.model.save(RUNS / f"{stem}.zip")
            venv.save(str(RUNS / f"{stem}_vecnorm.pkl"))
            (RUNS / f"{stem}_obs_stats.json").write_text(json.dumps({
                "mean": venv.obs_rms.mean.tolist(), "var": venv.obs_rms.var.tolist(), "stage": self.k,
                "level": SCHEDULE[self.k][0], "yaw": SCHEDULE[self.k][1], "bias": SCHEDULE[self.k][2],
                "success": succ, "clean": clean,
                "design": os.environ.get("CLAMP_DESIGN")}))

        def _apply(self):
            self.training_env.env_method("set_level", SCHEDULE[self.k][0])
            self.training_env.env_method("set_yaw", SCHEDULE[self.k][1])
            self.training_env.env_method("set_bias", SCHEDULE[self.k][2])

        def _on_training_start(self):
            self._apply()
            self._say(f"{time.strftime('%H:%M:%S')} start at {self._desc(self.k)} (design {os.environ.get('CLAMP_DESIGN')})")

        def _on_step(self):
            lvl, yaw, bias = SCHEDULE[self.k]
            for info in self.locals["infos"]:
                if info.get("episode") is None or abs(info.get("level", lvl) - lvl) > 1e-6 \
                        or abs(info.get("yaw_max", yaw) - yaw) > 1e-6 or abs(info.get("bias_max", bias) - bias) > 1e-6:
                    continue
                self.hist.append((bool(info.get("is_success")), info.get("bad_contact_steps", 0) == 0,
                                  bool(info.get("lifted")), bool(info.get("can_tipped"))))
            if len(self.hist) >= EVAL_WINDOW:
                h = np.array(self.hist[-EVAL_WINDOW:])
                succ, clean, lift, tip = h.mean(0)
                score = self.k + succ * clean
                self._say(f"{time.strftime('%H:%M:%S')} steps {self.num_timesteps:8d} {self._desc(self.k)} | last "
                          f"{EVAL_WINDOW}: success {succ:.2f} clean {clean:.2f} lifted {lift:.2f} box tipped {tip:.2f} "
                          f"[{time.time() - self.t0:.0f}s]")
                if score > self.best:
                    self.best = score
                    self._save("best", succ, clean)
                need = ADVANCE_A if lvl == 0 else ADVANCE_B
                self.hist = []
                if succ >= need and clean >= need - 0.1:
                    self._save(f"stage{self.k:02d}_passed", succ, clean)
                    if lvl == 0 and (self.k + 1 == len(SCHEDULE) or SCHEDULE[self.k + 1][0] > 0):
                        self._save("rotation_servo_done", succ, clean)
                        self._say("  == stage A2 done: any cube rotation in the +-1.5 cm area, with real-servo error ==")
                    elif lvl == 0 and bias == 0 and SCHEDULE[self.k + 1][2] > 0:
                        self._save("rotation_done", succ, clean)
                        self._say("  == stage A done: any cube rotation in the +-1.5 cm area (ideal servos) ==")
                    if self.k + 1 == len(SCHEDULE):
                        self._save("wide_done", succ, clean)
                        self._say("  == all stages done: cube +-12 cm, any rotation ==")
                        return False
                    self.k += 1
                    self._apply()
                    self._say(f"  -> {self._desc(self.k)}")
            if self.n_calls % 200 == 0:
                self.model.save(RUNS / "last.zip")
                venv.save(str(RUNS / "last_vecnorm.pkl"))
            return True

    model.learn(total_timesteps=steps, callback=Curriculum(), reset_num_timesteps=resume is None)
    model.save(RUNS / "final.zip")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--level", type=float, default=0.0)
    c.add_argument("--n", type=int, default=12)
    c.add_argument("--yaw", type=float, default=15.0)
    c.add_argument("--workers", type=int, default=WORKERS)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=4_000_000)
    t.add_argument("--workers", type=int, default=WORKERS)
    t.add_argument("--stage", type=int, default=0, help="index into SCHEDULE")
    t.add_argument("--resume", default=None)
    e = sub.add_parser("eval")
    e.add_argument("policy")
    e.add_argument("--level", type=float, default=1.0)
    e.add_argument("--n", type=int, default=48)
    e.add_argument("--yaw", type=float, default=45.0)
    e.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    if a.cmd == "check":
        check(a.level, a.n, workers=a.workers, yaw=a.yaw)
    elif a.cmd == "train":
        train(a.steps, a.workers, a.stage, a.resume)
    else:
        check(a.level, a.n, policy=str(HERE / a.policy) if not Path(a.policy).is_absolute() else a.policy, workers=a.workers, yaw=a.yaw)


if __name__ == "__main__":
    main()
