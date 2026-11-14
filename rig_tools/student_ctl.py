"""Controller for sim-74's ACTION-CHUNKED clamp student (squeeze/runs_chunk/rotation/student.pt) on the rig.

ONE class used by both the sim test (squeeze/student_ctl_simtest.py) and the real runner (real_chunk_student.py), so
the sim test exercises exactly the rig's control logic. Per tick it gets what the rig has -- the three camera frames
(640x480 RGB) and the measured joints already written into the kinematic shadow -- and returns the joint commands.

  cube    triangulated from every camera that sees red (cube_locator; wrist poses from the measured joints), raw as in
          training (cube_locator.SimCubeSensor: no smoothing, a held cube reads low because the tips hide its top),
          held while hidden. Plausibility: outside the table area, or > 2.5 cm from the fingertips while they hold
          the cube -> rejected (held). A jump > 2 cm is only taken once 3 readings agree (red wires, 2026-11-12).
  yaw     vision.detect_cube_pose on the overhead frame, held while not found (start refused until found once)
  box     vision.detect_box on the overhead frame, once, before moving
  clock   ticks since the current grab attempt / 450; a new attempt as clamp_rl._update_attempt does it, from what the
          rig measures (cube on the table and moved early / grab missed / dropped away from the box)
  IK      gripper orientation clamp_teacher.clamp_quat(side, cube_xy) of the twist-offset design, set at each attempt
          start and blended over 30 ticks (as ClampRLEnv)
  step    StudentRunner target -> workspace clip -> IK -> rate limit -> squeeze limiter (commanded grippers >= 1 mm
          apart; inward x dropped) ; the executed step is fed back as the student's "previous step"
The caller then runs the safety guard + floor guard on the returned command (real runner) or counts would-be stops
(sim test).

Real-time options (2026-11-13, rt_control / so101_control.json; all default off, so the sim test is unchanged):
  tick(..., dets=)  camera detections computed on a background thread ({"seq", "red": {cam: (uv, n)}, "pose"});
                    the same seq twice -> the cube reading is held, not re-counted
  tick(..., adv=)   training ticks this real tick stands for (real period / 0.04 s): attempt clock, attempt timers,
                    orientation blend, IK rate limit and the student's step limit all advance by it
  filters=          cfg["perception"]: a cube more than table_height_tol above the table while the fingertips are
                    NOT closed on it is re-read from the overhead ray on the table plane (2026-11-13 run: wrist-camera
                    triangulation put the cube at 4.5-5 cm on the table); yaw through rt_control.PeriodicAngleFilter
"""
from __future__ import annotations

import numpy as np

SIDES = ("left", "right")
CAMS = ("overhead", "left_wrist", "right_wrist")
CLOCK_STEPS, EARLY_STEPS, MISSED_STEPS, QUAT_STEPS = 450, 135, 210, 30
# "cube moved" needs a REAL move: training decides it from the true cube position; on the rig the triangulated reading
# jumps 1.2-1.4 cm while the closing grippers hide part of the cube (2026-11-13 sim test: up to 21 false restarts in one
# episode). So: readings taken while a fingertip is within MOVED_CLEAR of the cube do not count, and the shift must
# persist MOVED_TICKS ticks in a row.
# The arm bodies hide part of the cube from ~6 cm out and the shift lasts ~16 ticks, so: > 2.5 cm (the occlusion shift
# is 1.2-1.4 cm), fingertips > 6 cm away, 10 ticks in a row. Training (no disturbances) essentially never has a move.
MOVED_CLEAR, MOVED_TICKS, MOVED_DIST = 0.06, 10, 0.025
SQUEEZE_MIN_GAP = 0.001
# RELEASE GAP (2026-11-12 sim test, interim until the guard-trained student): once the fingertips are over the box,
# the commanded grippers are kept >= RELEASE_GAP apart (moved apart by up to half a step per tick), so they do not
# swing into each other while opening over the box (the real guard's measured arm-arm margin is 3 mm).
RELEASE_GAP, OVER_BOX = 0.015, 0.03
# PRELOAD CAP over the box: while squeezing, each arm is commanded deeper than the cube lets it go (sim: right pan
# commanded -45.7 deg, measured -37.5 = ~2.6 cm of preload at the tips). When the cube drops into the box the arms snap to
# those commands and hit each other within 3 ticks (the commanded-gap rule cannot see it: the commands are apart, the
# MEASURED arms are not). Over the box, a tip's inward command is kept within RELEASE_PRELOAD of where it measurably is.
RELEASE_PRELOAD = 0.005
JUMP = 0.02
CLOSED_GAP = 0.036          # fingertips (FK) closer than this in x-y = closed on the 3 cm cube (squeeze: ~14 mm)


class StudentCtl:
    def __init__(self, m, d, sim, C, V, CL, runner, guard, squeeze_limit=True, release_gap=RELEASE_GAP, filters=None):
        self.m, self.d, self.sim, self.C, self.V, self.CL = m, d, sim, C, V, CL
        self.run, self.guard, self.squeeze_limit, self.release_gap = runner, guard, squeeze_limit, release_gap
        self.filters, self.yaw_filter, self._seq, self._sensed = filters, None, None, None
        self.intr = {c: CL.cam_intrinsics(sim.camera_specs[c]) for c in CAMS}
        self.cam = V.OverheadCamera.from_env(sim)

    # ------------------------------------------------------------------ sensing
    def _triangulate(self, frames, dets=None):
        CL, m, d = self.CL, self.m, self.d
        rays, seen_by = [], []
        self._overhead_ray = None
        for c in CAMS:
            uv, n = dets["red"][c] if dets is not None else CL.detect_red(frames[c])
            if uv is not None:
                rays.append(CL.ray(self.intr[c], *CL.camera_pose(m, d, c), *uv))
                seen_by.append(f"{c}({n}px)")
                if c == "overhead":
                    self._overhead_ray = rays[-1]
        prefer = next((i for i, s in enumerate(seen_by) if s.startswith("overhead")), 0)
        xyz, how = CL.locate(rays, prefer=prefer)
        return xyz, how, seen_by

    def sense(self, frames, dets=None):
        """(cube_xyz, seen, yaw, info). Raw triangulation as in training, held while hidden or implausible."""
        if dets is not None and dets.get("seq") == self._seq and self._sensed is not None:
            return self._sensed                        # no new camera result since the last tick: hold, don't re-count
        C = self.C
        xyz, how, seen_by = self._triangulate(frames, dets)
        f = self.filters
        if f and xyz is not None and xyz[2] > f["cube_half_m"] + f["table_height_tol_m"] and self._overhead_ray is not None:
            tl0, tr0 = self.sim.robot["left"].position, self.sim.robot["right"].position
            if np.linalg.norm(tl0[:2] - tr0[:2]) >= CLOSED_GAP:      # not closed on it -> it is on the table
                xyz2, _ = self.CL.locate([self._overhead_ray])
                if xyz2 is not None:
                    xyz, how = xyz2, how + f" (read {xyz[2] * 100:.1f} cm high; overhead on the table plane)"
        tl, tr = self.sim.robot["left"].position, self.sim.robot["right"].position
        mid = 0.5 * (tl + tr)
        holding = self.xyz is not None and np.linalg.norm(tl[:2] - tr[:2]) < 0.06 and \
            np.linalg.norm(mid[:2] - self.xyz[:2]) < 0.035
        if xyz is not None and not (-0.12 <= xyz[0] <= 0.12 and C.CUBE_Y - 0.17 <= xyz[1] <= C.CUBE_Y + 0.10):
            xyz, how = None, how + " (rejected: outside the table area)"
        if xyz is not None and holding and np.linalg.norm(xyz[:2] - mid[:2]) > 0.025:
            xyz, how = None, how + " (rejected: not between the fingertips)"
        seen = 0.0
        if xyz is not None:
            if self.xyz is None or np.linalg.norm(xyz[:2] - self.xyz[:2]) <= JUMP or holding:
                self.xyz, seen, self.pending = np.array(xyz, float), 1.0, []
            else:
                self.pending.append(np.array(xyz, float))
                p = self.pending[-3:]
                if len(p) == 3 and max(np.linalg.norm(q[:2] - p[-1][:2]) for q in p) < 0.01:
                    self.xyz, seen, self.pending = np.mean(p, axis=0), 1.0, []
                    how += " (jump confirmed by 3 readings)"
                else:
                    how += " (jump rejected)"
        pose = dets["pose"] if dets is not None else self.V.detect_cube_pose(frames["overhead"], self.cam)
        if self.yaw_filter is not None:
            y = self.yaw_filter.update(None if pose is None else float(pose[2]))
            if y is not None:
                self.yaw = y
        elif pose is not None:
            self.yaw = float(pose[2])
        self.info = f"{how}; seen by {', '.join(seen_by) or 'none'}"
        if dets is not None:
            self._seq = dets.get("seq")
            self._sensed = (self.xyz, seen, self.yaw, self.info)
        return self.xyz, seen, self.yaw, self.info

    # ------------------------------------------------------------------ start / attempts
    def start(self, frames, dets=None):
        """Box + cube before moving. Returns (box (x, y, yaw, iou), cube_xyz, yaw) or raises RuntimeError."""
        self.xyz, self.yaw, self.pending, self.info = None, None, [], ""
        self._seq, self._sensed = None, None
        if self.filters:
            import rt_control as RT
            f = self.filters
            self.yaw_filter = RT.PeriodicAngleFilter(f["yaw_period_deg"], f["yaw_window"], f["yaw_jump_deg"], f["yaw_confirm"])
        box = self.V.detect_box(frames["overhead"], self.cam)
        if box is None:
            raise RuntimeError("box not found in the overhead frame")
        self.sense(frames, dets)
        if self.xyz is None:
            raise RuntimeError("cube not found by the cameras")
        if self.yaw is None:
            raise RuntimeError("cube orientation not found in the overhead frame")
        self.box = box
        self.run.reset(box=(box[0], box[1], box[2]))
        self.k = 0
        self.attempt_start, self.attempt_cube, self.attempts = 0, self.xyz[:2].copy(), 1
        self.ever_held, self.need_replan = False, False
        self.over_box = False
        self.moved_count = 0
        for s in SIDES:
            self.sim.pad_quat[s] = self.C.clamp_quat(s, tuple(self.xyz[:2]))
        self.quat_t, self.quat_from, self.quat_to = QUAT_STEPS, None, None
        return box, self.xyz.copy(), self.yaw

    def _holding_now(self):
        tl, tr = self.sim.robot["left"].position, self.sim.robot["right"].position
        mid = 0.5 * (tl + tr)
        return (np.linalg.norm(tl[:2] - tr[:2]) < 0.055 and np.linalg.norm(mid[:2] - self.xyz[:2]) < 0.025
                and min(tl[2], tr[2]) > 0.015)

    def _update_attempt(self, seen, adv=1.0):
        """clamp_rl._update_attempt from rig measurements: cube on the table (not between the tips) somewhere this
        attempt cannot finish (moved early / grab missed / dropped) and not at the box -> new attempt."""
        C = self.C
        held = self._holding_now()
        self.ever_held |= held
        on_table = seen and self.xyz[2] < C.CUBE_HALF + 0.006 and not held
        t = self.k - self.attempt_start
        tips = [self.sim.robot[s].position[:2] for s in SIDES]
        clear = min(np.linalg.norm(p - self.attempt_cube) for p in tips) > MOVED_CLEAR
        shifted = seen and clear and np.linalg.norm(self.xyz[:2] - self.attempt_cube) > MOVED_DIST
        self.moved_count = self.moved_count + adv if shifted else 0
        moved = t < EARLY_STEPS and self.moved_count >= MOVED_TICKS
        missed = t > MISSED_STEPS and not self.ever_held
        near_box = np.linalg.norm(np.asarray(self.box[:2]) - self.xyz[:2]) < 0.05
        if self.ever_held and on_table and not near_box:
            self.need_replan = True
        if on_table and not near_box and (moved or missed or self.need_replan):
            self.last_reason = "moved" if moved else "missed" if missed else "dropped"
            self.need_replan, self.ever_held, self.over_box, self.moved_count = False, False, False, 0
            self.attempt_start, self.attempt_cube = self.k, self.xyz[:2].copy()
            self.attempts += 1
            self.quat_from = {s: self.sim.pad_quat[s].copy() for s in SIDES}
            self.quat_to = {s: C.clamp_quat(s, tuple(self.xyz[:2])) for s in SIDES}
            self.quat_t = 0
        if self.quat_t < QUAT_STEPS:
            self.quat_t = min(QUAT_STEPS, self.quat_t + adv)
            w = self.quat_t / QUAT_STEPS
            for s in SIDES:
                q0, q1 = self.quat_from[s], self.quat_to[s]
                q = (1 - w) * q0 + w * (q1 if q0 @ q1 >= 0 else -q1)
                self.sim.pad_quat[s] = q / np.linalg.norm(q)

    # ------------------------------------------------------------------ one tick
    def tick(self, frames, ctrl_before, dets=None, adv=1.0):
        """frames: {cam: RGB}; shadow qpos = measured joints (mj_forward done); ctrl_before: {side: commanded joints
        (rad) of the previous tick}. dets / adv: see the module notes (defaults = the sim test's behaviour).
        Returns ({side: 6 command degrees (gripper 0)}, log dict)."""
        sim, d = self.sim, self.d
        xyz, seen, yaw, info = self.sense(frames, dets)
        self._update_attempt(seen, adv)
        clock = (self.k - self.attempt_start) / CLOCK_STEPS
        joints = {s: np.rad2deg(d.qpos[sim.robot[s].qpos_ids]) for s in SIDES}
        grip = {s: sim.robot[s].gripper_opening for s in SIDES}
        fk = np.array([sim.robot[s].position for s in SIDES])
        ee_before = sim._ee_targets.copy()
        target = self.run.step(joints, grip, fk, ee_before, xyz, seen, yaw, clock, adv=adv)
        rate = sim._rate_limit * adv

        def command(tgt):
            out = {}
            for i, s in enumerate(SIDES):
                arm = sim.robot[s]
                sim._ee_targets[i] = np.clip(tgt[i], sim._workspace_low[i], sim._workspace_high[i])
                jt, _ = sim._solve_ik(arm, sim._ee_targets[i], None)
                cur = ctrl_before[s]
                new = cur + np.clip(jt - cur, -rate, rate)
                d.ctrl[arm.actuator_ids] = new
                out[s] = np.r_[np.rad2deg(new), 0.0]
            return out
        mid = 0.5 * (fk[0] + fk[1])
        # over the box only AFTER the cube was held in this attempt (the start pose is already near the box)
        self.over_box |= bool(self.ever_held and np.linalg.norm(mid[:2] - np.asarray(self.box[:2])) < OVER_BOX)
        released = False
        if self.release_gap and self.over_box:
            target = target.copy()
            target[0, 0] = min(target[0, 0], fk[0, 0] + RELEASE_PRELOAD)     # left: at most 5 mm inward (+x) of measured
            target[1, 0] = max(target[1, 0], fk[1, 0] - RELEASE_PRELOAD)     # right: at most 5 mm inward (-x)
            cmd = command(target)
            if self.guard.distances(cmd)["arm-arm"][0] < self.release_gap:
                released = True
                sep = 0.5 * self.run.max_step * adv
                target = target.copy()
                target[0, 0] = min(target[0, 0], ee_before[0, 0]) - sep     # left arm away from the right
                target[1, 0] = max(target[1, 0], ee_before[1, 0]) + sep     # right arm away from the left
        cmd = command(target)
        gap = self.guard.distances(cmd)["arm-arm"][0]
        limited = False
        if self.squeeze_limit and gap < SQUEEZE_MIN_GAP:
            limited = True
            t2 = target.copy()
            t2[0, 0] = min(t2[0, 0], ee_before[0, 0])        # left: no further +x (toward the right arm)
            t2[1, 0] = max(t2[1, 0], ee_before[1, 0])        # right: no further -x
            cmd = command(t2)
            if self.guard.distances(cmd)["arm-arm"][0] < SQUEEZE_MIN_GAP:
                t3 = ee_before.copy()
                t3[:, 2] = target[:, 2]                      # only vertical
                cmd = command(t3)
                if self.guard.distances(cmd)["arm-arm"][0] < SQUEEZE_MIN_GAP:
                    # still too close (vertical motion / the IK reshuffling the joints can close it too, 2026-11-13: commanded
                    # left gripper 0.3-0.5 mm from the right wrist): hold the previous command exactly
                    sim._ee_targets[:] = ee_before
                    for s in SIDES:
                        d.ctrl[sim.robot[s].actuator_ids] = ctrl_before[s]
                    cmd = {s: np.r_[np.rad2deg(ctrl_before[s]), 0.0] for s in SIDES}
        self.run.prev_step = (sim._ee_targets - ee_before) / adv     # what was executed, per training tick
        self.k += adv
        log = {"cube_xyz": [round(float(v), 4) for v in xyz], "cube_seen": int(seen), "cube_yaw_deg": round(np.degrees(yaw), 1),
               "clock": round(clock, 3), "attempt": self.attempts, "attempt_reason": getattr(self, "last_reason", ""), "cube_how": info, "target": np.round(target, 4).tolist(),
               "ee_cmd": np.round(sim._ee_targets, 4).tolist(), "squeeze_limited": limited, "over_box": self.over_box,
               "release_gap_applied": released,
               "cmd_gap_mm": round(float(gap) * 1000, 2)}
        return cmd, log
