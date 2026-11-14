"""Slide the cube into the clamp's proven zone first, so a cube anywhere on the reachable table can be clamped.

The two-arm tip clamp (clamp_teacher.py, design v3) is verified only in a small patch (cube ~25.7-27.7 cm in front of
the bases, |x| <= 2 cm, box at its fixed spot). This plan uses ONE arm, fingers pointing straight down and jaws shut
(jaw line across the push direction so both tips meet the cube face), to push the cube along the table in a straight
line onto TARGET, then lifts and returns. The other arm holds still. Fingertips ride ~6 mm above the table and catch
the lower half of the 32 mm cube. Open loop from the cube position; the caller re-detects and repeats if needed.

    plan = PushPlan(env.sim, cube_xy)          # picks the arm, sets its gripper orientation
    if plan.ok: targets = plan.next_targets(env.sim._ee_targets)   # (2, 3) every 40 ms tick, like ClampPlan
"""
from __future__ import annotations

import mujoco
import numpy as np

import clamp_teacher as C

TARGET = np.array([0.0, C.CUBE_Y + 0.01])          # centre of the zone where clamp v3 is clean with the fixed box
ZONE_HALF = (0.01 if "twistoff" in __import__("os").environ.get("CLAMP_DESIGN", "") else 0.006, 0.008)                           # |dx|, |dy| from TARGET that count as "in the zone" (no push);
                                                     # x tight: the clamp takes a twisted cube only near the centre line
Z_PUSH = 0.020                                       # tool point height while pushing: tips ~14 mm up as commanded, ~7 mm with the arm's sag at full reach
Z_HIGH = 0.07                                        # travel height
BEHIND = C.CUBE_HALF + 0.025                         # start this far behind the cube centre (tool point)
CONTACT = C.CUBE_HALF + 0.008                        # tool point to cube centre when the fingers touch the face
PUSH_STEP = 0.5                                      # push speed as a fraction of the clamp's step cap
IK_TOL = 0.005


import os as _os
# how twisted a cube the clamp can take: 10 deg for the v3 design, 25 deg for design_clamp3b_twist.json
MAX_YAW_DEG = 22.0 if "twist" in _os.environ.get("CLAMP_DESIGN", "") else 10.0


def yaw_off(yaw_deg):
    """How far the (square) cube is from square, in degrees (0..45)."""
    return abs((yaw_deg + 45.0) % 90.0 - 45.0)


def in_zone(cube_xy, target=TARGET, yaw_deg=None):
    d = np.abs(np.asarray(cube_xy[:2]) - target)
    ok = bool(d[0] <= ZONE_HALF[0] and d[1] <= ZONE_HALF[1])
    return ok and (yaw_deg is None or yaw_off(yaw_deg) <= MAX_YAW_DEG)


DRIFT_Y_PER_X = 0.45                                 # sideways pushes drift the cube this much away from the arms
OVERSHOOT = 0.004                                    # aim this far past the target (drag shortens pushes)


def next_push_target(cube_xy, target=TARGET):
    """Axis-aligned pushes keep the cube square for the clamp (a diagonal push rotates it): first fix y (toward /
    away from the arms), then x (sideways, toward the middle). Returns the point to push the cube centre to, or
    None if it is already in the zone."""
    c = np.asarray(cube_xy[:2], float)
    t = np.asarray(target, float)
    if in_zone(c, t):
        return None
    if abs(c[1] - t[1]) > ZONE_HALF[1]:
        goal = np.array([c[0], t[1]])
    else:
        goal = np.array([t[0], c[1]])
    u = goal - c
    n = np.linalg.norm(u)
    return goal + u / n * OVERSHOOT if n > 1e-6 else goal


def down_quat(u, flip):
    approach = np.array([0.0, 0.0, -1.0])
    close = np.r_[-u[1], u[0], 0.0] * (-1.0 if flip else 1.0)
    close /= np.linalg.norm(close)
    R = np.column_stack([approach, np.cross(close, approach), close])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


class PushPlan:
    def __init__(self, sim, cube_xy, target=TARGET):
        c = np.asarray(cube_xy[:2], float)
        delta = np.asarray(target, float) - c
        self.distance = float(np.linalg.norm(delta))
        self.ok, self.why, self.side = False, "", None
        self.waypoints, self._i, self._k = [], 0, 0
        if self.distance < 1e-3:
            self.why = "already on target"
            return
        u = delta / self.distance
        pre = c - u * BEHIND
        end = np.asarray(target, float) - u * CONTACT
        # pick the arm + jaw direction whose IK reaches the start and the end of the push best
        best = None
        saved = {s: sim.pad_quat[s].copy() for s in C.SIDES}
        for side in C.SIDES:
            arm = sim.robot[side]
            for flip in (False, True):
                sim.pad_quat[side] = down_quat(u, flip)
                err = 0.0
                for p in (pre, end, (pre + end) / 2):
                    q, _ = sim._solve_ik(arm, np.r_[p, Z_PUSH], None)
                    err = max(err, self._fk_err(sim, arm, q, np.r_[p, Z_PUSH]))
                # prefer the arm on the side the push starts from (the other would reach across)
                score = err + (0.0 if np.sign(pre[0]) == (-1 if side == "left" else 1) or abs(pre[0]) < 0.02 else 0.002)
                if best is None or score < best[0]:
                    best = (score, err, side, flip)
            sim.pad_quat[side] = saved[side]
        _, err, side, flip = best
        if err > IK_TOL:
            self.why = f"push out of reach (IK error {err * 1000:.1f} mm)"
            return
        self.ok, self.side, self.flip, self.u = True, side, flip, u
        sim.pad_quat[side] = down_quat(u, flip)
        i = C.SIDES.index(side)
        hold = sim._ee_targets.copy()
        n_push = int(np.ceil(np.linalg.norm(end - pre) / (C.MAX_STEP * PUSH_STEP))) + 10

        def wp(name, p, k):
            t = hold.copy()
            t[i] = p
            return (name, t[0].copy(), t[1].copy(), k)
        self.waypoints = [wp("p_rise", np.r_[hold[i][:2], Z_HIGH], 30),
                          wp("p_over", np.r_[pre, Z_HIGH], 45),
                          wp("p_down", np.r_[pre, Z_PUSH], 30),
                          wp("p_push", np.r_[end, Z_PUSH], n_push),
                          wp("p_back", np.r_[end - u * 0.02, Z_PUSH], 12),
                          wp("p_up", np.r_[end - u * 0.02, Z_HIGH], 25),
                          wp("p_home", hold[i], 45)]

    @staticmethod
    def _fk_err(sim, arm, q, p):
        d = mujoco.MjData(sim.model)
        d.qpos[:] = sim.data.qpos
        d.qpos[arm.qpos_ids] = q
        mujoco.mj_kinematics(sim.model, d)
        return float(np.linalg.norm(d.site_xpos[arm.site_id] - p))

    @property
    def phase(self):
        return self.waypoints[min(self._i, len(self.waypoints) - 1)][0] if self.waypoints else "done"

    @property
    def done(self):
        return self._i >= len(self.waypoints)

    def next_targets(self, commanded):
        if self.done:
            return np.asarray(commanded, float).copy()
        name, l, r, k = self.waypoints[self._i]
        cap = C.MAX_STEP * (PUSH_STEP if name == "p_push" else 1.0)
        step = np.clip(np.stack([l, r]) - commanded, -cap, cap)
        self._k += 1
        if self._k >= k:
            self._i, self._k = self._i + 1, 0
        return commanded + step


class ClosedLoopPush:
    """Closed-loop pusher (the teacher for a push policy): every tick it re-reads the cube position and steers ONE
    arm's fingertips (straight down, jaws shut, ~Z_PUSH) to sit just behind the cube on the line to the target,
    advancing it toward the target. If the line swings by more than REAPPROACH_DEG it lifts and comes around again.
    Done when the cube is in the zone and the arm is back at its start pose.

        push = ClosedLoopPush(sim, cube_xy)
        while not push.done: targets = push.step(sim._ee_targets, cube_xy_now)
    """
    MAX_STROKES = 6
    ADVANCE = 0.004            # how far inside the contact point the fingertip target sits while pushing (m)
    TOL = 0.004                # stop pushing when the cube centre is this close to the target (m)

    def __init__(self, sim, cube_xy, target=TARGET):
        self.sim, self.target = sim, np.asarray(target, float)
        self.home = sim._ee_targets.copy()
        self.mode, self.side, self.u, self.done, self.why = "plan", None, None, False, ""
        self.ticks, self.strokes = 0, 0
        self._choose(np.asarray(cube_xy[:2], float))

    def _choose(self, c, yaw=None):
        delta = self.target - c
        if in_zone(c, self.target, yaw):
            self.mode = "home"
            return
        # sideways first (any drift in y is fixed afterwards), then a final straight forward / back stroke: a flat
        # jaw face square to y both sets the depth and squares the cube for the clamp
        if abs(delta[0]) > ZONE_HALF[0] * 0.9:
            # sideways strokes drift the cube AWAY from the arms (~0.45 cm per cm pushed, sim map 2026-11-12) and a
            # push back toward the arms is usually out of reach: aim toward the arms by that much, stop short in y
            goal = np.array([self.target[0], min(c[1], self.target[1]) - DRIFT_Y_PER_X * abs(delta[0])])
        else:
            dy = delta[1]
            if abs(dy) < 0.002:                                # depth fine, only twisted: short square-up stroke
                dy = -0.004 if c[1] > self.target[1] else 0.004
            goal = np.array([c[0], c[1] + dy])
        self.stroke_target = goal
        n = np.linalg.norm(goal - c)
        u = (goal - c) / max(n, 1e-6)
        probe = PushPlan(self.sim, c, goal)                  # reuse its arm / jaw choice and reach check
        if not probe.ok:
            self.done, self.why = True, probe.why
            return
        self.side, self.u, self.i = probe.side, u, C.SIDES.index(probe.side)
        self.mode = "approach_high"

    def _arm_target(self, commanded, p, cap):
        t = commanded.copy()
        hold = self.home.copy()
        t[1 - self.i] = commanded[1 - self.i] + np.clip(hold[1 - self.i] - commanded[1 - self.i], -cap, cap)
        t[self.i] = commanded[self.i] + np.clip(p - commanded[self.i], -cap, cap)
        return t

    def step(self, commanded, cube_xy, yaw_deg=None):
        self.ticks += 1
        self.yaw = yaw_deg
        c = np.asarray(cube_xy[:2], float)
        cap = C.MAX_STEP
        if self.done:
            return commanded.copy()
        if self.mode == "home":
            t = commanded + np.clip(self.home - commanded, -cap, cap)
            if np.abs(self.home - commanded).max() < 1e-3:
                self.done = True
            return t
        i = self.i
        here = commanded[i]
        delta = self.target - c
        dist = np.linalg.norm(delta)
        if self.mode == "push":
            u = self.u                                     # fixed for the whole stroke
            perp = np.r_[-u[1], u[0]]
            along_cube, along_tgt = c @ u, self.stroke_target @ u
            off_line = abs((c - self.stroke_target) @ perp)
            if along_cube >= along_tgt - 0.002 or self.ticks > 900 or off_line > 0.03:
                self.strokes += 1
                self.after = "lift" if self.strokes >= self.MAX_STROKES else "relift"
                self.mode, self.back_to = "retreat", np.r_[here[:2] - u * 0.012, Z_PUSH]
            else:
                along = min(here[:2] @ u + C.MAX_STEP * PUSH_STEP, along_cube - (C.CUBE_HALF - 0.002))
                p = np.r_[u * along + perp * (c @ perp), Z_PUSH]
                return self._arm_target(commanded, p, cap * PUSH_STEP)
        if self.mode == "retreat":                         # back off flat first: lifting while touching drags the cube
            if np.linalg.norm(here - self.back_to) < 0.002:
                self.mode = self.after
            return self._arm_target(commanded, self.back_to, cap * PUSH_STEP)
        if self.mode in ("lift", "relift"):
            p = np.r_[here[:2], Z_HIGH]
            if here[2] > Z_HIGH - 0.003:
                if self.mode == "lift" or in_zone(c, self.target, self.yaw):
                    self.mode = "home"
                else:
                    self.ticks = 0
                    self._choose(c, self.yaw)
                    if self.mode == "approach_high":
                        self.sim.pad_quat[self.side] = down_quat(self.u, PushPlan(self.sim, c, self.stroke_target).flip)
            return self._arm_target(commanded, p, cap)
        pre = c - self.u * BEHIND
        if self.mode == "approach_high":
            p = np.r_[pre, Z_HIGH]
            if np.linalg.norm(here - p) < 0.003:
                self.mode = "approach_low"
            return self._arm_target(commanded, p, cap)
        if self.mode == "approach_low":
            p = np.r_[pre, Z_PUSH]
            if np.linalg.norm(here - p) < 0.003:
                self.mode = "push"
            return self._arm_target(commanded, p, cap)
        return commanded.copy()


class SquarePlan:
    """Two-arm squaring move: both arms, fingers straight down and jaws shut (jaw line along y), come down beside the
    cube and close in from both sides to just outside its faces. The two flat faces turn the cube square and centre it
    in x; then they open, lift and return. Waypoints / next_targets like ClampPlan."""
    OUT, IN, Z = C.CUBE_HALF + 0.045, C.CUBE_HALF + 0.004, Z_PUSH

    def __init__(self, sim, cube_xy, x_centre=None):
        c = np.asarray(cube_xy[:2], float)
        xc = c[0] if x_centre is None else x_centre
        self.ok, self.why = True, ""
        home = sim._ee_targets.copy()
        for side, u in (("left", np.array([1.0, 0.0])), ("right", np.array([-1.0, 0.0]))):
            arm, best = sim.robot[side], None
            sgn = -1.0 if side == "left" else 1.0
            for flip in (False, True):
                sim.pad_quat[side] = down_quat(u, flip)
                err = max(PushPlan._fk_err(sim, arm, sim._solve_ik(arm, np.r_[xc + sgn * x, c[1], self.Z], None)[0],
                                           np.r_[xc + sgn * x, c[1], self.Z]) for x in (self.OUT, self.IN))
                if best is None or err < best[0]:
                    best = (err, flip)
            sim.pad_quat[side] = down_quat(u, best[1])
            if best[0] > IK_TOL:
                self.ok, self.why = False, f"{side} cannot reach the squaring pose ({best[0] * 1000:.1f} mm)"

        def both(xo, z):
            return (np.r_[xc - xo, c[1], z], np.r_[xc + xo, c[1], z])
        self.waypoints = [("s_rise", np.r_[home[0][:2], Z_HIGH], np.r_[home[1][:2], Z_HIGH], 30),
                          ("s_over", *both(self.OUT, Z_HIGH), 45), ("s_down", *both(self.OUT, self.Z), 30),
                          ("s_close", *both(self.IN, self.Z), 40), ("s_open", *both(self.OUT, self.Z), 25),
                          ("s_up", *both(self.OUT, Z_HIGH), 25), ("s_home", home[0], home[1], 45)]
        self.waypoints = [(n, np.asarray(l, float), np.asarray(r, float), k) for n, l, r, k in self.waypoints]
        self._i, self._k = 0, 0

    phase = PushPlan.phase
    done = PushPlan.done

    def next_targets(self, commanded):
        if self.done:
            return np.asarray(commanded, float).copy()
        name, l, r, k = self.waypoints[self._i]
        cap = C.MAX_STEP * (PUSH_STEP if name == "s_close" else 1.0)
        step = np.clip(np.stack([l, r]) - commanded, -cap, cap)
        self._k += 1
        if self._k >= k:
            self._i, self._k = self._i + 1, 0
        return commanded + step
