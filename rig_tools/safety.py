"""Safety layer for EVERY real SO-101 runner (real_clamp.py, real_student.py, ...). Added 2026-11-11 after a scripted
clamp run knocked a 3D-printed wrist camera holder off: the left wrist roll was asked for 162.8 deg but its real range
stops at ~107 deg (the command was silently clipped, the gripper stayed 56 deg off), and nothing checked for the arms
hitting each other or a camera mount.

Guard.step(measured, commanded) runs every tick BEFORE anything is sent to the servos and raises SafetyStop when:
  1. LIMITS     a commanded joint is outside that servo's real calibrated range (never silently clipped any more)
  2. COLLISION  the commanded pose, or the measured one, brings any part within a margin of a wrist camera mount
                (strictest), of the other arm, of the overhead camera post / head, or of the box walls -- exact convex
                distances (GJK, below) on the calibrated sim's collision shapes, one hull per arm link
  3. TRACKING   a joint stays > 15 deg from its command for 5 ticks (blocked, stalled, or at a hard stop)
It also limits how far a command may lead the measured joint (10 deg): a blocked joint can only be pushed with the
force of a 10 deg error, not an ever-growing one. Single-tick encoder wrap glitches (> 90 deg jumps) are ignored.
Guard.preflight(trajectory) checks a whole planned motion kinematically before the arms move.
Runners catch SafetyStop like STOP: the arms fold back to rest and torque goes off. The guard refuses to start if the
model lacks the camera-post or wrist-camera-mount collision shapes.
"""
from __future__ import annotations

import mujoco
import numpy as np

SIDES = ("left", "right")
NJ = 5                                   # arm joints checked (gripper is commanded shut separately)
LEAD_DEG = 10.0
TRACK_DEG, TRACK_TICKS = 15.0, 5
GLITCH_DEG = 90.0
LIMIT_MARGIN_DEG = 1.0
# metres. "cmd" = the pose about to be commanded (anticipates), "meas" = where the arms really are (last line).
# "camera-mount": any part of the other arm vs a WRIST CAMERA MOUNT (fragile 3D print: the part that broke);
# "arm-arm": every other arm-arm pair (solid printed links / jaws, which sit a cube-width apart by design).
# The commanded arm-arm margin is small on purpose: the gentle squeeze commands the tips 1.6 mm into the cube faces, so
# the COMMANDED grippers sit ~3 mm apart (measured ~10 mm, the cube holds them apart). 1.5 mm guarantees that even with
# no cube there they would stop short of touching; the camera mounts keep the large margins.
# 2026-11-12: 1.5 -> 0.5 mm after a real RL-policy run: the learned policy commands its grippers ~1.5-2.5 mm apart while
# squeezing the cube (real gap 14-15 mm, cube between them); 0.5 mm still forbids commanding them into each other.
MARGIN = {"cmd": {"camera-mount": 0.012, "arm-arm": 0.0005, "arm-camera": 0.015, "arm-box": 0.006},
          "meas": {"camera-mount": 0.006, "arm-arm": 0.003, "arm-camera": 0.008, "arm-box": 0.002}}
# per servo torque limit (0..1000): shoulder / elbow need force against gravity; wrist and jaw do not
TORQUE_LIMIT = {1: 700, 2: 700, 3: 700, 4: 500, 5: 450, 6: 350}
REG_TLIMIT = 48


class SafetyStop(Exception):
    pass


def real_limits(arms):
    """{side: (lo[5], hi[5])} in the runner's joint units (sim degrees, offsets included) from each servo's calibrated
    raw range, as stored in the servos."""
    out = {}
    for s in SIDES:
        a = arms[s]
        ulo, uhi = a.to_units(np.asarray(a.lo, float)), a.to_units(np.asarray(a.hi, float))
        out[s] = (np.minimum(ulo, uhi)[:NJ].copy(), np.maximum(ulo, uhi)[:NJ].copy())
    return out


def apply_torque_limits(arms):
    for a in arms.values():
        a.bus.sync_write(REG_TLIMIT, 2, [(i, [v & 0xFF, (v >> 8) & 0xFF]) for i, v in TORQUE_LIMIT.items()])


# ------------------------------------------------------------------ exact convex distance (GJK)
_SUBSETS = {k: [[i for i in range(k) if mask >> i & 1] for mask in range(1, 1 << k)] for k in range(1, 5)}


def _closest_on_simplex(S):
    """Closest point to the origin of conv(S), |S| <= 4, by checking every face (Johnson's sub-algorithm, brute force)."""
    best, keep = None, None
    for sub in _SUBSETS[len(S)]:
        P = S[sub]
        if len(sub) == 1:
            x = P[0]
        else:
            A = P[1:] - P[0]
            t = np.linalg.lstsq(A @ A.T, -A @ P[0], rcond=None)[0]
            if t.min() < -1e-12 or t.sum() > 1 + 1e-12:
                continue                                        # closest point of the affine hull is outside this face
            x = P[0] + t @ A
        n = x @ x
        if best is None or n < best[0]:
            best, keep = (n, x), sub
    return best[1], S[keep]


def convex_distance(XA, XB, cap):
    """Exact distance between the convex hulls of point sets XA, XB (0 if they overlap), capped at `cap`: it stops as
    soon as GJK proves the distance is at least `cap`."""
    v = XA.mean(0) - XB.mean(0)
    if not v.any():
        v = np.array([1.0, 0.0, 0.0])
    S = (XA[np.argmax(XA @ -v)] - XB[np.argmax(XB @ v)])[None]
    v = S[0]
    for _ in range(64):
        vn = float(np.linalg.norm(v))
        if vn < 1e-9:
            return 0.0
        w = XA[np.argmax(XA @ -v)] - XB[np.argmax(XB @ v)]             # support of A-B towards the origin
        lb = float(v @ w) / vn                                            # the distance is >= lb
        if lb >= cap:
            return cap
        if vn - lb <= 1e-7 + 1e-6 * vn:
            return min(vn, cap)
        v, S = _closest_on_simplex(np.vstack([S, w]))
    return min(float(np.linalg.norm(v)), cap)


class Guard:
    def __init__(self, model, limits, box_xy=None, log=print, box_yaw=0.0):
        self.m, self.d, self.log = model, mujoco.MjData(model), log
        m = model
        self.limits = limits
        body = lambda g: m.body(m.geom_bodyid[g]).name
        coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
        # one convex hull per arm link (all its collision shapes together: a little bulkier than the part = safe side)
        self.arm = {}
        for s in SIDES:
            parts = {}
            for g in range(m.ngeom):
                if body(g).startswith(s + "_") and coll(g):
                    parts.setdefault(int(m.geom_bodyid[g]), []).append(self._verts(g))
            self.arm[s] = [self._part(b, m.body(b).name, np.vstack(v)) for b, v in parts.items()]
        self.post = [self._part(int(m.geom_bodyid[g]), m.geom(g).name, self._verts(g)) for g in range(m.ngeom)
                     if m.geom(g).name in ("mount_post_collision", "mount_head_collision")]
        self.box = [self._part(int(m.geom_bodyid[g]), f"box/{m.geom(g).name}", self._verts(g)) for g in range(m.ngeom)
                    if body(g) == "can" and coll(g)]               # each wall separately (the open box is concave)
        for s in SIDES:
            if not any(p[1] == f"{s}_camera_mount" for p in self.arm[s]):
                raise SafetyStop(f"guard model has no {s} wrist camera mount collision shape: refusing to run")
        if len(self.post) < 2:
            raise SafetyStop("guard model has no overhead camera post / head collision shapes: refusing to run")
        self.qadr = {s: [m.jnt_qposadr[m.joint(f"{s}_{j}").id] for j in
                         ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")] for s in SIDES}
        self.grip = {s: (m.jnt_qposadr[m.joint(f"{s}_gripper").id],
                         *m.actuator_ctrlrange[m.actuator(f"{s}_gripper").id]) for s in SIDES}
        self._cube = m.jnt_qposadr[m.joint("cube_joint").id]
        self._can = m.jnt_qposadr[m.joint("can_joint").id]
        self.d.qpos[self._cube:self._cube + 7] = [5, 5, -1, 1, 0, 0, 0]       # the cube may be touched: park it
        self.set_box(box_xy, box_yaw)
        self.prev_meas, self.prev_cmd = None, None
        self.track = {s: np.zeros(NJ, int) for s in SIDES}
        self.closest = {}
        self._self_test()

    # ------------------------------------------------------------ geometry
    def _verts(self, g):
        """Collision shape g as points in its BODY frame (box corners / mesh vertices)."""
        m = self.m
        t = m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            h = m.geom_size[g]
            V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            ga = m.mesh_graphadr[mid]
            if ga >= 0:                                    # keep only the convex hull's vertices (same hull, fewer points)
                nv = m.mesh_graph[ga]
                V = V[m.mesh_graph[ga + 2 + nv: ga + 2 + 2 * nv]]
        elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE):
            r = m.geom_size[g][0]
            hl = m.geom_size[g][1] if t != mujoco.mjtGeom.mjGEOM_SPHERE else 0.0
            a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            ring = np.c_[r * np.cos(a), r * np.sin(a)]
            V = np.vstack([np.c_[ring, np.full(12, z)] for z in (-hl - (r if t == 3 else 0), hl + (r if t == 3 else 0))])
        else:
            raise SafetyStop(f"guard: unsupported collision shape type {t} ({m.geom(g).name})")
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        return m.geom_pos[g] + V @ R.reshape(3, 3).T

    @staticmethod
    def _part(b, name, V):
        if len(V) > 300:                                   # thin a big point cloud to its extreme points
            k = np.arange(1500) + 0.5                      # (1500 evenly spread directions, Fibonacci sphere)
            phi, th = np.arccos(1 - 2 * k / 1500), np.pi * (1 + 5 ** 0.5) * k
            dirs = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
            V = V[np.unique(np.argmax(V @ dirs.T, axis=0))]
        c = 0.5 * (V.min(0) + V.max(0))
        return (b, name, V, c, float(np.sqrt(((V - c) ** 2).sum(1).max())))

    def _world(self, part):
        b, name, V, c, r = part
        R = self.d.xmat[b].reshape(3, 3)
        return name, self.d.xpos[b] + V @ R.T, self.d.xpos[b] + R @ c, r

    def set_box(self, xy, yaw=0.0):
        """Box at xy (the detected position), rotated by yaw (rad) about the vertical (2026-11-13: the real box sat at
        -5 deg; an axis-aligned model is mm off at the walls). yaw 0 = the previous behaviour."""
        self.has_box = xy is not None
        p = [xy[0], xy[1], 0.015] if xy is not None else [6, 6, -1]
        y = float(yaw or 0.0)
        self.d.qpos[self._can:self._can + 7] = [*p, np.cos(y / 2), 0, 0, np.sin(y / 2)]

    def _pose(self, q):
        for s in SIDES:
            self.d.qpos[self.qadr[s]] = np.deg2rad(np.asarray(q[s][:NJ], float))
            adr, glo, ghi = self.grip[s]
            u = float(q[s][5]) if len(q[s]) > 5 else 0.0             # jaw: 0 shut .. 100 open (as the runners)
            self.d.qpos[adr] = glo + np.clip(u, 0, 100) / 100 * (ghi - glo)
        mujoco.mj_kinematics(self.m, self.d)

    def _min_dist(self, A, B, cap):
        WA = [self._world(p) for p in A]
        WB = [self._world(p) for p in B]
        cand = []
        for na, XA, ca, ra in WA:
            for nb, XB, cb, rb in WB:
                lb = float(np.linalg.norm(ca - cb)) - ra - rb          # bounding spheres: lower bound on the distance
                if lb < cap:
                    cand.append((lb, na, XA, nb, XB))
        best, pair = cap, None
        for lb, na, XA, nb, XB in sorted(cand, key=lambda x: x[0]):
            if lb >= best:
                break
            dist = convex_distance(XA, XB, best)
            if dist < best:
                best, pair = dist, (na, nb)
        return best, pair

    def _self_test(self):
        """Check the distance code against cases with a known answer before trusting it with the arms."""
        rng = np.random.default_rng(0)
        cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * 0.01
        worst = 0.0
        for _ in range(40):                                    # two 2 cm cubes, random rotation, known face gap
            gap = rng.uniform(0.0, 0.06)
            Rq = np.zeros(9)
            q = rng.normal(size=4)
            mujoco.mju_quat2Mat(Rq, q / np.linalg.norm(q))
            B = cube @ Rq.reshape(3, 3).T
            off = 0.01 + gap - B[:, 0].min()                   # B's lowest x sits `gap` beyond A's face x = +0.01
            got = convex_distance(cube, B + [off, 0, 0], 1.0)
            ref = gap if np.all(np.abs((B + [off, 0, 0])[np.argmin(B[:, 0])][1:]) <= 0.01) else None
            if ref is not None:
                worst = max(worst, abs(got - ref))
        overlap = convex_distance(cube, cube + [0.005, 0, 0], 1.0)
        if worst > 1e-4 or overlap != 0.0:
            raise SafetyStop(f"guard self-test failed (error {worst * 1000:.2f} mm, overlap {overlap}): refusing to run")
        self.self_test_mm = round(worst * 1000, 4)

    def distances(self, q, cap=0.05):
        """Closest distance per kind for joint pose q ({side: 5 or 6 deg/units}), with the parts involved."""
        self._pose(q)
        mount = {s: [p for p in self.arm[s] if p[1].endswith("camera_mount")] for s in SIDES}
        rest = {s: [p for p in self.arm[s] if not p[1].endswith("camera_mount")] for s in SIDES}
        out = {"camera-mount": min(self._min_dist(mount["left"], self.arm["right"], cap),
                                   self._min_dist(rest["left"], mount["right"], cap), key=lambda x: x[0]),
               "arm-arm": self._min_dist(rest["left"], rest["right"], cap)}
        out["arm-camera"] = min((self._min_dist(self.arm[s], self.post, cap) for s in SIDES), key=lambda x: x[0])
        if self.has_box:
            out["arm-box"] = min((self._min_dist(self.arm[s], self.box, cap) for s in SIDES), key=lambda x: x[0])
        return out

    def check_pose(self, q, which):
        bad = []
        dist = self.distances(q)
        for kind, margin in MARGIN[which].items():
            if kind in dist and dist[kind][0] < margin:
                bad.append(f"{kind} {dist[kind][0] * 1000:.1f} mm < {margin * 1000:.1f} mm ({' vs '.join(dist[kind][1])}) "
                           + ("in the COMMANDED pose (where the arms were being sent, not where they are)" if which == "cmd"
                              else "in the REAL (measured) pose"))
        self.closest[which] = {k: round(v[0] * 1000, 1) for k, v in dist.items()}
        return bad

    def check_limits(self, cmd):
        bad = []
        for s in SIDES:
            lo, hi = self.limits[s]
            q = np.asarray(cmd[s][:NJ], float)
            for j in np.nonzero((q < lo - LIMIT_MARGIN_DEG) | (q > hi + LIMIT_MARGIN_DEG))[0]:
                bad.append(f"{s} joint {j} commanded {q[j]:.1f} deg, real range {lo[j]:.1f}..{hi[j]:.1f}")
        return bad

    # ------------------------------------------------------------ per tick
    def filter_meas(self, meas):
        meas = {s: np.asarray(meas[s], float).copy() for s in SIDES}
        if self.prev_meas is not None:
            for s in SIDES:
                jump = np.abs(meas[s][:NJ] - self.prev_meas[s][:NJ]) > GLITCH_DEG
                meas[s][:NJ][jump] = self.prev_meas[s][:NJ][jump]          # encoder wrap glitch: keep last value
        self.prev_meas = meas
        return meas

    def step(self, meas, cmd, expect=None):
        """meas, cmd: {side: >= 5 joint degrees}. Returns the (lead-limited) command to send, or raises SafetyStop.
        expect: the pose the arm is expected to reach when the command deliberately over-shoots to cancel the servos'
        steady-state shortfall (tip feedback, clamp_control.py); the collision check then uses it instead of the
        command. Limits and the lead limit always apply to the real command."""
        meas = self.filter_meas(meas)
        problems = self.check_limits(cmd)
        out = {}
        for s in SIDES:
            c = np.asarray(cmd[s], float).copy()
            c[:NJ] = meas[s][:NJ] + np.clip(c[:NJ] - meas[s][:NJ], -LEAD_DEG, LEAD_DEG)
            out[s] = c
        problems += self.check_pose(meas, "meas")
        cmd_bad = self.check_pose(out if expect is None else expect, "cmd")
        if cmd_bad:
            real = self.closest.get("meas", {})
            cmd_bad[-1] += f" [the real arms at that moment: {', '.join(f'{k} {v:.1f} mm' for k, v in real.items())}]"
        problems += cmd_bad
        if self.prev_cmd is not None:
            for s in SIDES:
                err = np.abs(self.prev_cmd[s][:NJ] - meas[s][:NJ])
                self.track[s] = np.where(err > TRACK_DEG, self.track[s] + 1, 0)
                for j in np.nonzero(self.track[s] >= TRACK_TICKS)[0]:
                    problems.append(f"{s} joint {j} stuck {err[j]:.0f} deg from its command for {TRACK_TICKS} ticks "
                                    f"(blocked or at a limit)")
        if problems:
            raise SafetyStop("; ".join(problems))
        self.prev_cmd = out
        return out

    def preflight(self, trajectory):
        """trajectory: list of {side: 5 joint degrees} (commanded). Returns None, or (index, problems) of the first bad
        pose. Kinematic only: run it on the plan before the arms move."""
        for i, q in enumerate(trajectory):
            bad = self.check_limits(q) + self.check_pose(q, "cmd")
            if bad:
                return i, bad
        return None
