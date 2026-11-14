#!/usr/bin/env python3
"""Cube anywhere within +-12 cm: PUSH it into the clamp's proven spot, then CLAMP it into the box the camera found.

User, 2026-11-12: longer-distance fixes using the scripted IK solution. The residual-PPO curriculum (clamp_wide.py)
stalled at +-4.7 cm (53-64 % success for 750k steps; plan alone at +-12 cm: 4 %): the clamp's geometry only works near
the centre. So, two phases, both scripted IK at the base:
  1. push  -- push_teacher.ClosedLoopPush (sim-e5): one arm, fingers down, re-reads the cube every tick and pushes it to
              push_teacher.TARGET (the clamp's clean zone), up to 6 strokes; skipped when the cube is already there
  2. clamp -- the clamp plan from the cube's new position to the DETECTED box (vision.detect_box at reset) + the
              trained rotation/servo-spread residual policy (runs_clamp_wide/rotation_servo_done.zip)
Cube position: true in sim (teacher) or the CLASSICAL overhead detector (vision.detect_cube_pose) with --detect.

    CLAMP_GRIP_DR=1 CLAMP_SLIP=1 WIDE_SERVO_BIAS=1 python squeeze/push_clamp_wide.py --level 1 --n 48 --workers 8
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

CLAMP_POLICY = "runs_clamp_wide/rotation_servo_done.zip"
EPISODE_S = float(os.environ.get("PC_EPISODE_S", "80"))   # push (<= PUSH_MAX_TICKS + homing) + a full clamp with retries
PUSH_MAX_TICKS = 600


CUBE_ALPHA, CUBE_JUMP = 0.4, 0.04  # camera cube estimate: exponential smoothing; a jump > 4 cm is taken as is
CLAMP_REGION = 0.02               # skip the push when the cube is within this (x and y) of the clamp's trained spot
APPROACH_TICKS = 80               # abort a stroke whose approach has not arrived by then (unreachable start point)
BOX_CLEAR = 0.05 + 0.025          # box half-width + clearance: fingertips must not come down within this of its centre


# push the cube this much farther from the box than the design spacing. 2026-11-13, real guard on, cube from the
# cameras: 42/60 guard stops, nearly all arm-box (pusher arm passing the wall, clamp working against it) at 7 mm
PUSH_MARGIN = float(os.environ.get("PC_PUSH_MARGIN", "0.007"))
PUSH_Z = 0.024                    # pusher tool height (sim-e5's 0.020 scraped the table: 413 arm-table steps in 24 episodes)
PUSH_IK_TOL = 0.008               # pusher reach check: a fingertip up to 8 mm off still pushes (5 mm refused +3 cm cubes)


def make_box_aware_push():
    import clamp_teacher as C
    import push_teacher as PT
    PT.IK_TOL = PUSH_IK_TOL
    PT.Z_PUSH = PUSH_Z

    class BoxAwarePush(PT.ClosedLoopPush):
        """sim-e5's ClosedLoopPush, but it knows where the (camera-detected) box is: a stroke whose approach point
        (fingertips coming down behind the cube) lands on / next to the box is not used. The strokes are tried in
        both orders (sideways first, as the base class does, or forward / back first); if neither is clear the cube
        is first pushed sideways out from in front of the box. Without this the pusher tipped the box over for every
        cube 3-6 cm in front of it (2026-11-12 map)."""

        def __init__(self, sim, cube_xy, target, box_xy):
            self.box = np.asarray(box_xy, float)[:2]
            super().__init__(sim, cube_xy, target)

        def _blocked(self, c, goal):
            u = (goal - c) / max(np.linalg.norm(goal - c), 1e-6)
            for k in np.linspace(0.0, 1.0, 6):                    # approach point and the path down to the cube
                p = c - u * (PT.BEHIND * (1 - k) + (C.CUBE_HALF + 0.004) * k)
                if np.all(np.abs(p - self.box) < BOX_CLEAR):
                    return True
            return False

        def _choose(self, c, yaw=None):
            delta = self.target - c
            if PT.in_zone(c, self.target, yaw):
                self.mode = "home"
                return
            cands = []
            need_x = abs(delta[0]) > PT.ZONE_HALF[0] * 0.9
            side_goal = np.array([self.target[0], min(c[1], self.target[1]) - PT.DRIFT_Y_PER_X * abs(delta[0])])
            dy = delta[1] if abs(delta[1]) >= 0.002 else (-0.004 if c[1] > self.target[1] else 0.004)
            fwd_goal = np.array([c[0], c[1] + dy])
            cands = [side_goal, fwd_goal] if need_x else [fwd_goal, side_goal]
            for goal in cands:
                if np.linalg.norm(goal - c) > 0.003 and not self._blocked(c, goal):
                    break
            else:                                                   # detour: sideways out from in front of the box
                sgn = np.sign(c[0] - self.box[0]) or (1.0 if self.target[0] >= self.box[0] else -1.0)
                goal = np.array([self.box[0] + sgn * (BOX_CLEAR + 0.03), c[1]])
                if self._blocked(c, goal):
                    self.done, self.why = True, "every push stroke would hit the box"
                    return
            self.stroke_target = goal
            u = (goal - c) / max(np.linalg.norm(goal - c), 1e-6)
            probe = PT.PushPlan(self.sim, c, goal)
            if not probe.ok:
                self.done, self.why = True, probe.why
                return
            self.side, self.u, self.i = probe.side, u, C.SIDES.index(probe.side)
            self.mode = "approach_high"

    return BoxAwarePush


_NETS = {}


class StudentClamp:
    """The deployable chunked MLP student (chunk_student.py: 41-dim sensor obs, K-step tool-point chunks, ACT temporal
    ensembling, executed through the rig rules) driving the CLAMP phase after the scripted push -- the MLP version of
    push-then-clamp (user, 2026-11-13: teach the chunked model push then clamp). The net is loaded once per process."""

    def __init__(self, env, path, seed):
        import torch
        import chunk_student as CS
        self.CS, self.torch = CS, torch
        if path not in _NETS:
            net = CS.build_net()
            net.load_state_dict(torch.load(path, map_location="cpu"))
            _NETS[path] = net.eval()
        self.net = _NETS[path]
        self.sens = CS.Sensors(env, np.random.default_rng(int(seed) + 31))
        self.reset()

    def reset(self):
        self.ens, self.prev = self.CS.ChunkEnsembler(), np.zeros((2, 3))


class PushClamp:
    """Drives a ClampWideEnv episode: push phase, then clamp phase. step() returns the env step result."""

    def __init__(self, env, teacher, detect=False, student=None):
        self.env, self.teacher, self.detect, self.student = env, teacher, detect, student

    def cube(self):
        """Cube (x, y) for the pusher / clamp start. Truth in sim, or with --detect what the rig would use: the
        three-camera triangulation (cube_locator.SimCubeSensor here = ray-cast visibility, partial occlusion, pixel
        noise; on the rig sim-e5's cube_locator), falling back to the overhead red-blob detector, holding the last fix
        while nothing sees the cube. (Overhead-only lost the cube under the pushing arm: band 80 -> 53 %, every
        x >= +3 cm start failed.)"""
        env = self.env
        if not self.detect:
            return env.base.cube_pos()[:2]
        if getattr(self, "_tri", None) is None:
            import cube_locator as CL
            self._tri = CL.SimCubeSensor(env.model, env.data, env.sim.camera_specs, np.random.default_rng(0))
        xyz, how, _ = self._tri.locate()
        if xyz is not None:
            z = np.asarray(xyz[:2])
            prev = getattr(self, "_last", None)
            # light smoothing of the pusher's estimate: pixel noise + partial occlusion made the closed-loop pusher
            # steer on jitter (pushes landing in the clamp zone: 43 % with sim truth, 22 % with raw camera readings)
            self._last = z if prev is None or np.linalg.norm(z - prev) > CUBE_JUMP else prev + CUBE_ALPHA * (z - prev)
            return self._last
        import vision as V
        r = V.detect_cube_pose(env._render_overhead(), env.cam)
        if r is not None:
            self._last = np.array(r[:2])
        return getattr(self, "_last", env.base.cube_pos()[:2])

    def start(self, obs):
        import push_teacher as PT
        env = self.env
        self.obs, self.phase, self.ticks = obs, "push", 0
        # the arms' starting joint configuration: the clamp is trained from it, and the IK (warm-started from the
        # MEASURED joints every tick) only finds the trained gripper configuration from there
        self.q_start = {s: env.data.ctrl[env.sim.robot[s].actuator_ids].copy() for s in env.sim.robot}
        self.ee_start = env.sim._ee_targets.copy()
        env._update_attempt = lambda *a, **k: None          # no clamp re-plans while pushing
        import clamp_teacher as C
        c0 = self.cube()
        # the clamp's spot is set by the DETECTED box: the design's box spacing (ClampRLEnv: box = cube + BOX_DY) plus
        # PUSH_MARGIN; a fixed spot let the cube land too close to the box -> left gripper on the box wall (most of the
        # band's contacts)
        bx, by = env._box_xy
        target = np.array([bx, by - C.BOX_DY + PUSH_MARGIN])
        self.target = target
        self.push = make_box_aware_push()(env.sim, c0, target, env._box_xy)
        if np.all(np.abs(c0 - target) <= CLAMP_REGION):       # the clamp can take it from here: no push
            self.push.mode, self.push.done = "home", True
        self._mode_ticks, self._mode = 0, self.push.mode
        if self.push.side is not None and self.push.u is not None:
            # same cube estimate the push was planned from (a 2nd self.cube() read moves the smoothed camera estimate,
            # and a plan from there can come out "out of reach" with no .flip -- crashed every --detect run 2026-11-13)
            plan = PT.PushPlan(env.sim, c0, self.push.stroke_target)
            if plan.ok:
                env.sim.pad_quat[self.push.side] = PT.down_quat(self.push.u, plan.flip)
        self.pushed = not self.push.done or self.push.mode != "home"

    def _begin_clamp(self):
        import clamp_rl as CR
        import clamp_teacher as C
        env = self.env
        c_now = env.base.cube_pos()[:2]
        self.after_push = c_now.copy()
        self.push_landed = bool(np.all(np.abs(c_now - self.target) <= CLAMP_REGION))
        for s in C.SIDES:                                     # start the clamp from the clamp orientation, as trained
            env.sim.pad_quat[s] = C.clamp_quat(s, c_now)
        del env._update_attempt                               # back to ClampRLEnv's re-grab logic
        c = self.cube()
        env.plan = C.ClampPlan((c[0], c[1], 0.0), env._box_xy)
        env._plan_id = id(env.plan)
        env._nom = env.sim._ee_targets.copy()
        env.offset[:] = 0.0
        env.place_corr[:] = 0.0
        env._attempt_cube, env._attempt_start = np.array(c), env.base._step
        env._quat_t = CR.QUAT_STEPS                            # no blend: already in the clamp orientation
        env._shape_reach = True
        self.phase = "clamp"
        self.obs = env._observe()
        if self.student is not None:
            self.student.reset()

    def _rehome_tick(self):
        """Joint-space move back to the starting joint configuration (rate-limited like the sim's own joint targets).
        After a push the wrist can sit in another IK branch (measured: right wrist roll +86 deg after a push vs -157
        deg from the start pose, same cube pose -> tips 8-14 mm off, no lift). On the rig: a plain 'go to start pose'."""
        import mujoco
        env = self.env
        sim, d = env.sim, env.data
        err = 0.0
        for s, arm in sim.robot.items():
            cur = d.ctrl[arm.actuator_ids]
            d.ctrl[arm.actuator_ids] = cur + np.clip(self.q_start[s] - cur, -sim._rate_limit, sim._rate_limit)
            err = max(err, float(np.abs(self.q_start[s] - d.qpos[arm.qpos_ids]).max()))
        for _ in range(sim._substeps):
            mujoco.mj_step(env.model, d)
        mujoco.mj_forward(env.model, d)
        env.base._step += 1
        sim._ee_targets[:] = self.ee_start
        self.rehome_ticks = getattr(self, "rehome_ticks", 0) + 1
        if err < np.radians(2.0) or self.rehome_ticks > 100:
            self._begin_clamp()
        self.obs = env._observe()
        term = bool(env.base.can_tipped())
        trunc = env.base._step >= env.base.max_steps
        return self.obs, 0.0, term, trunc, {"is_success": False, "can_tipped": term, "bad_contact_steps": int(sum(env.bad_counts.values()))}

    def step(self):
        import clamp_rl as CR
        env = self.env
        if self.phase == "push":
            self.ticks += 1
            # yaw=None: no squareness requirement (the rotation-trained clamp takes any yaw; with the 22 deg limit the
            # pusher kept re-pushing a cube that was already on target until it timed out)
            tgt = self.push.step(env.sim._ee_targets.copy(), self.cube(), None)
            a = (tgt - env.sim._ee_targets) / (env.sim.delta_scale * CR.ACTION_SCALE)
            out = CR.ClampRLEnv.step(env, np.clip(a, -1, 1).ravel())
            self.obs = out[0]
            self._mode_ticks = self._mode_ticks + 1 if self.push.mode == self._mode else 0
            self._mode = self.push.mode
            stalled = self.push.mode in ("approach_high", "approach_low") and self._mode_ticks > APPROACH_TICKS
            if (self.ticks == PUSH_MAX_TICKS or stalled) and not self.push.done and self.push.mode not in ("lift", "home"):
                self.push.mode = "lift"                       # out of time / unreachable: lift, go home, then clamp
            if self.push.done or self.ticks >= PUSH_MAX_TICKS + 150:
                self.phase = "rehome"
            return out
        if self.phase == "rehome":
            return self._rehome_tick()
        if self.student is not None:                           # the chunked student clamps (as chunk_student's eval)
            import clamp_wide as W
            st, CS = self.student, self.student.CS
            o_s = CS.student_obs(env, st.sens, st.prev)
            a_t = self.teacher.act(self.obs).reshape(2, 3)     # shadow the teacher's residual offset (env bookkeeping)
            env.offset += np.clip(W.OFFSET_MAX * a_t - env.offset, -W.OFFSET_RATE, W.OFFSET_RATE)
            with st.torch.no_grad():
                chunk = st.net(st.torch.from_numpy(o_s[None])).numpy()[0].reshape(CS.K, 2, 3) / CS.SCALE
            st.ens.add(env.sim._ee_targets + chunk)
            out, st.prev = CS.execute_target(env, st.ens.next())
            self.obs = out[0]
            return out
        out = env.step(self.teacher.act(self.obs))
        self.obs = out[0]
        return out


def _set_start(env, xy):
    """Map mode: put the cube at a given start (box stays where reset put it), re-settle, keep the detected box."""
    import mujoco
    import clamp_teacher as C
    m, d = env.model, env.data
    q = d.qpos[env._cq:env._cq + 7].copy()
    d.qpos[env._cq:env._cq + 3] = [xy[0], xy[1], C.CUBE_HALF + 0.001]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(10):
        mujoco.mj_step(m, d)
    for s in C.SIDES:
        env.sim.pad_quat[s] = C.clamp_quat(s, xy)
    env.layout["cube"] = tuple(xy)
    return env._observe()


def _worker(args):
    level, yaw, seeds, detect, policy = args[:5]
    starts = args[5] if len(args) > 5 else None
    import clamp_wide as W
    import chunk_student as CS
    teacher = CS.Teacher(str(HERE / policy))
    Env = W.make_env_cls()
    env = Env(seed=int(seeds[0]), level=level, yaw_max=yaw, episode_seconds=EPISODE_S)
    out = []
    student_path = os.environ.get("PC_STUDENT")
    for k, s in enumerate(seeds):
        o, _ = env.reset(seed=int(s))
        if starts is not None:
            o = _set_start(env, starts[k])
        pc = PushClamp(env, teacher, detect, StudentClamp(env, student_path, s) if student_path else None)
        pc.start(o)
        start = env.base.cube_pos()[:2].copy()
        info, push_ticks = {}, 0
        for _ in range(env.base.max_steps):
            _, _, te, tr, info = pc.step()
            push_ticks += pc.phase == "push"
            if te or tr:
                break
        out.append({"needed_push": not (pc.push.done and pc.push.mode == "home" and pc.ticks <= 1),
                    "landed": getattr(pc, "push_landed", None),
                    "end": "tipped" if info.get("can_tipped") else ("time" if tr else ("term" if te else "?")),
                    "success": bool(info.get("is_success")), "clean": info.get("bad_contact_steps", 0) == 0,
                    "lifted": bool(info.get("lifted")), "start": start.tolist(), "push_ticks": push_ticks,
                    "pushed": pc.pushed, "push_why": pc.push.why, "box": env.layout["box_true"],
                    "guard": info.get("guard_stop"), "phase": pc.phase})
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--yaw", type=float, default=45.0)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--detect", action="store_true", help="cube position from the classical overhead detector")
    ap.add_argument("--policy", default=CLAMP_POLICY)
    ap.add_argument("--grid", type=float, default=0.0, help="map mode: start grid step in cm (box at its home spot)")
    ap.add_argument("--band", type=float, default=0.0, help="map mode: only rows within +-band cm of the nominal depth")
    ap.add_argument("--reps", type=int, default=1, help="map mode: episodes per cell (random cube yaw)")
    ap.add_argument("--student", default="", help="chunked MLP student (student.pt) for the clamp phase")
    ap.add_argument("--vla-starts", type=int, default=0,
                    help="N episodes on the SmolVLA test's layouts (eval_smolvla.py --pushclamp 3, seeds 300000..): "
                         "cube x +-12 cm, y +-3 cm, box at the design spacing; the first 20 = the VLA's 20")
    a = ap.parse_args()
    if a.student:
        os.environ["PC_STUDENT"] = str(Path(a.student) if Path(a.student).is_absolute() else HERE / a.student)
    if a.grid:
        return grid_map(a)
    if a.vla_starts:
        return vla_starts(a)
    seeds = np.arange(1000, 1000 + a.n)
    jobs = [(a.level, a.yaw, [int(x) for x in c], a.detect, a.policy) for c in np.array_split(seeds, a.workers) if len(c)]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = [r for part in pool.map(_worker, jobs) for r in part]
    succ = np.mean([r["success"] for r in res])
    clean = np.mean([r["success"] and r["clean"] for r in res])
    print(f"push+clamp level {a.level} (cube +-{1.5 + a.level * 10.5:.1f} cm) yaw +-{a.yaw:.0f}, cube from "
          f"{'classical detector' if a.detect else 'sim truth'}: success {succ:.2f} clean {clean:.2f} lifted "
          f"{np.mean([r['lifted'] for r in res]):.2f} (n={len(res)})", flush=True)
    # where it fails: start-position map (x, y relative to the nominal cube spot)
    for r in res:
        if not r["success"]:
            print(f"  fail: start ({r['start'][0] * 100:+.1f}, {(r['start'][1] - 0.0372) * 100:+.1f}) cm, pushed {r['pushed']}, "
                  f"push ticks {r['push_ticks']}, lifted {r['lifted']}, clean {r['clean']}, ended {r['end']} {r['push_why']}", flush=True)


def vla_starts(a):
    """Same layouts as the SmolVLA push + rotation test (eval_smolvla.py --pushclamp 3): level-0 box, cube moved to
    x +-12 cm, y +-3 cm (rng = default_rng(seed)), seeds 300000.. -- so MLP and VLA numbers line up episode by episode."""
    import clamp_teacher as C
    seeds = np.arange(300000, 300000 + a.vla_starts)
    starts = []
    for s in seeds:
        rng = np.random.default_rng(int(s))
        starts.append((rng.uniform(-0.12, 0.12), C.CUBE_Y + rng.uniform(-3.0, 3.0) / 100))
    idx = np.array_split(np.arange(len(seeds)), a.workers)
    jobs = [(0.0, a.yaw, [int(seeds[i]) for i in ii], a.detect, a.policy, [starts[i] for i in ii]) for ii in idx if len(ii)]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = [r for part in pool.map(_worker, jobs) for r in part]
    order = [i for ii in idx for i in ii]                      # pool results come back in job order
    res = [r for _, r in sorted(zip(order, res))]
    who = ("chunked MLP student " + Path(os.environ["PC_STUDENT"]).parent.name) if os.environ.get("PC_STUDENT") \
        else "teacher " + a.policy
    for q in res:
        q["rig_ok"] = q["success"] and not q["guard"]
        q["clean_ok"] = q["success"] and q["clean"]
        q["guarded"] = bool(q["guard"])
    for i, r in enumerate(res):
        print(f"  ep {i} start x {starts[i][0] * 100:+5.1f} cm: {'DELIVERED' if r['success'] else 'failed'}"
              f"{'' if r['clean'] or not r['success'] else ' (contact)'}"
              f"{' | guard stop (' + r['phase'] + '): ' + ', '.join(r['guard']) if r['guard'] else ''} | ended {r['end']}",
              flush=True)

    def rate(rs, k):
        return f"{sum(q[k] for q in rs)}/{len(rs)}"
    left = [q for q, s in zip(res, starts) if s[0] < 0]
    right = [q for q, s in zip(res, starts) if s[0] >= 0]
    print(f"SUMMARY push (scripted, cube from {'cameras' if a.detect else 'sim truth'}) + clamp by {who}: "
          f"delivered {rate(res, 'success')}, clean {rate(res, 'clean_ok')}, guard stops {rate(res, 'guarded')}; "
          f"first 20 (= the VLA test) delivered {rate(res[:20], 'success')}; left starts {rate(left, 'success')}, "
          f"right starts {rate(right, 'success')}; guard stops while pushing / re-homing "
          f"{sum(bool(q['guard']) and q['phase'] != 'clamp' for q in res)}, while clamping "
          f"{sum(bool(q['guard']) and q['phase'] == 'clamp' for q in res)}; flags: CLAMP_GUARD="
          f"{os.environ.get('CLAMP_GUARD')} CMD={os.environ.get('CLAMP_GUARD_CMD')} GRIP_DR={os.environ.get('CLAMP_GRIP_DR')} "
          f"SLIP={os.environ.get('CLAMP_SLIP')} BIAS={os.environ.get('WIDE_SERVO_BIAS')}", flush=True)


def grid_map(a):
    """Success map over cube start positions (x, y from the nominal spot, +-12 cm), box at the home spot (level 0
    layout = the design spacing would move it, so level 0 with the cube moved after reset keeps the box put)."""
    import clamp_teacher as C
    g = np.arange(-12, 12.01, a.grid)
    gy = np.arange(-a.band, a.band + 0.01, a.grid) if a.band else g
    pts = [(x, y) for y in gy for x in g for _ in range(a.reps)]
    starts = [(x / 100, C.CUBE_Y + y / 100) for x, y in pts]
    seeds = np.arange(3000, 3000 + len(pts))
    idx = np.array_split(np.arange(len(pts)), a.workers)
    jobs = [(0.0, a.yaw, [int(seeds[i]) for i in ii], a.detect, a.policy, [starts[i] for i in ii]) for ii in idx if len(ii)]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = [r for part in pool.map(_worker, jobs) for r in part]
    cell, allres = {}, {}
    for (x, y), r in zip(pts, res):
        allres.setdefault((x, y), []).append(r)
    for (x, y), rs in allres.items():
        r = rs[0] if len(rs) == 1 else min(rs, key=lambda q: (q["success"], q["clean"]))   # show the worse rep
        box = np.array(r["box"][:2])
        if max(abs(x / 100 - box[0]), abs(C.CUBE_Y + y / 100 - box[1])) < 0.075:
            c = "box"
        elif r["success"]:
            c = "OK" if r["clean"] else "ok"
        elif "reach" in r["push_why"]:
            c = "R"
        elif r["lifted"]:
            c = "L"
        else:
            c = "."
        cell[(x, y)] = c
        if c in (".", "L"):
            print(f"  fail ({x:+.0f},{y:+.0f}): ended {r['end']}, push ticks {r['push_ticks']}, lifted {r['lifted']}, "
                  f"clean {r['clean']} {r['push_why']}", flush=True)
    print("push+clamp map, cube start (cm from the nominal spot, + = away from the arms); OK clean success, ok success "
          "with contact, L lifted but not in box, . failed, R push out of reach, box = under the box", flush=True)
    print("   y \\ x " + "".join(f"{x:>6.0f}" for x in g))
    for y in gy[::-1]:
        print(f"{y:+7.0f}  " + "".join(f"{cell[(x, y)]:>6}" for x in g))
    eps = [q for (x, y), rs in allres.items() if cell[(x, y)] != "box" for q in rs]
    pushed = [q for q in eps if q["needed_push"]]
    landed = [q for q in pushed if q["landed"]]
    direct = [q for q in eps if not q["needed_push"]]
    print(f"stages: pushes needed {len(pushed)}/{len(eps)}, push landed in the clamp zone {len(landed)}/{len(pushed)}; "
          f"clamp success after a good push {np.mean([q['success'] for q in landed]) if landed else float('nan'):.2f} "
          f"({len(landed)}), after a bad push {np.mean([q['success'] for q in pushed if not q['landed']]) if len(pushed) > len(landed) else float('nan'):.2f}, "
          f"no push needed {np.mean([q['success'] for q in direct]) if direct else float('nan'):.2f} ({len(direct)})", flush=True)
    print(f"episodes (all reps, box cells excluded): success {np.mean([q['success'] for q in eps]):.2f} clean "
          f"{np.mean([q['success'] and q['clean'] for q in eps]):.2f} over {len(eps)} episodes (cells show the worse rep)", flush=True)


if __name__ == "__main__":
    main()
