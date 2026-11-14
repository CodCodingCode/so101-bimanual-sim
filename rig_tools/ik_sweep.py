"""IK area sweep of ONE real SO-101 arm for calibration: the hand visits a 3-D grid of points over that arm's half of
the table (fingers pointing down), and at every point the three cameras + all 12 measured joints are logged in the
calib_fit layout, so calib_from_sweep.py (overhead silhouette fit) can measure the arm's joint offsets from it.

Runs ON dzkchen (needs the COM ports + cameras: stop monitor.py first; remote_run.py does that). Also runs on the Mac
for --plan (no hardware).

Targets: hand positions (gripperframe site) on a grid  x: 4 columns across the arm's half (>= --midline-cm from the
centre line), y: 4 rows from 12 to 30 cm in front of the bases, z: 16 / 10 / 5 cm above the table; solved with the
training sim's IK (sim_mac = the Mac sim: real servo ranges, 232 mm mount, camera-mount / post collision shapes).

Safety (safety.py, the same guard as real_clamp.py):
  - the WHOLE plan (every target and every interpolated step of every path, from the arm's measured start pose) is
    preflighted before torque goes on: each servo's real range, wrist-camera-mount / other-arm / camera-post
    clearances, and a hand-height floor (lowest point of the moving forearm/wrist/gripper >= --floor-cm);
  - every tick guard.step(measured, commanded) runs before anything is sent (limits, clearances, tracking, lead limit);
  - only the chosen arm gets torque; the other arm is opened read-only (never torqued, no register writes);
  - per-joint torque caps (safety.apply_torque_limits), --deg-s speed cap, C:\\robot\\real\\STOP or Ctrl+C;
  - any stop or error folds the arm back to rest, then torque off.
The table must be CLEAR (no cube, no box) — the guard is run without a box and the grid goes down to 5 cm.

    python ik_sweep.py --side left --plan     # grid + IK + preflight against the ranges file, no hardware
    python ik_sweep.py --side left --dry      # opens both arms read-only, preflights from the real pose, no torque
    python ik_sweep.py --side left --live     # moves the left arm
Output: runs/ik_<side>_<time>/run_01/{events.jsonl, meta.json, frames/cNNN_raw_{overhead,left_wrist,right_wrist}.jpg}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

WIN = Path(r"C:\robot\real")
if WIN.exists():
    HERE, SIM = WIN, WIN / "sim_mac"
    sys.path[:0] = [str(SIM / "squeeze"), str(SIM), str(WIN)]
    sys.meta_path[:] = [f for f in sys.meta_path if "__editable___pi_sim" not in getattr(f, "__module__", "")]
    os.chdir(WIN)
else:                                                         # Mac: plan only
    HERE = Path(__file__).resolve().parent
    SIM = HERE.parent / "pi_sim_so101"
    sys.path[:0] = [str(SIM / "squeeze"), str(SIM), str(HERE)]

import mujoco                      # noqa: E402
import numpy as np                 # noqa: E402

import safety as SF                # noqa: E402

SIDES = ("left", "right")
CAMS = ("overhead", "left_wrist", "right_wrist")
RAISED = np.array([-45.8, 45.9, 90.0])          # lift, elbow, wrist_flex of SIM_HOME: upright, hand high
SIM_HOME6 = np.array([0.0, -45.8, 45.9, 90.0, -2.8, 0.0])
# folded rest in runner units (both arms read 2026-11-12, torque off) — used only by --plan, which has no hardware
PLAN_REST = {"left": np.array([-0.5, -86.8, 90.2, -37.8, -72.5, 3.8]),
             "right": np.array([-2.4, -100.6, 89.1, -42.2, -90.1, 4.0])}
STOP = HERE / "STOP"
TICK = 0.04                                     # 25 Hz, as real_clamp
LOW_PARTS = ("lower_arm", "wrist", "gripper", "camera_mount", "moving_jaw")


# ----------------------------------------------------------------------------- sim side (IK, floor, preflight)
class Planner:
    def __init__(self, limits, floor_m, midline_m):
        import env as E
        E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
        self.env = E.SqueezeEnv()
        self.env.reset(seed=0)
        self.sim, self.m, self.d = self.env.sim, self.env.model, self.env.data
        self.d.joint("cube_joint").qpos[:3] = [2, 2, -1]
        self.d.joint("can_joint").qpos[:3] = [2.5, 2, -1]
        self.guard = SF.Guard(self.m, limits, box_xy=None, log=lambda *a: None)
        self.floor, self.mid = floor_m, midline_m
        self.base = {s: self.d.xpos[self.m.body(f"{s}_base").id].copy() for s in SIDES}

    def ik(self, side, xyz):
        """Low layer (z < 7 cm): the training sim's fingers-down IK. Higher layers: position-only IK (fingers down
        is out of the wrist's real range up there), which also varies the wrist pitch across the sweep."""
        arm = self.sim.robot[side]
        self.d.qpos[arm.qpos_ids] = np.deg2rad(SIM_HOME6[:5])            # same seed every time: elbow-up branch
        xyz = np.asarray(xyz, float)
        if xyz[2] < 0.07:
            robot_env = type(self.sim).__mro__[1]                       # pi_sim RobotEnv._solve_ik
            q, err = robot_env._solve_ik(self.sim, arm, xyz, np.array([1.0, 0, 0, 0]), downward=True)
            return np.r_[np.rad2deg(q), 0.0], err
        m, d = self.m, mujoco.MjData(self.m)
        d.qpos[:] = self.d.qpos
        lim, jp = m.jnt_range[arm.joint_ids], np.zeros((3, m.nv))
        for _ in range(150):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e = xyz - d.site_xpos[arm.site_id]
            if np.linalg.norm(e) < 2e-4:
                break
            mujoco.mj_jacSite(m, d, jp, None, arm.site_id)
            J = jp[:, arm.dof_ids]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
            d.qpos[arm.qpos_ids] = np.clip(d.qpos[arm.qpos_ids] + np.clip(dq, -0.15, 0.15), lim[:, 0], lim[:, 1])
        mujoco.mj_kinematics(m, d)
        return np.r_[np.rad2deg(d.qpos[arm.qpos_ids]), 0.0], float(np.linalg.norm(xyz - d.site_xpos[arm.site_id]))

    def tip(self, side, p6):
        """Sim fingertip position for joint pose p6 (the other arm's pose does not matter for it)."""
        self.guard._pose({side: p6, ("right" if side == "left" else "left"): PLAN_REST["right" if side == "left" else "left"]})
        return self.guard.d.site_xpos[self.m.site(f"{side}_gripperframe").id].copy()

    def lowest(self, side):
        """Lowest point of the moving forearm / wrist / gripper / camera mount (call after a _pose)."""
        return min(self.guard._world(p)[1][:, 2].min() for p in self.guard.arm[side] if any(k in p[1] for k in LOW_PARTS))

    def problems(self, side, p6, other6):
        q = {side: p6, ("right" if side == "left" else "left"): other6}
        bad = self.guard.check_limits(q) + self.guard.check_pose(q, "cmd")    # _pose() runs inside check_pose
        low = self.lowest(side)
        if low < self.floor:
            bad.append(f"hand {low * 100:.1f} cm above the table < floor {self.floor * 100:.1f} cm")
        x = self.guard.d.site_xpos[self.m.site(f"{side}_gripperframe").id][0]
        if (side == "left" and x > -self.mid) or (side == "right" and x < self.mid):
            bad.append(f"hand crosses the midline (x {x * 100:+.1f} cm)")
        return bad

    def segment_problems(self, side, a, b, other6, step_deg=2.0):
        n = max(2, int(np.ceil(np.abs(b[:5] - a[:5]).max() / step_deg)) + 1)
        for f in np.linspace(0, 1, n):
            bad = self.problems(side, a + (b - a) * f, other6)
            if bad:
                return bad
        return []

    def route(self, side, cur, tgt, other6):
        """Joint-space waypoints cur -> tgt: direct if every step is clean, else up -> turn -> down. None if neither."""
        if not self.segment_problems(side, cur, tgt, other6):
            return [tgt.copy()]
        up1 = cur.copy(); up1[1:4] = RAISED
        up2 = tgt.copy(); up2[1:4] = RAISED
        wps, a = [up1, up2, tgt.copy()], cur
        for b in wps:
            if self.segment_problems(side, a, b, other6):
                return None
            a = b
        return wps


def grid(side, a):
    sx = -1 if side == "left" else 1
    xs = sx * np.linspace(a.midline_cm / 100 + 0.04, 0.24, 4)            # 7 .. 24 cm from the centre line
    ys = np.linspace(-0.10, 0.08, 4)                                       # 12 .. 30 cm in front of the bases
    zs = (0.16, 0.10, 0.05)
    pts = []
    for zi, z in enumerate(zs):                                            # high layer first; snake order
        for yi, y in enumerate(ys if zi % 2 == 0 else ys[::-1]):
            row = xs if yi % 2 == 0 else xs[::-1]
            pts += [(float(x), float(y), z) for x in row]
    return pts


def make_plan(P, side, start6, other6, a, log=print):
    targets, skipped = [], []
    for xyz in grid(side, a):
        p6, err = P.ik(side, xyz)
        why = [f"IK misses by {err * 1000:.0f} mm"] if err > 0.004 else P.problems(side, p6, other6)
        (skipped if why else targets).append((xyz, p6, err, why))
    plan, cur = [], start6.copy()
    for xyz, p6, err, _ in targets:
        wps = P.route(side, cur, p6, other6)
        if wps is None:
            skipped.append((xyz, p6, err, ["no clean path from the previous pose"]))
            continue
        plan.append(dict(xyz=xyz, p6=p6, ik_err=err, wps=wps))
        cur = p6
    home = cur.copy(); home[1:4] = RAISED
    back_ok = not P.segment_problems(side, cur, home, other6)
    log(f"{side}: {len(plan)} of {len(grid(side, a))} grid points planned, {len(skipped)} skipped; "
        f"path back up {'clean' if back_ok else 'NOT clean'}")
    for xyz, _, _, why in skipped:
        log(f"  skip ({xyz[0] * 100:+.0f}, {xyz[1] * 100:+.0f}, {xyz[2] * 100:.0f}) cm: {why[0]}")
    return plan, back_ok


def describe(P, side, plan, a, log=print):
    secs = 0.0
    cur = None
    for k, t in enumerate(plan):
        for w in t["wps"]:
            if cur is not None:
                secs += max(1.2, np.abs(w[:5] - cur[:5]).max() / a.deg_s)
            cur = w
        secs += 1.5
        if k < 60:
            P.problems(side, t["p6"], PLAN_REST["right" if side == "left" else "left"])
            c = P.guard.closest.get("cmd", {})
            log(f"  {k:2d} ({t['xyz'][0] * 100:+5.1f},{t['xyz'][1] * 100:+5.1f},{t['xyz'][2] * 100:4.1f}) cm  joints "
                + " ".join(f"{v:6.1f}" for v in t["p6"][:5]) + f"  via {len(t['wps'])}  hand low {P.lowest(side) * 100:4.1f} cm"
                + f"  mount {c.get('camera-mount', 0):.0f} mm, post {c.get('arm-camera', 0):.0f} mm")
    log(f"about {secs / 60:.1f} min of motion")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", required=True, choices=SIDES)
    ap.add_argument("--plan", action="store_true", help="no hardware: rest pose + ranges from files")
    ap.add_argument("--dry", action="store_true", help="read the real arms (no torque), preflight, no motion")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--floor-cm", type=float, default=2.5)
    ap.add_argument("--midline-cm", type=float, default=3.0)
    ap.add_argument("--deg-s", type=float, default=25.0)
    a = ap.parse_args()
    side, other = a.side, ("right" if a.side == "left" else "left")
    si = SIDES.index(side) * 6

    if a.plan or not (a.dry or a.live):
        spec = json.loads((SIM / "pi_sim" / "assets" / "so101_real_ranges.json").read_text())
        limits = {s: tuple(np.array(v, float).T) for s, v in spec["ranges_deg"].items()}
        P = Planner(limits, a.floor_cm / 100, a.midline_cm / 100)
        plan, back_ok = make_plan(P, side, PLAN_REST[side], PLAN_REST[other], a)
        describe(P, side, plan, a)
        return 0

    import so101_real as R
    import rt_control as RT        # min-jerk ramps at 100 Hz (2026-11-13 motor rule)
    from PIL import Image
    rig = json.loads((HERE / "rig.json").read_text())
    floor_cm = max(a.floor_cm, float(rig.get("table_floor_cm", 0.0)) + 1.5)
    off = rig.get("joint_offsets_deg", {})
    arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in SIDES}           # reads only
    for s, arm in arms.items():
        if arm.fingerprint != R.FINGERPRINTS[s]:
            raise SystemExit(f"{rig['arms'][s]} is not the {s} arm: fix rig.json")
    arm = arms[side]

    def read_units():
        return {s: arms[s].to_units(arms[s].read_raw()).astype(float) for s in SIDES}

    try:
        P = Planner(SF.real_limits(arms), floor_cm / 100, a.midline_cm / 100)
    except SF.SafetyStop as e:
        raise SystemExit(f"SAFETY: {e}")
    u0 = read_units()
    start6, other6 = u0[side].copy(), u0[other].copy()
    start6[5] = 0.0
    print(f"start: {side} " + " ".join(f"{v:.1f}" for v in u0[side]) + f" | {other} (read-only) "
          + " ".join(f"{v:.1f}" for v in u0[other]) + f"; floor {floor_cm:.1f} cm", flush=True)
    plan, back_ok = make_plan(P, side, start6, other6, a)
    if not plan or not back_ok:
        raise SystemExit("PREFLIGHT FAILED: nothing to do safely. Nothing moved.")
    describe(P, side, plan, a)
    print(f"safety preflight OK: every step of {len(plan)} paths within real ranges, wrist camera mounts >= "
          f"{SF.MARGIN['cmd']['camera-mount'] * 1000:.0f} mm, post >= {SF.MARGIN['cmd']['arm-camera'] * 1000:.0f} mm, "
          f"hand >= {floor_cm:.1f} cm", flush=True)
    if not a.live:
        print("DRY: nothing moves", flush=True)
        for x in arms.values():
            x.close()
        return 0

    cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}
    run = HERE / "runs" / f"ik_{side}_{time.strftime('%Y%m%d_%H%M%S')}" / "run_01"
    (run / "frames").mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps({"rig": rig, "side": side, "source": "ik_sweep.py (robot-driven IK grid)",
                                               "floor_cm": floor_cm, "midline_cm": a.midline_cm, "poses": len(plan),
                                               "targets": [t["xyz"] for t in plan]}, indent=1))
    ev = open(run / "events.jsonl", "a", encoding="utf-8")
    print(f"logging to {run}", flush=True)
    time.sleep(1.5)                                                     # camera auto-exposure

    def tick(cmd6):
        if STOP.exists():
            raise KeyboardInterrupt("STOP file")
        meas = read_units()
        out = P.guard.step(meas, {side: np.asarray(cmd6, float), other: meas[other]})
        arm.goal(arm.to_raw(out[side]))
        return meas

    def ramp(target6):
        start = read_units()[side]
        start[5] = target6[5]
        secs = max(1.2, float(np.abs(target6[:5] - start[:5]).max()) / a.deg_s)
        n = int(secs / TICK)
        for k in range(1, n + 1):
            t0 = time.perf_counter()
            tick(start + (target6 - start) * RT.min_jerk(k / n))
            time.sleep(max(0.0, TICK - (time.perf_counter() - t0)))

    def settle(target6, timeout=3.0):
        t_start, still, prev = time.time(), time.time(), None
        while time.time() - t_start < timeout:
            meas = tick(target6)[side]
            if prev is not None and np.abs(meas[:5] - prev[:5]).max() > 0.3:
                still = time.time()
            prev = meas
            if time.time() - still >= 0.6:
                return
            time.sleep(TICK)

    def go_rest():
        """Unguarded fold, the same path every runner uses (map_sweep / real_clamp): straight up, then fold."""
        u = arm.to_units(arm.read_raw()).astype(float)
        up = u.copy(); up[1:4] = RAISED
        r0, r1 = arm.read_raw(), arm.to_raw(up)
        n = int(max(1.2, np.abs(up[:5] - u[:5]).max() / a.deg_s) / 0.01)
        for k in range(1, n + 1):
            arm.goal(np.round(r0 + (r1 - r0) * RT.min_jerk(k / n)).astype(int))
            time.sleep(0.01)
        rest = arm.neutral_raw()
        mid = arm.read_raw().copy()
        mid[0], mid[3], mid[4], mid[5] = rest[0], rest[3], rest[4], rest[5]
        for tgt in (mid, rest):
            r0 = arm.read_raw()
            for k in range(1, 251):
                arm.goal(np.round(r0 + (tgt - r0) * RT.min_jerk(k / 250)).astype(int))
                time.sleep(0.01)
        time.sleep(0.4)

    try:
        STOP.unlink(missing_ok=True)
        arm.prepare()
        arm.goal(arm.read_raw().astype(int))
        SF.apply_torque_limits({side: arm})
        arm.torque(True)
        print(f"torque ON ({side} arm only)", flush=True)
        t_run = time.time()
        for k, t in enumerate(plan):
            for w in t["wps"]:
                ramp(w)
            settle(t["p6"])
            time.sleep(0.3)
            meas = read_units()
            s12 = np.r_[meas["left"], meas["right"]]
            imgs = {n: cams[n].latest() for n in CAMS}
            for n in CAMS:
                Image.fromarray(imgs[n]).save(run / "frames" / f"c{k:03d}_raw_{n}.jpg", quality=92)
            tip_meas = P.tip(side, meas[side])
            ev.write(json.dumps({"event": "model_call", "chunk": k, "side": side, "t": round(time.time() - t_run, 2),
                                 "state_in": [round(float(x), 3) for x in s12],
                                 "target": [round(float(x), 2) for x in t["p6"]],
                                 "target_xyz": [round(v, 4) for v in t["xyz"]],
                                 "sim_tip_from_measured": [round(float(v), 4) for v in tip_meas]}) + "\n")
            ev.flush()
            err = np.abs(meas[side][:5] - t["p6"][:5]).max()
            off_mm = np.linalg.norm(tip_meas - np.array(t["xyz"])) * 1000
            print(f"pose {k:2d}/{len(plan) - 1}: joints within {err:4.1f} deg, sim hand {off_mm:4.0f} mm from target",
                  flush=True)
        print("sweep done, returning to rest", flush=True)
        go_rest()
    except BaseException as e:                                          # SafetyStop, STOP file, Ctrl+C, I/O errors
        print(f"stopping early: {e!r}; returning to rest", flush=True)
        try:
            go_rest()
        except Exception as e2:
            print(f"go_rest failed: {e2!r}", flush=True)
    finally:
        arm.torque(False)
        print("torque OFF", flush=True)
        for c in cams.values():
            c.stop.set()
        for x in arms.values():
            x.close()
    print(f"run saved: {run}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
