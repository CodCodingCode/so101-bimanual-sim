"""Scripted two-arm TIP CLAMP on the REAL SO-101 arms: clamp the cube with the fingertips (jaws shut), lift it level,
carry it to the grey box (between the arms and the cube), drop it in, back off apart. Runs ON dzkchen, monitor stopped.

Exactly the sim teacher (squeeze_rt/clamp_teacher.py, per-arm settings from squeeze_rt/runs/design_clamp2.json):
  - cube centre from the calibrated overhead camera (red blob -> table plane), box assumed at cube + BOX_DY
  - a kinematic shadow of the training sim (squeeze_rt/env.py SqueezeEnv) gives IK, workspace limits, rate limit
  - 25 Hz, the teacher's own speed cap (0.12 m/s); jaws commanded shut
    python real_clamp.py --dry     # find the cube, print the plan, NO motion
    python real_clamp.py --live    # run it
Safety (safety.py, added after a run knocked a wrist camera holder off): before anything moves, a kinematic PREFLIGHT
of the whole plan against each servo's real range and the arm / camera-mount / camera-post / box clearances (the dry
check runs it too); every tick the guard re-checks the commanded AND measured pose, watches tracking, and limits how
far a command may lead the joint; per-joint torque caps; HandHeight floor guard (--floor-cm); C:\\robot\\real\\STOP.
Any stop ALWAYS folds the arms back to rest before torque off. Logs to runs/real_clamp_<ts>/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(r"C:\robot\real")
SIM = HERE / "sim_mac"                  # the Mac training sim (collision shapes for camera mounts / post / grippers)
SQ = SIM / "squeeze"
sys.path[:0] = [str(SQ), str(SIM), str(HERE)]
sys.meta_path[:] = [f for f in sys.meta_path if "__editable___pi_sim" not in getattr(f, "__module__", "")]
os.chdir(HERE)

import mujoco                      # noqa: E402
import numpy as np                 # noqa: E402
from PIL import Image              # noqa: E402

import clamp_control as CC        # noqa: E402
import safety as SF                # noqa: E402
import rt_control as RT            # noqa: E402
import so101_real as R             # noqa: E402

STOP = HERE / "STOP"
SIDES = ("left", "right")
CAMS = ("overhead", "left_wrist", "right_wrist")


def camera_model():
    rig = json.loads(Path(r"C:\robot\pi\pi_sim\assets\so101_camera_rig.json").read_text())["cameras"]["overhead"]
    fx, fy, cx, cy = (rig["intrinsics"][k] for k in ("fx", "fy", "cx", "cy"))
    k1, k2 = rig["distortion"][:2]
    pos = np.array(rig["pose"]["pos"])
    w, x, y, z = rig["pose"]["quat_wxyz"]
    Rwc = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def backproject(u, v, zp):
        xd, yd = (u - cx) / fx, (v - cy) / fy
        xn, yn = xd, yd
        for _ in range(30):
            r2 = xn * xn + yn * yn
            dd = 1 + k1 * r2 + k2 * r2 * r2
            xn, yn = xd / dd, yd / dd
        ray = Rwc @ np.array([xn, -yn, -1.0])
        return pos + (zp - pos[2]) / ray[2] * ray

    return backproject


def find_cube(rgb, backproject, half):
    a = rgb.astype(int)
    m = (a[..., 0] > 120) & (a[..., 0] - a[..., 1] > 60) & (a[..., 0] - a[..., 2] > 40)
    if m.sum() < 30:
        return None, int(m.sum())
    vs, us = np.nonzero(m)
    return backproject(float(us.mean()), float(vs.mean()), half), int(m.sum())


def rollout(sim, m, d, plan, home_units):
    """Kinematic preview of the whole plan from the start pose (arms assumed to track): commanded joints per tick."""
    saved = (d.qpos.copy(), d.ctrl.copy(), sim._ee_targets.copy())
    for i, s in enumerate(SIDES):
        arm = sim.robot[s]
        d.qpos[arm.qpos_ids] = np.deg2rad(home_units[i * 6:i * 6 + 5])
    mujoco.mj_forward(m, d)
    for i, s in enumerate(SIDES):
        arm = sim.robot[s]
        sim._ee_targets[i] = arm.position.copy()
        d.ctrl[arm.actuator_ids] = d.qpos[arm.qpos_ids]
    traj = []
    while not plan.done:
        tg = plan.next_targets(sim._ee_targets.copy())
        q = {}
        for i, s in enumerate(SIDES):
            arm = sim.robot[s]
            sim._ee_targets[i] = np.clip(tg[i], sim._workspace_low[i], sim._workspace_high[i])
            target, _ = sim._solve_ik(arm, sim._ee_targets[i], None)
            cur = d.ctrl[arm.actuator_ids].copy()
            new = cur + np.clip(target - cur, -sim._rate_limit, sim._rate_limit)
            d.ctrl[arm.actuator_ids] = new
            d.qpos[arm.qpos_ids] = new
            q[s] = np.r_[np.rad2deg(new), 0.0]
        mujoco.mj_forward(m, d)
        traj.append((plan.phase, q))
    d.qpos[:], d.ctrl[:], sim._ee_targets[:] = saved
    mujoco.mj_forward(m, d)
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--floor-cm", type=float, default=-0.5)
    ap.add_argument("--grab-only", action="store_true",
                    help="grab, lift 4 cm, hold, put the cube back where it was, release, retreat (no carry / box)")
    ap.add_argument("--speed", type=float, default=1.0, help="fraction of the designed speed (0.5 = half)")
    ap.add_argument("--no-stiff", action="store_true", help="keep the servos' soft position gain (P 16) for the run")
    ap.add_argument("--no-feedback", action="store_true", help="replay the sim preview without joint feedback")
    a = ap.parse_args()
    live = a.live and not a.dry
    speed = float(np.clip(a.speed, 0.2, 1.0))

    import clamp_teacher as C
    C.MAX_STEP *= speed
    import env as E
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    shadow = E.SqueezeEnv()
    shadow.reset(seed=0)
    sim, m, d = shadow.sim, shadow.model, shadow.data
    d.joint("cube_joint").qpos[:3] = [2, 2, -1]
    d.joint("can_joint").qpos[:3] = [2.5, 2, -1]
    glo, ghi = m.actuator_ctrlrange[sim.robot["left"].gripper_actuator]
    hh = R.HandHeight()
    backproject = camera_model()

    rig = json.loads((HERE / "rig.json").read_text())
    off = rig.get("joint_offsets_deg", {})
    arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in SIDES}
    for s, arm in arms.items():
        if arm.fingerprint != R.FINGERPRINTS[s]:
            raise SystemExit(f"{rig['arms'][s]} is not the {s} arm")
    cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}
    run = HERE / "runs" / f"real_clamp_{time.strftime('%Y%m%d_%H%M%S')}"
    (run / "frames").mkdir(parents=True, exist_ok=True)

    time.sleep(2.0)                                                # let auto-exposure settle
    cube, npx = None, 0
    for _ in range(15):
        cube, npx = find_cube(cams["overhead"].latest(), backproject, C.CUBE_HALF)
        if cube is not None:
            break
        time.sleep(0.2)
    Image.fromarray(cams["overhead"].latest()).save(run / "detect_overhead.jpg", quality=90)
    if cube is None:
        raise SystemExit(f"cube not found in the overhead camera ({npx} red px)")
    cx, cy = float(cube[0]), float(cube[1])
    bx, by = cx, cy + C.BOX_DY - 0.005
    dev = (cx * 100, (cy - C.CUBE_Y) * 100)
    print(f"cube at x {cx * 100:+.1f} cm, y {cy * 100:+.1f} cm ({(cy + 0.22) * 100:.1f} cm in front of the bases); "
          f"offset from nominal {dev[0]:+.1f}, {dev[1]:+.1f} cm (allowed +-1.5); box assumed at y {by * 100:+.1f} cm", flush=True)
    if abs(dev[0]) > 1.5 or abs(dev[1]) > 1.5:          # clamp v3 is verified clean within +-1.5 cm (sim, 24/24)
        raise SystemExit("cube outside the trained area: move it into the green zone")
    for s in SIDES:
        sim.pad_quat[s] = C.clamp_quat(s, (cx, cy))
    def make_plan():
        p = C.ClampPlan((cx, cy, 0.0), None if a.grab_only else (bx, by))
        p.waypoints = [(n, l, r, int(round(k / speed))) for n, l, r, k in p.waypoints]
        return p

    plan = make_plan()
    print(("GRAB TEST (puts the cube back) " if a.grab_only else "") + f"at {speed:.0%} speed, "
          f"{sum(w[3] for w in plan.waypoints) / 25:.0f} s: " + " -> ".join(f"{n}({k})" for n, _, _, k in plan.waypoints), flush=True)
    # ---- safety: guard on the training sim's collision shapes + each servo's real range; preflight the whole plan
    try:
        guard = SF.Guard(m, SF.real_limits(arms), box_xy=(bx, by))
    except SF.SafetyStop as e:
        for c in cams.values():
            c.stop.set()
        raise SystemExit(f"SAFETY: {e}")
    lim = guard.limits
    print("real joint ranges (deg): " + " | ".join(f"{s} " + " ".join(f"{lo:.0f}..{hi:.0f}" for lo, hi in zip(*lim[s]))
                                                   for s in SIDES), flush=True)
    home_units = R.SIM_HOME.copy()
    home_units[5] = home_units[11] = 0.0
    traj = rollout(sim, m, d, make_plan(), home_units)
    bad = guard.preflight([q for _, q in traj])
    if bad is not None:
        i, why = bad
        Image.fromarray(cams["overhead"].latest()).save(run / "start_overhead.jpg", quality=90)
        for c in cams.values():
            c.stop.set()
        raise SystemExit(f"SAFETY PREFLIGHT FAILED at t={i / 25:.2f}s ({traj[i][0]}): {'; '.join(why)}. Nothing moved.")
    print(f"safety preflight OK: {len(traj)} ticks within real joint ranges and clearances "
          f"(wrist camera mounts >= {SF.MARGIN['cmd']['camera-mount'] * 1000:.0f} mm, arm-arm >= {SF.MARGIN['cmd']['arm-arm'] * 1000:.1f} mm, camera post >= {SF.MARGIN['cmd']['arm-camera'] * 1000:.0f} mm, "
          f"box >= {SF.MARGIN['cmd']['arm-box'] * 1000:.0f} mm)", flush=True)
    # ---- SIM PREVIEW: the whole plan with the calibrated physics from the real start pose; its joint trajectory is
    # what the real arms replay (clamp_control.py). Refuse if the simulated grab does not lift cleanly.
    import design_squeeze as DS
    t_pv = time.perf_counter()
    pv_env = E.SqueezeEnv(episode_seconds=60.0)
    pv_env.reset(seed=0)
    ref = CC.preview(pv_env, C, make_plan(), (cx, cy), None if a.grab_only else (bx, by), home_units, DS.bad_contacts)
    pv_env.close()
    print(f"sim preview ({time.perf_counter() - t_pv:.1f} s): lifted {ref['lifted_mm']:.0f} mm, contacts {ref['contacts']}, "
          f"{len(ref['ctrl'])} ticks", flush=True)
    bad = guard.preflight([{s: np.r_[np.rad2deg(c[s]), 0.0] for s in SIDES} for c in ref["ctrl"]])
    in_box = bool(np.linalg.norm(ref["cube_end"][:2] - np.array([bx, by])) < 0.03 and ref["cube_end"][2] < 0.03)
    if not a.grab_only:
        print(f"sim preview: cube ends {'IN' if in_box else 'NOT in'} the box", flush=True)
    if ref["lifted_mm"] < 25 or ref["contacts"] > 0 or bad is not None or (not a.grab_only and not in_box):
        for c in cams.values():
            c.stop.set()
        why = "; ".join(bad[1]) if bad is not None else ("the simulated grab did not lift cleanly" if ref["lifted_mm"] < 25 or ref["contacts"] > 0 else "the simulated cube did not end in the box")
        raise SystemExit(f"SIM PREVIEW FAILED: {why}. Nothing moved.")
    ctl = CC.ReplayController(ref, feedback=not a.no_feedback)
    (run / "meta.json").write_text(json.dumps({"cube": [cx, cy], "box": [bx, by], "rig": rig, "live": live,
                                               "params": C.P, "stiff": not a.no_stiff, "feedback": not a.no_feedback,
                                               "preview": {"lifted_mm": ref["lifted_mm"], "contacts": ref["contacts"]}},
                                              indent=1))
    Image.fromarray(cams["overhead"].latest()).save(run / "start_overhead.jpg", quality=90)
    if not live:
        print("DRY: nothing moves", flush=True)
        for c in cams.values():
            c.stop.set()
        return

    def read_units():
        return {s: arms[s].to_units(arms[s].read_raw()).astype(float) for s in SIDES}

    def to_shadow(u):
        for s in SIDES:
            arm = sim.robot[s]
            d.qpos[arm.qpos_ids] = np.deg2rad(u[s][:5])
            d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = glo + np.clip(u[s][5], 0, 100) / 100 * (ghi - glo)
        mujoco.mj_forward(m, d)

    cfg = RT.load_config()
    fold_log = []

    def ramp_raw(targets, secs, check_stop=True, log=None):
        """Scripted move: minimum-jerk S-curve at the stream rate (rt_control.smooth_ramp)."""
        return RT.smooth_ramp(arms, targets, cfg, secs, check_stop=STOP.exists if check_stop else None, log=log)

    log, stream, saver = [], None, None
    try:
        STOP.unlink(missing_ok=True)
        for arm in arms.values():
            arm.prepare()
            arm.goal(arm.read_raw().astype(int))
        SF.apply_torque_limits(arms)
        if not a.no_stiff:
            print(f"stiffer servos for this run: P {CC.P_SOFT} -> {CC.P_STIFF}, read back "
                  f"{CC.set_p_gain(arms, CC.P_STIFF)}", flush=True)
        for arm in arms.values():
            arm.torque(True)
        home = R.SIM_HOME.copy()
        home[5] = home[11] = 0.0                                   # jaws shut (the clamp never uses them)
        hraw = {s: arms[s].to_raw(home[i * 6:(i + 1) * 6]) for i, s in enumerate(SIDES)}
        stage1 = {}
        for s in SIDES:
            mid = arms[s].read_raw().copy()
            mid[1], mid[2] = hraw[s][1], hraw[s][2]
            stage1[s] = np.round(mid).astype(int)
        ramp_raw(stage1, 3.0)
        ramp_raw(hraw, 2.5)
        time.sleep(0.6)
        print("at start pose, replaying the sim preview" + ("" if a.no_feedback else " with joint feedback"), flush=True)
        # real-time replay (rt_control): the preview's tick index follows the wall clock (elapsed / 0.04 s); ticks a slow
        # loop skipped still run through the controller (its joint feedback integrates per preview tick); the newest
        # command is streamed to the servos at 100 Hz
        clock = RT.TickClock(cfg)
        saver = RT.FrameSaver(run / "frames", cfg)
        stream = RT.ServoStreamer(arms, cfg)
        stream.start(hraw)
        nominal, n_ref = cfg["control"]["nominal_tick_s"], len(ref["ctrl"])
        last_phase, t0, kdone = None, time.perf_counter(), 0
        while kdone < n_ref:
            if STOP.exists():
                raise KeyboardInterrupt("STOP file")
            tick = time.perf_counter()
            clock.tick()
            u = read_units()
            meas = {s: np.deg2rad(u[s][:5]) for s in SIDES}
            k_target = min(n_ref, max(kdone + 1, int((tick - t0) / nominal) + 1))
            while kdone < k_target:
                cmd, expect, info = ctl.tick(kdone, meas)
                kdone += 1
            k = kdone - 1
            phase = info["phase"]
            if phase != last_phase:
                print(f"  t={k / 25:5.2f}s {phase}   joint correction (deg) L {info['joint_corr_deg']['left']} "
                      f"R {info['joint_corr_deg']['right']}", flush=True)
                last_phase = phase
            cmd = guard.step(u, cmd, expect=expect)   # raises SafetyStop -> fold to rest; returns the lead-limited command
            low = hh.lowest(np.r_[cmd["left"], cmd["right"]])
            blocked = [s for s in SIDES if low[s] * 100 < a.floor_cm]
            for s in SIDES:
                if s in blocked:
                    stream.hold(s)
                else:
                    stream.set(s, arms[s].to_raw(cmd[s]), clock.period())
            if blocked:
                print(f"  floor guard held {blocked} (lowest {min(low.values()) * 100:.1f} cm)", flush=True)
            log.append({"t": round(time.perf_counter() - t0, 3), "k": k, "phase": phase, "units": {s: u[s].round(2).tolist() for s in SIDES},
                        "cmd": {s: cmd[s].round(2).tolist() for s in SIDES}, "expect": {s: expect[s].round(2).tolist() for s in SIDES},
                        "joint_corr_deg": info["joint_corr_deg"], "joint_err_deg": info["joint_err_deg"],
                        "closest_mm": {w: dict(v) for w, v in guard.closest.items()}})
            saver.maybe(time.perf_counter() - t0, {c: cams[c].latest() for c in CAMS}, f"t{len(log) - 1:04d}_{phase}.jpg")
            clock.sleep_rest(tick)
        time.sleep(0.3)                                       # let the streamer reach the last command
        Image.fromarray(cams["overhead"].latest()).save(run / "end_overhead.jpg", quality=90)
        dts = np.diff([r["t"] for r in log]) * 1000 if len(log) > 2 else np.array([0.0])
        print(f"clamp sequence finished: {len(log)} loop ticks for {n_ref} preview ticks, period median "
              f"{np.median(dts):.0f} ms / p90 {np.percentile(dts, 90):.0f} ms", flush=True)
    except SF.SafetyStop as e:
        print(f"SAFETY STOP: {e}", flush=True)
    except BaseException as e:
        print(f"stopping: {e!r}", flush=True)
    finally:
        for w in (stream, saver):                             # the streamer MUST stop before anything else writes goals
            if w is not None:
                try:
                    w.stop()
                except Exception as e:
                    print(f"stopping {type(w).__name__} failed: {e!r}", flush=True)
        (run / "log.json").write_text(json.dumps(log))
        print(f"log: {run} ({len(log)} ticks)", flush=True)
        STOP.unlink(missing_ok=True)
        print("returning to rest (raise, wrist first, then fold)", flush=True)
        try:
            if not a.no_stiff:
                try:
                    print(f"softer servos for the fold: P {cfg['ramp']['fold_p']}, read back "
                          f"{CC.set_p_gain(arms, cfg['ramp']['fold_p'])}", flush=True)
                except Exception as e:
                    print(f"could not lower P before the fold ({e!r})", flush=True)
            back = {}
            for s in SIDES:
                uu = arms[s].to_units(arms[s].read_raw())
                back[s] = arms[s].to_raw(np.r_[uu[0], -45.8, 45.9, 90.0, uu[4], 0.0])
            ramp_raw(back, 3.0, check_stop=False, log=fold_log)
            rest = {s: arms[s].neutral_raw() for s in SIDES}
            mid = {}
            for s in SIDES:
                r = arms[s].read_raw().copy()
                r[0], r[3], r[4], r[5] = rest[s][0], rest[s][3], rest[s][4], rest[s][5]
                mid[s] = np.round(r).astype(int)
            ramp_raw(mid, 2.5, check_stop=False, log=fold_log)
            ramp_raw(rest, 3.0, check_stop=False, log=fold_log)
            time.sleep(0.4)
            (run / "fold.json").write_text(json.dumps(fold_log))
        except Exception as e:
            print(f"return to rest failed: {e!r}", flush=True)
        finally:
            if not a.no_stiff:
                try:
                    print(f"servo position gain back to P {CC.P_SOFT}: read back {CC.set_p_gain(arms, CC.P_SOFT)}", flush=True)
                except Exception as e:
                    print(f"restoring P gain failed ({e!r}); it resets to {CC.P_SOFT} at the next power-off", flush=True)
            for arm in arms.values():
                arm.torque(False)
            print("rest, torque OFF", flush=True)
            for c in cams.values():
                c.stop.set()


if __name__ == "__main__":
    main()
