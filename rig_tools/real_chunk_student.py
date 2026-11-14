"""sim-74's ACTION-CHUNKED clamp student on the REAL SO-101 arms, through the same safety stack as real_policy.py.

Policy: squeeze_rt/chunk_rotation/student.pt (DAgger-distilled from the residual-PPO twist-tolerant clamp; sim: 86 %
delivered / 69-75 % clean, cube +-1.5 cm, ANY yaw, +-1 deg servo spread, weak-grip randomisation). It outputs 8-tick
chunks of tool-point targets, executed with ACT temporal ensembling (squeeze/chunk_student_real.StudentRunner).
Control: rig_tools/student_ctl.StudentCtl (the same class the sim test ran): triangulated cube, overhead cube yaw
(vision.detect_cube_pose) and box (vision.detect_box), the grab-attempt clock, the twist-offset design's gripper
orientation, IK + rate limit + squeeze limiter.

Protection as in every real runner: safety guard every tick (camera mounts / other arm / post / box / real joint ranges /
stuck joint / 10 deg lead limit) in the commanded AND measured pose, per-joint torque caps, servo gain P 32 for the run
only (back to 16), floor guard, STOP file. Any stop folds the arms back to rest, torque off. Trained speed.

    python real_chunk_student.py --dry       # find box + cube + its yaw, run the controller once: NOTHING moves
    python real_chunk_student.py --dry-loop  # the whole live loop for --seconds with torque OFF and no goal writes:
                                             # measures the real tick rate / stage timings. NOTHING moves
    python real_chunk_student.py --live

Real-time (2026-11-13, rt_control.py + so101_control.json; the first live run shook at a ~13 Hz loop):
  * servo goals are streamed at 100 Hz on a thread, sliding linearly to each new command over the measured tick
    period, instead of one snap per policy tick
  * cameras -> red blobs + overhead cube pose run on their own thread; the loop uses the newest result, never waits
  * run frames are saved on a thread every 0.2 s
  * the attempt clock, chunk ensembling, IK rate limit and step limit advance by elapsed / 0.04 s, not by 1 per tick
  * cube filters: height re-read from the overhead on the table plane unless the tips are closed on it; yaw filtered
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(r"C:\robot\real")
SIM = HERE / "sim_mac"
SQ = SIM / "squeeze"
sys.path[:0] = [str(SQ), str(SIM), str(HERE)]
sys.meta_path[:] = [f for f in sys.meta_path if "__editable___pi_sim" not in getattr(f, "__module__", "")]
os.chdir(HERE)
os.environ["SQUEEZE_TASK"] = "clamp"
os.environ["CLAMP_DESIGN"] = "design_clamp3c_twistoff.json"       # before clamp_teacher is imported

import mujoco                      # noqa: E402
import numpy as np                 # noqa: E402
import torch                       # noqa: E402
from PIL import Image              # noqa: E402

import clamp_control as CC        # noqa: E402  (servo gain)
import safety as SF                # noqa: E402
import so101_real as R             # noqa: E402
import cube_locator as CL          # noqa: E402
import student_ctl as SC           # noqa: E402
import rt_control as RT            # noqa: E402

STOP = HERE / "STOP"
SIDES = SC.SIDES
CAMS = SC.CAMS
STEPS = 600                        # 24 s at 25 Hz (the student's episodes are 18 s; margin for the drop)
STUDENT = HERE / "squeeze_rt" / "chunk_rotation" / "student.pt"
CUBE_NOMINAL_RANGE_CM = 1.5        # trained: cube within +-1.5 cm of the nominal spot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--dry-loop", action="store_true", help="full loop timing test, torque off, nothing written")
    ap.add_argument("--floor-cm", type=float, default=-0.5)
    ap.add_argument("--seconds", type=float, default=STEPS / 25)
    ap.add_argument("--no-limiter", action="store_true", help="no squeeze limiter (the guard still runs)")
    a = ap.parse_args()
    live = a.live and not a.dry and not a.dry_loop
    cfg = RT.load_config()
    HZ = cfg["control"]["hz"]
    print(f"control config {cfg['_path']}: policy {HZ} Hz, servo stream {cfg['servo_stream']['hz']} Hz, "
          f"P {cfg['servo_gain']['run_p']} during the run", flush=True)
    torch.set_num_threads(2)

    import clamp_teacher as C
    import env as E
    import vision as V
    from chunk_student_real import StudentRunner
    if C._D.name != "design_clamp3c_twistoff.json":
        raise SystemExit(f"wrong clamp design loaded: {C._D}")
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    rig_cams = Path(r"C:\robot\pi\pi_sim\assets\so101_camera_rig.json")
    base_sim = E._Sim

    class _RigSim(base_sim):
        def __init__(self, *a_, **k_):
            k_.setdefault("camera_rig", str(rig_cams))
            super().__init__(*a_, **k_)
    E._Sim = _RigSim
    shadow = E.SqueezeEnv()
    shadow.reset(seed=0)
    sim, m, d = shadow.sim, shadow.model, shadow.data
    d.joint("cube_joint").qpos[:3] = [2, 2, -1]
    d.joint("can_joint").qpos[:3] = [2.5, 2, -1]
    glo, ghi = m.actuator_ctrlrange[sim.robot["left"].gripper_actuator]
    hh = R.HandHeight()
    runner = StudentRunner(STUDENT, delta_scale=sim.delta_scale)

    rig = json.loads((HERE / "rig.json").read_text())
    off = rig.get("joint_offsets_deg", {})
    arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in SIDES}
    for s, arm in arms.items():
        if arm.fingerprint != R.FINGERPRINTS[s]:
            raise SystemExit(f"{rig['arms'][s]} is not the {s} arm")
    cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}
    run = HERE / "runs" / f"real_chunk_student_{time.strftime('%Y%m%d_%H%M%S')}"
    (run / "frames").mkdir(parents=True, exist_ok=True)

    def stop_cams():
        for c in cams.values():
            c.stop.set()

    def read_units():
        return {s: arms[s].to_units(arms[s].read_raw()).astype(float) for s in SIDES}

    def to_shadow(u):
        for s in SIDES:
            arm = sim.robot[s]
            d.qpos[arm.qpos_ids] = np.deg2rad(u[s][:5])
            d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = glo + np.clip(u[s][5], 0, 100) / 100 * (ghi - glo)
        mujoco.mj_forward(m, d)

    def frames():
        return {c: cams[c].latest() for c in CAMS}

    time.sleep(2.0)                                                # let auto-exposure settle
    to_shadow(read_units())
    for i, s in enumerate(SIDES):
        sim._ee_targets[i] = sim.robot[s].position.copy()
        d.ctrl[sim.robot[s].actuator_ids] = d.qpos[sim.robot[s].qpos_ids]
    guard0 = SF.Guard(m, SF.real_limits(arms), box_xy=None)
    ctl = SC.StudentCtl(m, d, sim, C, V, CL, runner, guard0, squeeze_limit=not a.no_limiter,
                        filters=cfg["perception"])
    err = None
    for _ in range(15):
        try:
            box, cube, yaw = ctl.start(frames())
            err = None
            break
        except RuntimeError as e:
            err = e
            time.sleep(0.2)
    Image.fromarray(cams["overhead"].latest()).save(run / "detect_overhead.jpg", quality=90)
    if err is not None:
        stop_cams()
        raise SystemExit(f"cannot start: {err}")
    dev = (cube[0] * 100, (cube[1] - C.CUBE_Y) * 100)
    print(f"box at ({box[0] * 100:+.1f}, {box[1] * 100:+.1f}) cm, yaw {np.degrees(box[2]):+.1f} deg (fit IoU {box[3]:.2f})",
          flush=True)
    print(f"cube at ({cube[0] * 100:+.1f}, {cube[1] * 100:+.1f}, {cube[2] * 100:.1f}) cm, yaw {np.degrees(yaw):+.1f} deg; "
          f"offset from nominal {dev[0]:+.1f}, {dev[1]:+.1f} cm (trained +-{CUBE_NOMINAL_RANGE_CM}); {ctl.info}", flush=True)
    if abs(dev[0]) > CUBE_NOMINAL_RANGE_CM or abs(dev[1]) > CUBE_NOMINAL_RANGE_CM:
        stop_cams()
        raise SystemExit("cube outside the trained area: move it into the green zone")
    # trained layout (clamp_wide stage 4): box = cube + (+-0.5 cm, BOX_DY - 0.5 cm - 0..1 cm) = dy -11.1..-12.1 cm
    box_dy = box[1] - cube[1]
    if not -0.127 <= box_dy <= -0.105 or abs(box[0] - cube[0]) > 0.011:
        stop_cams()
        raise SystemExit(f"box not at the trained spacing from the cube (dx {100 * (box[0] - cube[0]):+.1f}, dy "
                         f"{100 * box_dy:+.1f} cm; trained dx -0.5..+0.5, dy -11.1..-12.1 cm)")
    try:
        guard = SF.Guard(m, SF.real_limits(arms), box_xy=(box[0], box[1]), box_yaw=box[2])
    except SF.SafetyStop as e:
        stop_cams()
        raise SystemExit(f"SAFETY: {e}")
    ctl.guard = guard
    print("safety guard ready (self-test OK)", flush=True)
    (run / "meta.json").write_text(json.dumps({"cube": cube.tolist(), "cube_yaw": yaw, "box": list(box), "live": live,
                                               "policy": "sim-74 chunked rotation student", "limiter": not a.no_limiter},
                                              indent=1))

    if a.dry:
        t0 = time.perf_counter()
        f = frames()
        u = read_units()
        to_shadow(u)
        ctrl = {s: d.ctrl[sim.robot[s].actuator_ids].copy() for s in SIDES}
        cmd, log = ctl.tick(f, ctrl)
        ms = (time.perf_counter() - t0) * 1000
        print(f"one control tick (cameras + detectors + student + IK + limiter) took {ms:.0f} ms (budget 40 ms); cube "
              f"{log['cube_xyz']} seen={log['cube_seen']} yaw {log['cube_yaw_deg']} deg; first target (arms where they are "
              f"now, NOT the start pose) {log['target']}", flush=True)
        print("DRY: nothing moved", flush=True)
        stop_cams()
        return

    fold_log = []

    def ramp_raw(targets, secs, check_stop=True, log=None):
        """Scripted move: minimum-jerk S-curve at the stream rate (rt_control.smooth_ramp), not a 50 Hz linear ramp."""
        return RT.smooth_ramp(arms, targets, cfg, secs, check_stop=STOP.exists if check_stop else None, log=log)

    def perceive():                                   # runs on the perception thread (no MuJoCo state touched)
        f = frames()
        return {"frames": f, "red": {c: CL.detect_red(f[c]) for c in CAMS},
                "pose": V.detect_cube_pose(f["overhead"], ctl.cam)}

    log, stream, perc, saver = [], None, None, None
    P_RUN, P_REST = cfg["servo_gain"]["run_p"], cfg["servo_gain"]["rest_p"]
    try:
        STOP.unlink(missing_ok=True)
        if live:
            for arm in arms.values():
                arm.prepare()
                arm.goal(arm.read_raw().astype(int))
            SF.apply_torque_limits(arms)
            print(f"stiffer servos for this run: P {P_REST} -> {P_RUN}, read back {CC.set_p_gain(arms, P_RUN)}", flush=True)
            for arm in arms.values():
                arm.torque(True)
            home = R.SIM_HOME.copy()
            home[5] = home[11] = 0.0
            hraw = {s: arms[s].to_raw(home[i * 6:(i + 1) * 6]) for i, s in enumerate(SIDES)}
            stage1 = {}
            for s in SIDES:
                mid = arms[s].read_raw().copy()
                mid[1], mid[2] = hraw[s][1], hraw[s][2]
                stage1[s] = np.round(mid).astype(int)
            ramp_raw(stage1, 3.0)
            ramp_raw(hraw, 2.5)
            time.sleep(0.6)
            last_goal = hraw
        else:
            print("DRY LOOP: torque stays OFF, no goal is written; timing only", flush=True)
            last_goal = {s: arms[s].read_raw().astype(int) for s in SIDES}
        u = read_units()
        to_shadow(u)
        for i, s in enumerate(SIDES):
            arm = sim.robot[s]
            sim._ee_targets[i] = arm.position.copy()
            d.ctrl[arm.actuator_ids] = d.qpos[arm.qpos_ids]
        perc = RT.Latest(perceive, name="perception")
        perc.wait_first()
        res, seq, _ = perc.get()
        ctl.start(res["frames"], dets=dict(res, seq=seq))   # fresh start at the start pose (re-reads cube + yaw, keeps the box)
        clock = RT.TickClock(cfg)
        saver = RT.FrameSaver(run / "frames", cfg)
        stream = RT.ServoStreamer(arms, cfg, dry=not live)
        stream.start(last_goal)
        print("at start pose: the chunked student is driving" if live else "dry loop running", flush=True)
        t0, k, dry_stops, next_print = time.perf_counter(), 0, 0, 0.0
        while time.perf_counter() - t0 < a.seconds:
            if STOP.exists():
                raise KeyboardInterrupt("STOP file")
            tick = time.perf_counter()
            adv = clock.tick()
            u = read_units()
            to_shadow(u)
            t_read = time.perf_counter()
            res, seq, t_det = perc.get()
            f = res["frames"]
            ctrl = {s: d.ctrl[sim.robot[s].actuator_ids].copy() for s in SIDES}
            cmd, info = ctl.tick(f, ctrl, dets=dict(res, seq=seq), adv=adv)
            t_ctl = time.perf_counter()
            expect = {s: cmd[s].copy() for s in SIDES}
            try:
                cmd = guard.step(u, cmd, expect=expect)        # raises SafetyStop -> fold to rest
            except SF.SafetyStop as e:
                if live:
                    raise
                dry_stops += 1                                 # dry loop: the arms do not follow, so the lead limit trips
                info["dry_guard"] = str(e)[:120]
            t_guard = time.perf_counter()
            low = hh.lowest(np.r_[cmd["left"], cmd["right"]])
            blocked = [s for s in SIDES if low[s] * 100 < a.floor_cm]
            for s in SIDES:
                if s in blocked:
                    stream.hold(s)
                else:
                    stream.set(s, arms[s].to_raw(cmd[s]), clock.period())
            if blocked:
                print(f"  floor guard held {blocked} (lowest {min(low.values()) * 100:.1f} cm)", flush=True)
            t = time.perf_counter() - t0
            saver.maybe(t, f, f"t{k:04d}.jpg")
            log.append({"t": round(t, 3), **info, "adv": round(adv, 2), "det_age_ms": round((tick - t_det) * 1000, 1),
                        "ms": {"read": round((t_read - tick) * 1000, 1), "ctl": round((t_ctl - t_read) * 1000, 1),
                               "guard": round((t_guard - t_ctl) * 1000, 1)},
                        "units": {s: u[s].round(2).tolist() for s in SIDES},
                        "cmd": {s: cmd[s].round(2).tolist() for s in SIDES}, "floor_blocked": blocked,
                        "closest_mm": {w: dict(v) for w, v in guard.closest.items()}})
            if t >= next_print:
                next_print += 1.0
                print(f"  t={t:4.1f}s  {1 / max(clock.period(), 1e-3):4.1f} Hz  attempt {info['attempt']} clock "
                      f"{info['clock']:.2f}  cube {np.round(np.array(info['cube_xyz']) * 100, 1).tolist()} cm "
                      f"seen={info['cube_seen']} yaw {info['cube_yaw_deg']:+.0f}  [{info['cube_how'][:60]}]"
                      f"{'  [squeeze limited]' if info['squeeze_limited'] else ''}", flush=True)
            k += 1
            clock.sleep_rest(tick)
        if live:
            Image.fromarray(cams["overhead"].latest()).save(run / "end_overhead.jpg", quality=90)
        dts = np.diff([r["t"] for r in log]) * 1000 if len(log) > 2 else np.array([0.0])
        st = {w: np.median([r["ms"][w] for r in log]) for w in ("read", "ctl", "guard")} if log else {}
        print(f"chunked student finished: {len(log)} ticks, period median {np.median(dts):.0f} ms / p90 "
              f"{np.percentile(dts, 90):.0f} ms ({100 * np.mean(dts > 44):.0f} % over 40 ms); stage medians "
              f"{ {w: round(v, 1) for w, v in st.items()} } ms; perception {np.median(perc.ms):.0f} ms/frame-set; "
              f"servo writes {stream.writes / max(1e-3, time.perf_counter() - t0):.0f}/s"
              f"{'; dry guard trips ' + str(dry_stops) if not live else ''}", flush=True)
    except SF.SafetyStop as e:
        print(f"SAFETY STOP: {e}", flush=True)
    except BaseException as e:
        print(f"stopping: {e!r}", flush=True)
    finally:
        for w in (stream, perc, saver):                       # the streamer MUST stop before anything else writes goals
            if w is not None:
                try:
                    w.stop()
                except Exception as e:
                    print(f"stopping {type(w).__name__} failed: {e!r}", flush=True)
        (run / "meta.json").write_text(json.dumps({**json.loads((run / "meta.json").read_text()), "live": live,
                                                   "dry_loop": not live, "control": {k: cfg[k] for k in
                                                   ("control", "servo_stream", "servo_gain")}}, indent=1))
        (run / "log.json").write_text(json.dumps(log))
        print(f"log: {run} ({len(log)} ticks)", flush=True)
        STOP.unlink(missing_ok=True)
        if not live:
            print("DRY LOOP: nothing moved", flush=True)
            stop_cams()
            return
        print("returning to rest (raise, wrist first, then fold)", flush=True)
        try:
            fold_p = cfg["ramp"]["fold_p"]
            try:
                print(f"softer servos for the fold: P {fold_p}, read back {CC.set_p_gain(arms, fold_p)}", flush=True)
            except Exception as e:
                print(f"could not lower P before the fold ({e!r}); folding at P {P_RUN}", flush=True)
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
            print(f"fold logged: {len(fold_log)} samples in fold.json", flush=True)
        except Exception as e:
            print(f"return to rest failed: {e!r}", flush=True)
        finally:
            try:
                print(f"servo position gain back to P {P_REST}: read back {CC.set_p_gain(arms, P_REST)}", flush=True)
            except Exception as e:
                print(f"restoring P gain failed ({e!r}); it resets to {P_REST} at the next power-off", flush=True)
            for arm in arms.values():
                arm.torque(False)
            print("rest, torque OFF", flush=True)
            stop_cams()


if __name__ == "__main__":
    main()
