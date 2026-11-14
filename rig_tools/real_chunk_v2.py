"""LIVE runner for the camera-detector CHUNKED clamp policy v2 (bb1b28 retrain) on the REAL SO-101 arms.

Policy: squeeze_rt/chunk_policy_v2_bb1b28/best.pt (squeeze/chunk_policy_v2.py: DAgger from the state-based grip_teacher,
labels = the teacher's own 8-step rollouts; sim, policy alone: 1.00 delivered / 0.94 clean). Trained world: ClampRLEnv,
design_clamp3, cube within +-1.5 cm of the nominal spot, yaw +-15 deg, box 11.4-12.4 cm behind the cube, grip
randomisation, recovery disturbances; cameras + joint offsets = calibration "cams 6349a0 + kinematics bb1b28".

Per tick (25 Hz, so101_control.json):
  sense     joints from the servos (+ bb1b28 joint offsets) -> kinematic shadow; the 3 cameras -> red blobs + overhead
            cube pose on a background thread (rt_control.Latest); the policy reads the cube from the OVERHEAD ray on the
            cube-half-height plane, as in training (chunk_v2_ctl)
  decide    chunk_v2_ctl.ChunkV2Ctl (= student_ctl.StudentCtl + this policy's input): 34-number input -> 8-step chunk ->
            ACT temporal ensembling -> step cap -> IK -> rate limit -> squeeze limiter (commanded grippers >= 1 mm apart)
            -> release gap over the box
  filter    cmd_filter.CmdFilter: never SEND a pose inside the guard's commanded-pose margins (other arm 1 mm, camera
            mounts 12 mm, camera post 15 mm, box 6 mm); send the farthest point toward it that passes
  guard     safety.Guard on the commanded AND measured pose every tick (camera mounts / other arm / post / box / real
            joint ranges / stuck joint / 10 deg lead limit), floor guard, STOP file. Any stop -> fold to rest, torque off
  act       servo goals streamed at 100 Hz on a thread, sliding to each new command over the measured tick period
            (rt_control.ServoStreamer); per-joint torque caps; servo P 32 for the run only (back to 16)

Before moving it refuses unless: box found by the overhead, cube found by the overhead and within the trained area,
cube yaw within the trained range, box at the trained spacing from the cube.

    python real_chunk_v2.py --dry        # box + cube + yaw + one control tick: NOTHING moves
    python real_chunk_v2.py --dry-loop   # the whole loop for --seconds, torque OFF, no goal written: timing. NOTHING moves
    python real_chunk_v2.py --live       # moves the arms
  options: --offsets new|current (joint offsets: bb1b28 = what the policy was trained with / rig.json's, for the A/B)
           --no-filter (guard still runs)  --no-limiter  --seconds S  --floor-cm C
           --arm-arm-guard (keep the measured 3 mm arm-arm stop; REMOVED by default for this runner, user 2026-11-13;
                            the commanded arm-arm check and all wrist-camera-mount margins always stay)
Log + frames: C:\\robot\\real\\runs\\real_chunk_v2_<time>.
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
os.environ["CLAMP_DESIGN"] = "design_clamp3.json"                 # before clamp_teacher is imported

import mujoco                      # noqa: E402
import numpy as np                 # noqa: E402
import torch                       # noqa: E402
from PIL import Image              # noqa: E402

import clamp_control as CC         # noqa: E402  (servo gain)
import safety as SF                # noqa: E402
import so101_real as R             # noqa: E402
import cube_locator as CL          # noqa: E402
import rt_control as RT            # noqa: E402
import chunk_v2_ctl as CV          # noqa: E402
from cmd_filter import CmdFilter   # noqa: E402

STOP = HERE / "STOP"
SIDES, CAMS = CV.SC.SIDES, CV.SC.CAMS
STEPS = 640                        # 25.6 s at 25 Hz (training episodes: up to 640 ticks)
POLICY = HERE / "squeeze_rt" / "chunk_policy_v2_bb1b28" / "best.pt"
CALIB = Path(r"C:\robot\vla\calib_bb1b28")         # cams 6349a0 + kinematics bb1b28 (camera rig, joint offsets)
CALIB_TAG = "cams 6349a0 + kinematics bb1b28"
CUBE_RANGE_CM = 1.5                # trained: cube within +-1.5 cm of the nominal spot (clamp_rl.CUBE_JITTER)
CUBE_YAW_DEG = 15.0 + 5.0          # trained +-15 deg; 5 deg of detector slack
BOX_DX_CM = 0.5 + 0.6              # trained box x = cube x +-0.5 cm; 0.6 cm of detector slack
BOX_DY_SLACK_CM = 0.6              # trained box y = cube y + BOX_DY - 0..1 cm


def drop_measured_arm_arm():
    """User, 2026-11-13: remove the guard's MEASURED arm-arm stop (3 mm between the two arms' solid links / jaws in the
    real pose) for this runner. In the sim test of this runner it stopped 4/12 episodes at 2.9-3.0 mm while squeezing,
    3 of which delivered. This process only: safety.py and every other runner keep it. Still on: the commanded arm-arm
    check (never COMMAND them < 0.5 mm apart; filter 1 mm; squeeze limiter 1 mm) and every wrist-camera-mount margin
    (the 3D-printed part that broke on 2026-11-11: other arm vs a camera mount, 12 mm commanded / 6 mm measured)."""
    SF.MARGIN["meas"].pop("arm-arm", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--dry-loop", action="store_true", help="full loop timing test, torque off, nothing written")
    ap.add_argument("--offsets", default="new", choices=("new", "current"))
    ap.add_argument("--no-filter", action="store_true", help="no command filter (the guard still runs)")
    ap.add_argument("--no-limiter", action="store_true", help="no squeeze limiter (the guard still runs)")
    ap.add_argument("--arm-arm-guard", action="store_true",
                    help="keep the guard's measured arm-arm stop (3 mm); default: removed for this runner (user)")
    ap.add_argument("--floor-cm", type=float, default=-0.5)
    ap.add_argument("--seconds", type=float, default=STEPS / 25)
    a = ap.parse_args()
    live = a.live and not a.dry and not a.dry_loop
    if not a.arm_arm_guard:
        drop_measured_arm_arm()
    print(f"guard margins (mm): commanded {({k: v * 1000 for k, v in SF.MARGIN['cmd'].items()})}, measured "
          f"{({k: v * 1000 for k, v in SF.MARGIN['meas'].items()})}"
          + ("" if a.arm_arm_guard else " -- measured arm-arm stop REMOVED (camera-mount margins kept)"), flush=True)
    cfg = RT.load_config()
    print(f"control config {cfg['_path']}: policy {cfg['control']['hz']} Hz, servo stream {cfg['servo_stream']['hz']} Hz, "
          f"P {cfg['servo_gain']['run_p']} during the run", flush=True)
    torch.set_num_threads(2)

    import clamp_teacher as C
    import env as E
    import vision as V
    if C._D.name != "design_clamp3.json":
        raise SystemExit(f"wrong clamp design loaded: {C._D}")
    # ---------------------------------------------------------------- calibration (the one the policy was trained on)
    rig_cams = CALIB / "so101_camera_rig.json"
    tag = json.loads(rig_cams.read_text()).get("calibration_tag")
    if tag != CALIB_TAG:
        raise SystemExit(f"camera rig {rig_cams} is calibration {tag!r}, the policy was trained on {CALIB_TAG!r}")
    rig = json.loads((HERE / "rig.json").read_text())
    off = rig.get("joint_offsets_deg", {}) if a.offsets == "current" else \
        json.loads((CALIB / "joint_offsets.json").read_text())["new"]
    print(f"calibration: camera rig {tag}; joint offsets {a.offsets}: L {off['left'][:5]} R {off['right'][:5]}", flush=True)

    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
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
    runner = CV.ChunkV2Runner(POLICY, delta_scale=sim.delta_scale)

    arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in SIDES}
    for s, arm in arms.items():
        if arm.fingerprint != R.FINGERPRINTS[s]:
            raise SystemExit(f"{rig['arms'][s]} is not the {s} arm")
    cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}
    run = HERE / "runs" / f"real_chunk_v2_{time.strftime('%Y%m%d_%H%M%S')}"
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

    # ---------------------------------------------------------------- box + cube, checked against the trained layout
    time.sleep(2.0)                                                # let auto-exposure settle
    to_shadow(read_units())
    for i, s in enumerate(SIDES):
        sim._ee_targets[i] = sim.robot[s].position.copy()
        d.ctrl[sim.robot[s].actuator_ids] = d.qpos[sim.robot[s].qpos_ids]
    guard0 = SF.Guard(m, SF.real_limits(arms), box_xy=None)
    ctl = CV.ChunkV2Ctl(m, d, sim, C, V, CL, runner, guard0, squeeze_limit=not a.no_limiter, filters=cfg["perception"])
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
        raise SystemExit(f"cannot start: {err} (camera indices in rig.json swapped? see {run / 'detect_overhead.jpg'})")
    det = runner.det_xy
    dev = (det[0] * 100, (det[1] - C.CUBE_Y) * 100)
    yaw90 = (np.degrees(yaw) + 45.0) % 90.0 - 45.0                 # the cube looks the same every 90 deg
    print(f"box at ({box[0] * 100:+.1f}, {box[1] * 100:+.1f}) cm, yaw {np.degrees(box[2]):+.1f} deg (fit IoU {box[3]:.2f})",
          flush=True)
    print(f"cube (overhead detector, what the policy sees) at ({det[0] * 100:+.1f}, {det[1] * 100:+.1f}) cm, yaw "
          f"{yaw90:+.1f} deg; offset from nominal {dev[0]:+.1f}, {dev[1]:+.1f} cm (trained +-{CUBE_RANGE_CM}); "
          f"triangulated ({cube[0] * 100:+.1f}, {cube[1] * 100:+.1f}, {cube[2] * 100:.1f}) cm; {ctl.info}", flush=True)
    refuse = []
    if abs(dev[0]) > CUBE_RANGE_CM or abs(dev[1]) > CUBE_RANGE_CM:
        refuse.append(f"cube {dev[0]:+.1f}, {dev[1]:+.1f} cm from the nominal spot (trained +-{CUBE_RANGE_CM} cm): move it")
    if abs(yaw90) > CUBE_YAW_DEG:
        refuse.append(f"cube turned {yaw90:+.0f} deg (trained +-15): square it up")
    bdx, bdy = (box[0] - det[0]) * 100, (box[1] - det[1]) * 100
    train_dy = C.BOX_DY - 0.005            # clamp_rl.py moves the box 5 mm further back on import (line 35); not
    dy_lo, dy_hi = (train_dy - 0.01) * 100 - BOX_DY_SLACK_CM, train_dy * 100 + BOX_DY_SLACK_CM   # imported here
    if abs(bdx) > BOX_DX_CM or not dy_lo <= bdy <= dy_hi:
        refuse.append(f"box at dx {bdx:+.1f}, dy {bdy:+.1f} cm from the cube (trained dx +-0.5, dy "
                      f"{train_dy * 100 - 1:.1f}..{train_dy * 100:.1f} cm): move the box")
    if refuse:
        stop_cams()
        raise SystemExit("not the trained layout:\n  " + "\n  ".join(refuse))
    try:
        guard = SF.Guard(m, SF.real_limits(arms), box_xy=(box[0], box[1]), box_yaw=box[2])
    except SF.SafetyStop as e:
        stop_cams()
        raise SystemExit(f"SAFETY: {e}")
    ctl.guard = guard
    filt = None if a.no_filter else CmdFilter(guard)
    print(f"safety guard ready (self-test {guard.self_test_mm} mm)"
          + ("" if filt is None else f"; command filter ready (self-test {filt.self_test_mm} mm)"), flush=True)
    (run / "meta.json").write_text(json.dumps({
        "cube": cube.tolist(), "cube_detector_xy": det.tolist(), "cube_yaw": yaw, "box": list(box), "live": live,
        "policy": str(POLICY), "calibration": tag, "offsets": a.offsets, "joint_offsets": off,
        "filter": filt is not None, "limiter": not a.no_limiter,
        "guard_margins_mm": {w: {k: v * 1000 for k, v in SF.MARGIN[w].items()} for w in ("cmd", "meas")}}, indent=1))

    if a.dry:
        t0 = time.perf_counter()
        f = frames()
        u = read_units()
        to_shadow(u)
        ctrl = {s: d.ctrl[sim.robot[s].actuator_ids].copy() for s in SIDES}
        prev_cmd = {s: np.r_[np.rad2deg(ctrl[s]), 0.0] for s in SIDES}
        ee_prev = sim._ee_targets.copy()
        cmd, log = ctl.tick(f, ctrl)
        kept = 1.0 if filt is None else CV.filter_cmd(filt, ctl, prev_cmd, cmd, ee_prev, 1.0)[1]
        ms = (time.perf_counter() - t0) * 1000
        print(f"one control tick (cameras + detectors + policy + IK + limiter + filter) took {ms:.0f} ms (budget 40 ms); "
              f"detector cube {np.round(runner.det_xy * 100, 1).tolist()} cm seen={runner.det_seen:.0f}; filter kept "
              f"{kept:.2f}; first target (arms where they are now, NOT the start pose) {log['target']}", flush=True)
        print("DRY: nothing moved", flush=True)
        stop_cams()
        return

    fold_log = []

    def ramp_raw(targets, secs, check_stop=True, log=None):
        """Scripted move: minimum-jerk S-curve at the stream rate (rt_control.smooth_ramp)."""
        return RT.smooth_ramp(arms, targets, cfg, secs, check_stop=STOP.exists if check_stop else None, log=log)

    def perceive():                                   # runs on the perception thread (no MuJoCo state touched)
        f = frames()
        return {"frames": f, "red": {c: CL.detect_red(f[c]) for c in CAMS},
                "pose": V.detect_cube_pose(f["overhead"], ctl.cam)}

    log, stream, perc, saver = [], None, None, None
    P_RUN, P_REST = cfg["servo_gain"]["run_p"], cfg["servo_gain"]["rest_p"]
    try:
        STOP.unlink(missing_ok=True)
        # ------------------------------------------------------------ to the start pose (training's start: SIM_HOME)
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
        ctl.start(res["frames"], dets=dict(res, seq=seq))   # fresh start at the start pose (re-reads the cube; keeps the box)
        prev_cmd = {s: np.r_[np.rad2deg(d.ctrl[sim.robot[s].actuator_ids]), 0.0] for s in SIDES}
        clock = RT.TickClock(cfg)
        saver = RT.FrameSaver(run / "frames", cfg)
        stream = RT.ServoStreamer(arms, cfg, dry=not live)
        stream.start(last_goal)
        print("at start pose: the chunk policy v2 is driving" if live else "dry loop running", flush=True)
        t0, k, dry_stops, next_print, cut = time.perf_counter(), 0, 0, 0.0, 0
        # ------------------------------------------------------------ the control loop
        while time.perf_counter() - t0 < a.seconds:
            if STOP.exists():
                raise KeyboardInterrupt("STOP file")
            tick = time.perf_counter()
            adv = clock.tick()
            u = read_units()                                       # sense: measured joints -> shadow
            to_shadow(u)
            t_read = time.perf_counter()
            res, seq, t_det = perc.get()                           # newest camera result (never waits)
            f = res["frames"]
            ctrl = {s: d.ctrl[sim.robot[s].actuator_ids].copy() for s in SIDES}
            ee_prev = sim._ee_targets.copy()
            cmd, info = ctl.tick(f, ctrl, dets=dict(res, seq=seq), adv=adv)    # decide: policy -> IK -> limiter
            t_ctl = time.perf_counter()
            kept = 1.0
            if filt is not None:                                   # filter: never send a pose inside a margin
                cmd, kept = CV.filter_cmd(filt, ctl, prev_cmd, cmd, ee_prev, adv)
                cut += kept < 1.0
            t_filt = time.perf_counter()
            expect = {s: cmd[s].copy() for s in SIDES}
            try:
                cmd = guard.step(u, cmd, expect=expect)            # guard: raises SafetyStop -> fold to rest
            except SF.SafetyStop as e:
                if live:
                    raise
                dry_stops += 1                                     # dry loop: the arms do not follow -> lead limit trips
                info["dry_guard"] = str(e)[:120]
            t_guard = time.perf_counter()
            prev_cmd = {s: cmd[s].copy() for s in SIDES}
            low = hh.lowest(np.r_[cmd["left"], cmd["right"]])      # floor guard
            blocked = [s for s in SIDES if low[s] * 100 < a.floor_cm]
            for s in SIDES:                                        # act: to the 100 Hz servo streamer
                if s in blocked:
                    stream.hold(s)
                else:
                    stream.set(s, arms[s].to_raw(cmd[s]), clock.period())
            if blocked:
                print(f"  floor guard held {blocked} (lowest {min(low.values()) * 100:.1f} cm)", flush=True)
            t = time.perf_counter() - t0
            saver.maybe(t, f, f"t{k:04d}.jpg")
            log.append({"t": round(t, 3), **info, "adv": round(adv, 2), "det_age_ms": round((tick - t_det) * 1000, 1),
                        "det_xy": np.round(runner.det_xy, 4).tolist(), "det_seen": runner.det_seen,
                        "filter_kept": round(float(kept), 3),
                        "ms": {"read": round((t_read - tick) * 1000, 1), "ctl": round((t_ctl - t_read) * 1000, 1),
                               "filter": round((t_filt - t_ctl) * 1000, 1), "guard": round((t_guard - t_filt) * 1000, 1)},
                        "units": {s: u[s].round(2).tolist() for s in SIDES},
                        "cmd": {s: cmd[s].round(2).tolist() for s in SIDES}, "floor_blocked": blocked,
                        "closest_mm": {w: dict(v) for w, v in guard.closest.items()}})
            if t >= next_print:
                next_print += 1.0
                print(f"  t={t:4.1f}s  {1 / max(clock.period(), 1e-3):4.1f} Hz  attempt {info['attempt']} clock "
                      f"{info['clock']:.2f}  cube(det) {np.round(runner.det_xy * 100, 1).tolist()} cm seen="
                      f"{runner.det_seen:.0f}  cube(tri) {np.round(np.array(info['cube_xyz']) * 100, 1).tolist()} cm"
                      f"{'  [squeeze limited]' if info['squeeze_limited'] else ''}"
                      f"{f'  [filter kept {kept:.2f}]' if kept < 1 else ''}", flush=True)
            k += 1
            clock.sleep_rest(tick)
        if live:
            Image.fromarray(cams["overhead"].latest()).save(run / "end_overhead.jpg", quality=90)
        dts = np.diff([r["t"] for r in log]) * 1000 if len(log) > 2 else np.array([0.0])
        st = {w: np.median([r["ms"][w] for r in log]) for w in ("read", "ctl", "filter", "guard")} if log else {}
        print(f"chunk policy v2 finished: {len(log)} ticks, period median {np.median(dts):.0f} ms / p90 "
              f"{np.percentile(dts, 90):.0f} ms ({100 * np.mean(dts > 44):.0f} % over 40 ms); stage medians "
              f"{ {w: round(v, 1) for w, v in st.items()} } ms; perception {np.median(perc.ms):.0f} ms/frame-set; "
              f"servo writes {stream.writes / max(1e-3, time.perf_counter() - t0):.0f}/s; filter cut {cut} ticks"
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
        # ------------------------------------------------------------ any end / stop: fold to rest, torque off
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
