"""LIVE runner for SmolVLA (vision-language-action, bb1b28 retrain) on the REAL SO-101 arms.

Policy: C:\\robot\\vla\\out\\smolvla_bb1b28\\checkpoints\\020000\\pretrained_model (LeRobot SmolVLA, trained on 292 sim
episodes rendered with the bb1b28 calibration + 11 real episodes x8; sim: 5/12 delivered, 12/12 squeezed + lifted, the
failures time out carrying the cube to the box). Served on the laptop GPU by vla_server.py (C:\\robot\\vla\\venv).

Executed exactly as in the sim evaluation (squeeze/vla_sim_eval.py), which is how its numbers were measured:
  every EXEC = 10 ticks: the 3 camera frames (640x480 -> 320x240 bilinear -> JPEG q90, as the dataset) + the measured
  joints (10 sim degrees, bb1b28 offsets) -> one 50-step chunk of joint targets. The arms HOLD STILL while it thinks
  (the sim's time is frozen during inference too). Then 10 ticks at 25 Hz: target = chunk[k], through the sim's joint
  rate limit (2 rad/s = 4.6 deg per 40 ms tick). Grippers commanded shut (0), as every clamp runner.

Protection, every tick, as every real runner (real_chunk_v2.py):
  filter    cmd_filter.CmdFilter: never SEND a pose inside the guard's commanded-pose margins (other arm 1 mm, camera
            mounts 12 mm, camera post 15 mm, box 6 mm); send the farthest point toward it that passes
  guard     safety.Guard on the commanded AND measured pose (camera mounts / other arm / post / box / real joint ranges /
            stuck joint / 10 deg lead limit), floor guard, STOP file. Any stop -> fold to rest, torque off.
            The MEASURED arm-arm stop (3 mm) is removed by default, as in real_chunk_v2.py (user, 2026-11-13);
            --arm-arm-guard keeps it. The commanded arm-arm check and every camera-mount margin always stay.
  act       servo goals streamed at 100 Hz (rt_control.ServoStreamer); per-joint torque caps; servo P 32 for the run only

Before moving it refuses unless the overhead finds the box and the cube, the cube is within the trained area (+-1.5 cm,
yaw +-15 deg) and the box at the trained spacing behind it (the VLA data came from the same env as chunk policy v2).

    python real_vla_live.py --dry-loop   # server + start checks + the whole loop for --ticks with torque OFF, no goal
                                         # written: measures inference time and the loop. NOTHING moves
    python real_vla_live.py --live       # moves the arms
  options: --ckpt DIR  --offsets new|current  --exec N (chunk steps per inference, default 10 as evaluated)
           --state-dim 10|12 (12 = + gripper % per arm; actions then carry the gripper too)  --task TEXT
           --max-yaw-deg D (cube rotation allowed at the start; default 20 = trained +-15 + slack; 45 = any)
           --ticks N (policy ticks, default 450 = the sim episode)  --no-filter  --arm-arm-guard  --floor-cm C
Log + frames: C:\\robot\\real\\runs\\real_vla_live_<time>.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(r"C:\robot\real")
SIM = HERE / "sim_mac"
SQ = SIM / "squeeze"
sys.path[:0] = [str(SQ), str(SIM), str(HERE)]
sys.meta_path[:] = [f for f in sys.meta_path if "__editable___pi_sim" not in getattr(f, "__module__", "")]
os.chdir(HERE)
os.environ["SQUEEZE_TASK"] = "clamp"
os.environ["CLAMP_DESIGN"] = "design_clamp3.json"

import mujoco                      # noqa: E402
import numpy as np                 # noqa: E402
from PIL import Image              # noqa: E402

import clamp_control as CC         # noqa: E402  (servo gain)
import safety as SF                # noqa: E402
import so101_real as R             # noqa: E402
import cube_locator as CL          # noqa: E402
import rt_control as RT            # noqa: E402
from cmd_filter import CmdFilter   # noqa: E402

STOP = HERE / "STOP"
SIDES = ("left", "right")
CAMS = ("overhead", "left_wrist", "right_wrist")
VLA_PY = r"C:\robot\vla\venv\Scripts\python.exe"
SERVER = r"C:\robot\vla\vla_server.py"
PORT = 8791
DEFAULT_CKPT = "C:/robot/vla/out/smolvla_bb1b28/checkpoints/020000/pretrained_model"
CALIB = Path(r"C:\robot\vla\calib_bb1b28")
CALIB_TAG = "cams 6349a0 + kinematics bb1b28"
CUBE_RANGE_CM, CUBE_YAW_DEG = 1.5, 15.0 + 5.0
BOX_DX_CM, BOX_DY_SLACK_CM = 0.5 + 0.6, 0.6
RATE_RAD_PER_TICK = 2.0 * 0.04     # pi_sim joint_rate_limit x control_dt: the sim eval's rate limit


def ask(f320, state):
    buf = io.BytesIO()
    np.savez(buf, camera1=f320["overhead"], camera2=f320["left_wrist"], camera3=f320["right_wrist"],
             state=np.asarray(state, np.float32))
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/chunk", data=buf.getvalue(), method="POST")
    z = np.load(io.BytesIO(urllib.request.urlopen(req, timeout=60).read()))
    return z["chunk"], float(z["ms"])


def as_dataset(frame):
    """640x480 RGB -> 320x240 bilinear -> JPEG q90 -> RGB: exactly how the dataset / the sim eval made its frames."""
    im = Image.fromarray(frame).resize((320, 240), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--dry-loop", action="store_true", help="full loop with torque OFF and no goal written")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--offsets", default="new", choices=("new", "current"))
    ap.add_argument("--exec", type=int, default=10, help="chunk steps executed per inference (sim eval: 10)")
    ap.add_argument("--ticks", type=int, default=450, help="policy ticks (the sim episode: 450 = 18 s)")
    ap.add_argument("--state-dim", type=int, default=10, choices=(10, 12),
                    help="10 = 5 joints per arm; 12 = + gripper %% per arm (sim-74's SmolVLA), actions then 12 too")
    ap.add_argument("--task", default=None, help="the instruction this checkpoint was trained with")
    ap.add_argument("--max-yaw-deg", type=float, default=CUBE_YAW_DEG, help="start refused beyond this cube rotation")
    ap.add_argument("--no-filter", action="store_true", help="no command filter (the guard still runs)")
    ap.add_argument("--arm-arm-guard", action="store_true",
                    help="keep the guard's measured arm-arm stop (3 mm); default: removed (user, as real_chunk_v2)")
    ap.add_argument("--floor-cm", type=float, default=-0.5)
    a = ap.parse_args()
    live = a.live and not a.dry_loop
    if not a.arm_arm_guard:
        SF.MARGIN["meas"].pop("arm-arm", None)
    cfg = RT.load_config()
    print(f"control config {cfg['_path']}: policy {cfg['control']['hz']} Hz, servo stream {cfg['servo_stream']['hz']} Hz; "
          f"guard margins (mm): commanded {({k: v * 1000 for k, v in SF.MARGIN['cmd'].items()})}, measured "
          f"{({k: v * 1000 for k, v in SF.MARGIN['meas'].items()})}", flush=True)

    import clamp_teacher as C
    import env as E
    import vision as V
    rig_cams = CALIB / "so101_camera_rig.json"
    tag = json.loads(rig_cams.read_text()).get("calibration_tag")
    if tag != CALIB_TAG:
        raise SystemExit(f"camera rig {rig_cams} is calibration {tag!r}, the model was trained on {CALIB_TAG!r}")
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
    intr_o = CL.cam_intrinsics(sim.camera_specs["overhead"])
    ocam = V.OverheadCamera.from_env(sim)
    hh = R.HandHeight()

    run = HERE / "runs" / f"real_vla_live_{time.strftime('%Y%m%d_%H%M%S')}"
    (run / "frames").mkdir(parents=True, exist_ok=True)
    print(f"starting the VLA server on the GPU: {a.ckpt}", flush=True)
    srv_log = open(run / "vla_server.log", "w")
    srv = subprocess.Popen([VLA_PY, "-u", SERVER, a.ckpt, "--port", str(PORT)] + (["--task", a.task] if a.task else []),
                           stdout=srv_log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONWARNINGS="ignore"))
    arms, cams = None, None
    log, stream, saver, fold_log = [], None, None, []
    P_RUN, P_REST = cfg["servo_gain"]["run_p"], cfg["servo_gain"]["rest_p"]
    torque_on = False
    try:
        t0 = time.time()
        while time.time() - t0 < 300:
            if srv.poll() is not None:
                raise SystemExit(f"VLA server exited ({srv.returncode}); see {run / 'vla_server.log'}")
            if "serving on" in (run / "vla_server.log").read_text(errors="ignore"):
                break
            time.sleep(1)
        else:
            raise SystemExit("VLA server did not come up in 5 min")
        print(f"VLA server ready ({time.time() - t0:.0f} s)", flush=True)

        arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in SIDES}
        for s, arm in arms.items():
            if arm.fingerprint != R.FINGERPRINTS[s]:
                raise SystemExit(f"{rig['arms'][s]} is not the {s} arm")
        cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}

        def read_units():
            return {s: arms[s].to_units(arms[s].read_raw()).astype(float) for s in SIDES}

        def to_shadow(u):
            for s in SIDES:
                d.qpos[sim.robot[s].qpos_ids] = np.deg2rad(u[s][:5])
                d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = glo + np.clip(u[s][5], 0, 100) / 100 * (ghi - glo)
            mujoco.mj_forward(m, d)

        def frames():
            return {c: cams[c].latest() for c in CAMS}

        # ------------------------------------------------------------ box + cube, checked against the trained layout
        time.sleep(2.0)                                            # let auto-exposure settle
        to_shadow(read_units())
        box = cube = pose = None
        for _ in range(15):
            f = frames()
            box = V.detect_box(f["overhead"], ocam)
            uv, _ = CL.detect_red(f["overhead"])
            cube = None if uv is None else CL.locate([CL.ray(intr_o, *CL.camera_pose(m, d, "overhead"), *uv)])[0]
            pose = V.detect_cube_pose(f["overhead"], ocam)
            if box is not None and cube is not None and pose is not None:
                break
            time.sleep(0.2)
        Image.fromarray(cams["overhead"].latest()).save(run / "detect_overhead.jpg", quality=90)
        if box is None or cube is None or pose is None:
            raise SystemExit(f"cannot start: box {'found' if box is not None else 'NOT found'}, cube "
                             f"{'found' if cube is not None else 'NOT found'}, cube yaw {'found' if pose is not None else 'NOT found'}"
                             f" by the overhead (camera indices swapped? see {run / 'detect_overhead.jpg'})")
        dev = (cube[0] * 100, (cube[1] - C.CUBE_Y) * 100)
        yaw90 = (np.degrees(float(pose[2])) + 45.0) % 90.0 - 45.0
        bdx, bdy = (box[0] - cube[0]) * 100, (box[1] - cube[1]) * 100
        train_dy = C.BOX_DY - 0.005                               # clamp_rl.py moves the box 5 mm back on import
        print(f"box at ({box[0] * 100:+.1f}, {box[1] * 100:+.1f}) cm yaw {np.degrees(box[2]):+.1f} deg; cube at "
              f"({cube[0] * 100:+.1f}, {cube[1] * 100:+.1f}) cm yaw {yaw90:+.1f} deg; offset from nominal {dev[0]:+.1f}, "
              f"{dev[1]:+.1f} cm; box from cube dx {bdx:+.1f} dy {bdy:+.1f} cm", flush=True)
        refuse = []
        if abs(dev[0]) > CUBE_RANGE_CM or abs(dev[1]) > CUBE_RANGE_CM:
            refuse.append(f"cube {dev[0]:+.1f}, {dev[1]:+.1f} cm from the nominal spot (trained +-{CUBE_RANGE_CM} cm): move it")
        if abs(yaw90) > a.max_yaw_deg:
            refuse.append(f"cube turned {yaw90:+.0f} deg (allowed +-{a.max_yaw_deg:.0f}): square it up")
        if abs(bdx) > BOX_DX_CM or not (train_dy - 0.01) * 100 - BOX_DY_SLACK_CM <= bdy <= train_dy * 100 + BOX_DY_SLACK_CM:
            refuse.append(f"box at dx {bdx:+.1f}, dy {bdy:+.1f} cm from the cube (trained dx +-0.5, dy "
                          f"{train_dy * 100 - 1:.1f}..{train_dy * 100:.1f} cm): move the box")
        if refuse:
            raise SystemExit("not the trained layout:\n  " + "\n  ".join(refuse))
        guard = SF.Guard(m, SF.real_limits(arms), box_xy=(box[0], box[1]), box_yaw=box[2])
        filt = None if a.no_filter else CmdFilter(guard)
        print(f"safety guard ready (self-test {guard.self_test_mm} mm)"
              + ("" if filt is None else f"; command filter ready (self-test {filt.self_test_mm} mm)"), flush=True)
        (run / "meta.json").write_text(json.dumps({
            "cube": [float(v) for v in cube], "cube_yaw": float(pose[2]), "box": [float(v) for v in box], "live": live,
            "policy": "SmolVLA " + a.ckpt, "calibration": tag, "offsets": a.offsets, "joint_offsets": off,
            "exec": a.exec, "filter": filt is not None,
            "guard_margins_mm": {w: {k: v * 1000 for k, v in SF.MARGIN[w].items()} for w in ("cmd", "meas")}}, indent=1))

        def ramp_raw(targets, secs, check_stop=True, log=None):
            return RT.smooth_ramp(arms, targets, cfg, secs, check_stop=STOP.exists if check_stop else None, log=log)

        STOP.unlink(missing_ok=True)
        # ------------------------------------------------------------ to the start pose (the sim episodes' start)
        if live:
            for arm in arms.values():
                arm.prepare()
                arm.goal(arm.read_raw().astype(int))
            SF.apply_torque_limits(arms)
            print(f"stiffer servos for this run: P {P_REST} -> {P_RUN}, read back {CC.set_p_gain(arms, P_RUN)}", flush=True)
            for arm in arms.values():
                arm.torque(True)
            torque_on = True
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
            print("DRY LOOP: torque stays OFF, no goal is written; inference + loop timing only", flush=True)
            last_goal = {s: arms[s].read_raw().astype(int) for s in SIDES}
        u = read_units()
        to_shadow(u)
        ctrl = {s: np.r_[u[s][:5], 0.0] for s in SIDES}            # last command (deg); starts at where the arms are
        prev_cmd = {s: ctrl[s].copy() for s in SIDES}
        t_w = time.perf_counter()
        def state_of(u):
            if a.state_dim == 10:
                return np.r_[u["left"][:5], u["right"][:5]]
            return np.r_[u["left"][:5], np.clip(u["left"][5], 0, 100), u["right"][:5], np.clip(u["right"][5], 0, 100)]
        ask({c: as_dataset(v) for c, v in frames().items()}, state_of(u))   # GPU warm-up
        print(f"warm-up inference {1000 * (time.perf_counter() - t_w):.0f} ms", flush=True)
        clock = RT.TickClock(cfg)
        saver = RT.FrameSaver(run / "frames", cfg)
        stream = RT.ServoStreamer(arms, cfg, dry=not live)
        stream.start(last_goal)
        print("at start pose: SmolVLA is driving" if live else "dry loop running", flush=True)
        plan, k, n_inf, inf_ms, cut, dry_stops = None, 0, 0, [], 0, 0
        t0 = time.perf_counter()
        # ------------------------------------------------------------ the control loop
        for tick_i in range(a.ticks):
            if STOP.exists():
                raise KeyboardInterrupt("STOP file")
            if plan is None or k >= a.exec:                         # THINK: arms hold still meanwhile (as the sim)
                u = read_units()
                to_shadow(u)
                f = frames()
                t_a = time.perf_counter()
                plan, ms = ask({c: as_dataset(v) for c, v in f.items()}, state_of(u))
                inf_ms.append(1000 * (time.perf_counter() - t_a))
                n_inf += 1
                k = 0
                if plan.shape[1] != a.state_dim:
                    raise SystemExit(f"action size {plan.shape[1]}, expected {a.state_dim} (--state-dim)")
                clock.last = None                                   # the pause is not a control tick
            tick = time.perf_counter()
            adv = clock.tick()
            u = read_units()                                        # SENSE
            to_shadow(u)
            a_ = plan[k]                                            # DECIDE: chunk step k, through the sim's rate limit
            if a.state_dim == 10:
                tgt, grip = {"left": a_[:5], "right": a_[5:10]}, {"left": 0.0, "right": 0.0}   # jaws shut, as every clamp runner
            else:
                tgt, grip = {"left": a_[0:5], "right": a_[6:11]}, {"left": float(np.clip(a_[5], 0, 100)),
                                                                   "right": float(np.clip(a_[11], 0, 100))}
            rate = np.degrees(RATE_RAD_PER_TICK) * adv
            cmd = {s: np.r_[ctrl[s][:5] + np.clip(tgt[s] - ctrl[s][:5], -rate, rate), grip[s]] for s in SIDES}
            k += 1
            kept = 1.0
            if filt is not None:                                    # FILTER: never send a pose inside a margin
                cmd, kept = filt(prev_cmd, cmd)
                cmd = {s: np.asarray(cmd[s], float) for s in SIDES}
                cut += kept < 1.0
            try:
                cmd = guard.step(u, cmd, expect={s: cmd[s].copy() for s in SIDES})   # GUARD
            except SF.SafetyStop as e:
                if live:
                    raise
                dry_stops += 1
            ctrl = {s: np.asarray(cmd[s], float).copy() for s in SIDES}
            prev_cmd = {s: ctrl[s].copy() for s in SIDES}
            low = hh.lowest(np.r_[cmd["left"], cmd["right"]])       # floor guard
            blocked = [s for s in SIDES if low[s] * 100 < a.floor_cm]
            for s in SIDES:                                         # ACT
                if s in blocked:
                    stream.hold(s)
                else:
                    stream.set(s, arms[s].to_raw(cmd[s]), clock.period())
            if blocked:
                print(f"  floor guard held {blocked} (lowest {min(low.values()) * 100:.1f} cm)", flush=True)
            t = time.perf_counter() - t0
            saver.maybe(t, frames(), f"t{tick_i:04d}.jpg")
            log.append({"t": round(t, 3), "tick": tick_i, "chunk_step": k - 1, "inference": n_inf, "adv": round(adv, 2),
                        "target": {s: np.round(tgt[s], 2).tolist() for s in SIDES}, "filter_kept": round(float(kept), 3),
                        "units": {s: u[s].round(2).tolist() for s in SIDES},
                        "cmd": {s: np.asarray(cmd[s]).round(2).tolist() for s in SIDES}, "floor_blocked": blocked,
                        "closest_mm": {w: dict(v) for w, v in guard.closest.items()}})
            if k == 1:
                tips = {s: np.round(sim.robot[s].position * 100, 1).tolist() for s in SIDES}
                print(f"  t={t:5.1f}s tick {tick_i} inference {n_inf} ({inf_ms[-1]:.0f} ms) tips L {tips['left']} "
                      f"R {tips['right']} cm{f'  [filter kept {kept:.2f}]' if kept < 1 else ''}", flush=True)
            clock.sleep_rest(tick)
        if live:
            Image.fromarray(cams["overhead"].latest()).save(run / "end_overhead.jpg", quality=90)
        print(f"SmolVLA finished: {len(log)} ticks, {n_inf} inferences (median {np.median(inf_ms):.0f} ms, max "
              f"{max(inf_ms):.0f} ms), {time.perf_counter() - t0:.0f} s; filter cut {cut} ticks"
              f"{'; dry guard trips ' + str(dry_stops) if not live else ''}", flush=True)
    except SF.SafetyStop as e:
        print(f"SAFETY STOP: {e}", flush=True)
    except BaseException as e:
        print(f"stopping: {e!r}", flush=True)
    finally:
        for w in (stream, saver):                              # the streamer MUST stop before anything else writes goals
            if w is not None:
                try:
                    w.stop()
                except Exception as e:
                    print(f"stopping {type(w).__name__} failed: {e!r}", flush=True)
        if log:
            (run / "log.json").write_text(json.dumps(log))
        print(f"log: {run} ({len(log)} ticks)", flush=True)
        STOP.unlink(missing_ok=True)
        srv.terminate()
        try:
            srv.wait(10)
        except Exception:
            srv.kill()
        if torque_on:
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
                RT.smooth_ramp(arms, back, cfg, 3.0, check_stop=None, log=fold_log)
                rest = {s: arms[s].neutral_raw() for s in SIDES}
                mid = {}
                for s in SIDES:
                    r = arms[s].read_raw().copy()
                    r[0], r[3], r[4], r[5] = rest[s][0], rest[s][3], rest[s][4], rest[s][5]
                    mid[s] = np.round(r).astype(int)
                RT.smooth_ramp(arms, mid, cfg, 2.5, check_stop=None, log=fold_log)
                RT.smooth_ramp(arms, rest, cfg, 3.0, check_stop=None, log=fold_log)
                time.sleep(0.4)
                (run / "fold.json").write_text(json.dumps(fold_log))
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
        elif not live:
            print("DRY LOOP: nothing moved", flush=True)
        if cams is not None:
            for c in cams.values():
                c.stop.set()


if __name__ == "__main__":
    main()
