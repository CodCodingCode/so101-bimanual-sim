"""CAMERA POLICY on the REAL SO-101 arms: the vision student (squeeze/models/camera_student.pt) clamps the cube with the
fingertips, carries it and drops it in the box. Runs ON dzkchen with the monitor stopped.

What the policy gets, exactly as in sim (squeeze/distill.py observe()):
  - overhead / left wrist / right wrist frames, 640x480 -> 64x64 bilinear (the sim renders 640x480 with the calibrated
    lens and resizes the same way)
  - per arm: joint angles (deg / 90), gripper opening, tool point (forward kinematics) x5, commanded tool point x5
  - its previous action, the grab-attempt clock (steps / 450; one attempt: no re-grab detection on the rig yet)
Its action moves each arm's commanded tool point by a * 0.5 * delta_scale (sim ee_delta control); a kinematic shadow of
the training sim does IK + rate limit, as in real_clamp.py. Gripper orientation from the cube spot (overhead camera),
as at sim reset. 25 Hz, 18 s (the clamp takes ~16 s), then fold to rest.

    python real_student.py --dry     # cameras + policy, NO motion: saves the 64x64 inputs and the first actions
    python real_student.py --live    # run it
Safety (safety.py): every tick the guard checks the commanded AND measured pose against each servo's real range and the
arm / wrist-camera-mount / camera-post / box clearances, watches tracking, and limits how far a command may lead the
joint; per-joint torque caps; HandHeight floor guard; C:\\robot\\real\\STOP. Any stop ALWAYS folds the arms back to rest
before torque off. Logs to runs/real_student_<ts>/.
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
import torch                       # noqa: E402
import torch.nn as nn              # noqa: E402
from PIL import Image              # noqa: E402

import clamp_control as CC        # noqa: E402  (servo gain)
import safety as SF                # noqa: E402
import rt_control as RT            # noqa: E402
sys.path.insert(0, str(HERE / "so101_actuation"))
import tracking as TR              # noqa: E402  (capped joint correction for the servos' shortfall)
import so101_real as R             # noqa: E402
from real_clamp import camera_model, find_cube   # noqa: E402

STOP = HERE / "STOP"
SIDES = ("left", "right")
CAMS = ("overhead", "left_wrist", "right_wrist")
IMG, PROP = 64, 31
ACTION_SCALE = 0.5
CLOCK_STEPS = 450
STEPS = 550                        # 22 s at 25 Hz (the clamp + place takes ~16 s)
POLICY_FILE = HERE / "squeeze_rt" / "camera_student_v3.pt"
NEAR_CUBE = 0.03                   # m: joint correction learns only while both tips are farther than this from the cube


class Student(nn.Module):          # identical to squeeze/distill.py build_student()
    def __init__(self):
        super().__init__()

        def enc():
            return nn.Sequential(
                nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU())
        self.encs = nn.ModuleList([enc() for _ in CAMS])
        self.prop = nn.Sequential(nn.Linear(PROP, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 * len(CAMS) + 64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 8))

    def forward_all(self, imgs, prop):     # (action x6 in [-1, 1], cube xy estimate x2: (x / 5 cm, (y - CUBE_Y) / 5 cm))
        feats = [e(imgs[:, i]) for i, e in enumerate(self.encs)]
        out = self.head(torch.cat(feats + [self.prop(prop)], 1))
        return torch.tanh(out[:, :6]), out[:, 6:8]


def small(rgb):
    return np.asarray(Image.fromarray(rgb).resize((IMG, IMG), Image.Resampling.BILINEAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--floor-cm", type=float, default=-0.5)
    ap.add_argument("--seconds", type=float, default=STEPS / 25)
    a = ap.parse_args()
    live = a.live and not a.dry
    torch.set_num_threads(2)

    import clamp_teacher as C
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

    net = Student()
    net.load_state_dict(torch.load(POLICY_FILE, map_location="cpu"))
    net.eval()

    rig = json.loads((HERE / "rig.json").read_text())
    off = rig.get("joint_offsets_deg", {})
    arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in SIDES}
    for s, arm in arms.items():
        if arm.fingerprint != R.FINGERPRINTS[s]:
            raise SystemExit(f"{rig['arms'][s]} is not the {s} arm")
    cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}
    run = HERE / "runs" / f"real_student_{time.strftime('%Y%m%d_%H%M%S')}"
    (run / "inputs").mkdir(parents=True, exist_ok=True)

    def read_units():
        return {s: arms[s].to_units(arms[s].read_raw()).astype(float) for s in SIDES}

    def to_shadow(u):
        for s in SIDES:
            arm = sim.robot[s]
            d.qpos[arm.qpos_ids] = np.deg2rad(u[s][:5])
            d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = glo + np.clip(u[s][5], 0, 100) / 100 * (ghi - glo)
        mujoco.mj_forward(m, d)

    def observe(prev, k):
        frames = {c: cams[c].latest() for c in CAMS}
        imgs = np.stack([small(frames[c]) for c in CAMS])
        parts = []
        for i, s in enumerate(SIDES):
            arm = sim.robot[s]
            parts += [np.rad2deg(d.qpos[arm.qpos_ids]) / 90.0, [arm.gripper_opening], arm.position * 5.0,
                      sim._ee_targets[i] * 5.0]
        parts += [prev, [k / CLOCK_STEPS]]
        return frames, imgs, np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])

    def act(imgs, prop):
        """(action, the policy's own estimate of the cube xy in m)"""
        x = torch.from_numpy(imgs[None]).float().div(255.0).permute(0, 1, 4, 2, 3).contiguous()
        with torch.no_grad():
            a_, q = net.forward_all(x, torch.from_numpy(prop[None]))
        q = q.numpy()[0]
        return a_.numpy()[0].astype(np.float32), np.array([q[0] * 0.05, C.CUBE_Y + q[1] * 0.05])

    time.sleep(2.0)                                                # let auto-exposure settle
    cube, npx = None, 0
    for _ in range(15):
        cube, npx = find_cube(cams["overhead"].latest(), backproject, C.CUBE_HALF)
        if cube is not None:
            break
        time.sleep(0.2)
    Image.fromarray(cams["overhead"].latest()).save(run / "detect_overhead.jpg", quality=90)
    if cube is None:
        for c in cams.values():
            c.stop.set()
        raise SystemExit(f"cube not found in the overhead camera ({npx} red px)")
    cx, cy = float(cube[0]), float(cube[1])
    dev = (cx * 100, (cy - C.CUBE_Y) * 100)
    print(f"cube at x {cx * 100:+.1f} cm, y {cy * 100:+.1f} cm; offset from nominal {dev[0]:+.1f}, {dev[1]:+.1f} cm "
          f"(allowed +-1.5)", flush=True)
    if abs(dev[0]) > 1.5 or abs(dev[1]) > 1.5:          # clamp v3 is verified clean within +-1.5 cm (sim, 24/24)
        for c in cams.values():
            c.stop.set()
        raise SystemExit("cube outside the trained area: move it into the green zone")
    for s in SIDES:
        sim.pad_quat[s] = C.clamp_quat(s, (cx, cy))
    box = (cx, cy + C.BOX_DY - 0.005)               # where training put the box relative to the cube
    try:
        guard = SF.Guard(m, SF.real_limits(arms), box_xy=box)
    except SF.SafetyStop as e:
        for c in cams.values():
            c.stop.set()
        raise SystemExit(f"SAFETY: {e}")
    print("safety guard ready (self-test OK); real joint ranges (deg): " + " | ".join(
        f"{s} " + " ".join(f"{lo:.0f}..{hi:.0f}" for lo, hi in zip(*guard.limits[s])) for s in SIDES), flush=True)
    (run / "meta.json").write_text(json.dumps({"cube": [cx, cy], "rig": rig, "live": live, "policy": POLICY_FILE.name}, indent=1))

    if not live:
        u = read_units()
        to_shadow(u)
        for i, s in enumerate(SIDES):
            sim._ee_targets[i] = sim.robot[s].position.copy()
        frames, imgs, prop = observe(np.zeros(6, np.float32), 0)
        for c in CAMS:
            Image.fromarray(frames[c]).save(run / "inputs" / f"{c}_full.jpg", quality=92)
            Image.fromarray(imgs[CAMS.index(c)]).save(run / "inputs" / f"{c}_64.png")
        np.save(run / "inputs" / "units.npy", np.r_[u["left"], u["right"]])
        t0 = time.perf_counter()
        outs = [act(imgs, prop) for _ in range(10)]
        ms = (time.perf_counter() - t0) / 10 * 1000
        est = outs[0][1]
        err = np.linalg.norm(est - np.array([cx, cy])) * 100
        print(f"policy runs: {ms:.1f} ms per step (budget 40 ms)", flush=True)
        print(f"VISION CHECK: the policy thinks the cube is at ({est[0] * 100:+.1f}, {(est[1] - C.CUBE_Y) * 100:+.1f}) cm; "
              f"the overhead detector says ({cx * 100:+.1f}, {(cy - C.CUBE_Y) * 100:+.1f}) cm -> off by {err:.1f} cm "
              f"({'OK' if err < 1.0 else 'TOO FAR: do not run live'})", flush=True)
        print(f"DRY: nothing moved. inputs saved to {run / 'inputs'}", flush=True)
        for c in cams.values():
            c.stop.set()
        return

    cfg = RT.load_config()
    fold_log = []
    P_RUN, P_REST = cfg["servo_gain"]["run_p"], cfg["servo_gain"]["rest_p"]

    def ramp_raw(targets, secs, check_stop=True, log=None):
        """Scripted move: minimum-jerk S-curve at the stream rate (rt_control.smooth_ramp)."""
        return RT.smooth_ramp(arms, targets, cfg, secs, check_stop=STOP.exists if check_stop else None, log=log)

    log, stream = [], None
    try:
        STOP.unlink(missing_ok=True)
        for arm in arms.values():
            arm.prepare()
            arm.goal(arm.read_raw().astype(int))
        SF.apply_torque_limits(arms)
        print(f"stiffer servos for this run: P {P_REST} -> {P_RUN}, read back {CC.set_p_gain(arms, P_RUN)}", flush=True)
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
        u = read_units()
        to_shadow(u)
        for i, s in enumerate(SIDES):
            arm = sim.robot[s]
            sim._ee_targets[i] = arm.position.copy()
            d.ctrl[arm.actuator_ids] = d.qpos[arm.qpos_ids]
        print("at start pose, camera policy driving", flush=True)
        clock = RT.TickClock(cfg)
        stream = RT.ServoStreamer(arms, cfg)
        stream.start(hraw)
        prev, t0, slow, k, kf = np.zeros(6, np.float32), time.perf_counter(), 0, 0, 0.0
        corr = TR.TrackingCorrector(TR.load_profile(HERE / "so101_actuation" / "so101_profile.json"))
        while time.perf_counter() - t0 < a.seconds:
            if STOP.exists():
                raise KeyboardInterrupt("STOP file")
            tick = time.perf_counter()
            adv = clock.tick()
            u = read_units()
            to_shadow(u)
            frames, imgs, prop = observe(prev, kf)
            action, est = act(imgs, prop)
            action = np.clip(action, -1, 1)
            delta = action.reshape(2, 3) * ACTION_SCALE * sim.delta_scale * adv      # per training tick x ticks elapsed
            cmd = {}
            for i, s in enumerate(SIDES):
                arm = sim.robot[s]
                sim._ee_targets[i] = np.clip(sim._ee_targets[i] + delta[i], sim._workspace_low[i], sim._workspace_high[i])
                target, _ = sim._solve_ik(arm, sim._ee_targets[i], None)
                cur = d.ctrl[arm.actuator_ids].copy()
                new = cur + np.clip(target - cur, -sim._rate_limit * adv, sim._rate_limit * adv)
                d.ctrl[arm.actuator_ids] = new
                cmd[s] = np.r_[np.rad2deg(new), 0.0]
            # servo shortfall: nudge each joint toward its target while both tips are still away from the cube
            near = min(np.linalg.norm(sim.robot[s].position[:2] - np.array([cx, cy])) for s in SIDES) < NEAR_CUBE
            if near:
                corr.freeze()
            expect = cmd
            cmd = corr.step(cmd, {s: u[s][:5] for s in SIDES}, active=not near)
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
            prev = action.astype(np.float32)
            log.append({"t": round(time.perf_counter() - t0, 3), "adv": round(adv, 2), "action": action.round(3).tolist(), "closest_mm": {w: dict(v) for w, v in guard.closest.items()},
                        "cube_est": np.round(est, 4).tolist(), "joint_corr_deg": {s_: np.round(corr.corr[s_], 2).tolist() for s_ in SIDES},
                        "units": {s: u[s].round(2).tolist() for s in SIDES}, "ee_target": sim._ee_targets.round(4).tolist()})
            if k % 25 == 0:
                print(f"  t={time.perf_counter() - t0:4.1f}s  {1 / max(clock.period(), 1e-3):4.1f} Hz  tool L {np.round(sim._ee_targets[0] * 100, 1).tolist()} cm  "
                      f"R {np.round(sim._ee_targets[1] * 100, 1).tolist()} cm  | policy sees cube at "
                      f"({est[0] * 100:+.1f}, {(est[1] - C.CUBE_Y) * 100:+.1f}) cm{'  [correction frozen]' if corr.frozen else ''}", flush=True)
            if k % 5 == 0:
                Image.fromarray(np.concatenate([np.asarray(Image.fromarray(frames[c]).resize((320, 240))) for c in CAMS]
                                               + [np.asarray(Image.fromarray(np.concatenate(list(imgs), 0)).resize((80, 240), Image.NEAREST))], 1)) \
                    .save(run / "inputs" / f"t{k:04d}.jpg", quality=80)
            slow += (time.perf_counter() - tick) > 0.04
            k += 1
            kf += adv
            clock.sleep_rest(tick)
        Image.fromarray(cams["overhead"].latest()).save(run / "end_overhead.jpg", quality=90)
        print(f"camera policy finished ({slow} of {len(log)} ticks over 40 ms)", flush=True)
    except SF.SafetyStop as e:
        print(f"SAFETY STOP: {e}", flush=True)
    except BaseException as e:
        print(f"stopping: {e!r}", flush=True)
    finally:
        if stream is not None:                                # the streamer MUST stop before anything else writes goals
            try:
                stream.stop()
            except Exception as e:
                print(f"stopping the servo streamer failed: {e!r}", flush=True)
        (run / "log.json").write_text(json.dumps(log))
        print(f"log: {run} ({len(log)} ticks)", flush=True)
        STOP.unlink(missing_ok=True)
        print("returning to rest (raise, wrist first, then fold)", flush=True)
        try:
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
            try:
                print(f"servo position gain back to P {P_REST}: read back {CC.set_p_gain(arms, P_REST)}", flush=True)
            except Exception as e:
                print(f"restoring P gain failed ({e!r}); it resets to {P_REST} at the next power-off", flush=True)
            for arm in arms.values():
                arm.torque(False)
            print("rest, torque OFF", flush=True)
            for c in cams.values():
                c.stop.set()


if __name__ == "__main__":
    main()
