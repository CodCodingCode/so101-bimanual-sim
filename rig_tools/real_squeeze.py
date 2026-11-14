"""Run the camera-student squeeze policy on the REAL bimanual SO-101. Runs ON dzkchen with monitor.py stopped.

The policy's world is reproduced exactly with a kinematic SHADOW of the training sim (squeeze.env.SqueezeEnv,
same pi_sim, same pad orientation, workspace limits, IK and joint rate limit): every 40 ms (25 Hz, the training rate)
  1. read both real arms -> shadow joint angles (no physics is stepped)
  2. grab the real overhead / left_wrist / right_wrist frames, resized to 64x64 exactly like pi_sim's render path
  3. proprio from the shadow (joint angles, gripper, tool point by forward kinematics, commanded tool point) + last action
  4. student -> 6 hand deltas -> the shadow's own ee_delta update + IK + rate limit -> joint targets -> real servos
Jaws are commanded shut, as in training. Start pose = training start: SIM_HOME, grippers open.

    python real_squeeze.py --dry               # NO motion: torque off, prints what the policy would do
    python real_squeeze.py --live --speed 0.5  # moves the arms at half the policy's speed
Safety: torque caps 70 % (gripper 45 %); the shadow's workspace clip (tool point >= 5 mm above the table);
HandHeight guard (a command whose hand dips below --floor-cm is not sent); per-tick joint rate limit; --seconds cap;
C:\\robot\\real\\STOP aborts; always ends with arms at rest and torque off. Logs to runs/real_squeeze_<ts>/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(r"C:\robot\real")
SQ = HERE / "squeeze_rt"
sys.path[:0] = [str(SQ), str(HERE), r"C:\robot\pi"]
sys.meta_path[:] = [f for f in sys.meta_path if "__editable___pi_sim" not in getattr(f, "__module__", "")]
os.chdir(HERE)

import mujoco                          # noqa: E402
import numpy as np                     # noqa: E402
import torch                           # noqa: E402
from PIL import Image                  # noqa: E402

import so101_real as R                 # noqa: E402
import rt_control as RT             # noqa: E402  (2026-11-13 motor rule: streamed goals, min-jerk ramps)

STOP = HERE / "STOP"
CAMS = ("overhead", "left_wrist", "right_wrist")
SIDES = ("left", "right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(SQ / "student.pt"))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--speed", type=float, default=0.5, help="scales the policy's hand deltas (live)")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--floor-cm", type=float, default=-0.3, help="HandHeight guard: lowest hand point (cm)")
    a = ap.parse_args()
    live = a.live and not a.dry

    import distill as D
    import env as E
    from rl_env import ACTION_SCALE
    shadow = E.SqueezeEnv()
    shadow.reset(seed=0)
    sim, m, d = shadow.sim, shadow.model, shadow.data
    d.joint("cube_joint").qpos[:3] = [2, 2, -1]           # the shadow has no cube / box: those are only in the cameras
    d.joint("can_joint").qpos[:3] = [2.5, 2, -1]
    student = D.build_student()
    student.load_state_dict(torch.load(a.model, map_location="cpu"))
    student.eval()
    hh = R.HandHeight()

    rig = json.loads((HERE / "rig.json").read_text())
    off = rig.get("joint_offsets_deg", {})
    arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in SIDES}
    for s, arm in arms.items():
        if arm.fingerprint != R.FINGERPRINTS[s]:
            raise SystemExit(f"{rig['arms'][s]} is not the {s} arm")
    cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}
    run = HERE / "runs" / f"real_squeeze_{time.strftime('%Y%m%d_%H%M%S')}"
    (run / "frames").mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps({"args": vars(a), "rig": rig}, indent=1))
    glo, ghi = m.actuator_ctrlrange[sim.robot["left"].gripper_actuator]

    def read_units():
        return {s: arms[s].to_units(arms[s].read_raw()).astype(float) for s in SIDES}

    def to_shadow(u):
        for s in SIDES:
            arm = sim.robot[s]
            d.qpos[arm.qpos_ids] = np.deg2rad(u[s][:5])
            d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = glo + np.clip(u[s][5], 0, 100) / 100 * (ghi - glo)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

    def frames():
        return np.stack([np.asarray(Image.fromarray(cams[c].latest()).resize((D.IMG, D.IMG), Image.Resampling.BILINEAR))
                         for c in CAMS]).astype(np.uint8)

    def prop(prev):
        parts = []
        for i, s in enumerate(SIDES):
            arm = sim.robot[s]
            parts += [np.rad2deg(d.qpos[arm.qpos_ids]) / 90.0, [arm.gripper_opening], arm.position * 5.0, sim._ee_targets[i] * 5.0]
        parts += [prev]
        return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])

    cfg = RT.load_config()

    def ramp_raw(targets, secs, check_stop=True):
        """Scripted move: minimum-jerk S-curve at the stream rate (rt_control.smooth_ramp)."""
        return RT.smooth_ramp(arms, targets, cfg, secs, check_stop=STOP.exists if check_stop else None)

    log = []
    try:
        STOP.unlink(missing_ok=True)
        if live:
            for arm in arms.values():
                arm.prepare()
                arm.goal(arm.read_raw().astype(int))
                arm.torque(True)
            home = {s: arms[s].to_raw(R.SIM_HOME[i * 6:(i + 1) * 6]) for i, s in enumerate(SIDES)}
            stage1 = {}
            for s in SIDES:
                mid = arms[s].read_raw().copy()
                mid[1], mid[2] = home[s][1], home[s][2]
                stage1[s] = np.round(mid).astype(int)
            ramp_raw(stage1, 3.0)
            ramp_raw(home, 2.5)
            time.sleep(0.8)
            print("at the training start pose (SIM_HOME, grippers open); policy starts", flush=True)
        else:
            print("DRY RUN: torque off, nothing is sent; the arms stay where they are", flush=True)
        u = read_units()
        to_shadow(u)
        for i, s in enumerate(SIDES):
            arm = sim.robot[s]
            sim._ee_targets[i] = arm.position.copy()
            d.ctrl[arm.actuator_ids] = d.qpos[arm.qpos_ids]
        ee0 = sim._ee_targets.copy()
        prev = np.zeros(6, np.float32)
        clock = RT.TickClock(cfg)
        stream = RT.ServoStreamer(arms, cfg, dry=not live)
        stream.start({s: arms[s].read_raw().astype(int) for s in SIDES})
        t0 = time.perf_counter()
        k = 0
        while time.perf_counter() - t0 < a.seconds:
            if STOP.exists():
                raise KeyboardInterrupt("STOP file")
            tick = time.perf_counter()
            adv = clock.tick()
            u = read_units()
            to_shadow(u)
            imgs, p = frames(), prop(prev)
            with torch.no_grad():
                act = student(D.to_tensor(imgs[None], "cpu"), torch.from_numpy(p[None])).numpy()[0].astype(np.float32)
            scaled = act * (a.speed if live else 1.0)
            cmd = {}
            for i, s in enumerate(SIDES):
                arm = sim.robot[s]
                row = scaled[i * 3:(i + 1) * 3] * ACTION_SCALE
                sim._ee_targets[i] = np.clip(sim._ee_targets[i] + row * sim.delta_scale * adv, sim._workspace_low[i], sim._workspace_high[i])
                target, ik_err = sim._solve_ik(arm, sim._ee_targets[i], sim._orientations[i])
                cur = d.ctrl[arm.actuator_ids].copy()
                new = cur + np.clip(target - cur, -sim._rate_limit * adv, sim._rate_limit * adv)
                d.ctrl[arm.actuator_ids] = new
                cmd[s] = np.r_[np.rad2deg(new), 0.0]                      # jaws shut, as in training
            low = hh.lowest(np.r_[cmd["left"], cmd["right"]])
            blocked = [s for s in SIDES if low[s] * 100 < a.floor_cm]
            for s in SIDES:                                       # streamed at 100 Hz (writes nothing when dry)
                if s in blocked:
                    stream.hold(s)
                else:
                    stream.set(s, arms[s].to_raw(cmd[s]), clock.period())
            if blocked:
                print(f"  t={k / 25:5.2f}s floor guard held {blocked} (lowest {min(low.values()) * 100:.1f} cm)", flush=True)
            log.append({"t": round(time.perf_counter() - t0, 3), "units": {s: u[s].round(2).tolist() for s in SIDES},
                        "action": act.round(4).tolist(), "ee_target": sim._ee_targets.round(4).tolist(),
                        "cmd": {s: cmd[s].round(2).tolist() for s in SIDES}, "blocked": blocked,
                        "loop_ms": round((time.perf_counter() - tick) * 1000, 1)})
            if k % 5 == 0:
                Image.fromarray(np.concatenate(list(imgs), 1)).resize((D.IMG * 3 * 4, D.IMG * 4), Image.Resampling.NEAREST) \
                    .save(run / "frames" / f"t{k:04d}.png")
            if k % 25 == 0:
                dee = (sim._ee_targets - ee0) * 100
                print(f"  t={k / 25:4.1f}s action L {np.round(act[:3], 2)} R {np.round(act[3:], 2)} | target moved "
                      f"L {np.round(dee[0], 1)} R {np.round(dee[1], 1)} cm | loop {log[-1]['loop_ms']} ms", flush=True)
            prev = act
            k += 1
            clock.sleep_rest(tick)
        print("time limit reached", flush=True)
    except BaseException as e:
        print(f"stopping: {e!r}", flush=True)
    finally:
        try:
            stream.stop()                                     # MUST stop before the fold writes goals
        except Exception:
            pass
        (run / "log.json").write_text(json.dumps(log))
        print(f"log: {run} ({len(log)} ticks)", flush=True)
        if live:
            STOP.unlink(missing_ok=True)       # STOP means: end the policy and fold back, never "go limp mid-air"
            print("returning to rest (raise, wrist first, then fold)", flush=True)
            try:
                back = {}
                for s in SIDES:
                    raw = arms[s].read_raw().copy()
                    up = arms[s].to_raw(np.r_[arms[s].to_units(raw)[0], -45.8, 45.9, 90.0, arms[s].to_units(raw)[4], 0.0])
                    back[s] = up
                ramp_raw(back, 2.5, check_stop=False)
                rest = {s: arms[s].neutral_raw() for s in SIDES}
                mid = {}
                for s in SIDES:
                    r = arms[s].read_raw().copy()
                    r[0], r[3], r[4], r[5] = rest[s][0], rest[s][3], rest[s][4], rest[s][5]
                    mid[s] = np.round(r).astype(int)
                ramp_raw(mid, 2.5, check_stop=False)
                ramp_raw(rest, 3.0, check_stop=False)
                time.sleep(0.4)
            except Exception as e:
                print(f"return to rest failed: {e!r}", flush=True)
            finally:
                for arm in arms.values():
                    arm.torque(False)
                print("rest, torque OFF", flush=True)
        for c in cams.values():
            c.stop.set()


if __name__ == "__main__":
    main()
