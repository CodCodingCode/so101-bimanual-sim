"""Scripted two-arm TIP CLAMP on the REAL SO-101 arms: clamp the cube with the fingertips (jaws shut), lift it level,
carry it to the grey box (between the arms and the cube), drop it in, back off apart. Runs ON dzkchen, monitor stopped.

Exactly the sim teacher (squeeze_rt/clamp_teacher.py, per-arm settings from squeeze_rt/runs/design_clamp2.json):
  - cube centre from the calibrated overhead camera (red blob -> table plane), box assumed at cube + BOX_DY
  - a kinematic shadow of the training sim (squeeze_rt/env.py SqueezeEnv) gives IK, workspace limits, rate limit
  - 25 Hz, the teacher's own speed cap (0.12 m/s); jaws commanded shut
    python real_clamp.py --dry     # find the cube, print the plan, NO motion
    python real_clamp.py --live    # run it
Safety: torque caps 70 % / 45 %, HandHeight guard (--floor-cm), joint rate limit, C:\\robot\\real\\STOP ends the run
and ALWAYS folds the arms back to rest before torque off. Logs to runs/real_clamp_<ts>/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(r"C:\robot\real")
RT = HERE / "squeeze_rt"
sys.path[:0] = [str(RT), str(HERE), r"C:\robot\pi"]
sys.meta_path[:] = [f for f in sys.meta_path if "__editable___pi_sim" not in getattr(f, "__module__", "")]
os.chdir(HERE)

import mujoco                      # noqa: E402
import numpy as np                 # noqa: E402
from PIL import Image              # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--floor-cm", type=float, default=-0.5)
    a = ap.parse_args()
    live = a.live and not a.dry

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
          f"offset from nominal {dev[0]:+.1f}, {dev[1]:+.1f} cm (trained +-2.5); box assumed at y {by * 100:+.1f} cm", flush=True)
    if abs(dev[0]) > 3.0 or abs(dev[1]) > 3.0:
        raise SystemExit("cube outside the trained area: move it into the green zone")
    for s in SIDES:
        sim.pad_quat[s] = C.clamp_quat(s, (cx, cy))
    plan = C.ClampPlan((cx, cy, 0.0), (bx, by))
    print("plan: " + " -> ".join(f"{n}({k})" for n, _, _, k in plan.waypoints), flush=True)
    (run / "meta.json").write_text(json.dumps({"cube": [cx, cy], "box": [bx, by], "rig": rig, "live": live,
                                               "params": C.P}, indent=1))
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

    def ramp_raw(targets, secs, check_stop=True):
        start = {s: arms[s].read_raw() for s in SIDES}
        n = int(secs * 50)
        for k in range(1, n + 1):
            if check_stop and STOP.exists():
                raise KeyboardInterrupt("STOP file")
            for s in SIDES:
                arms[s].goal(np.round(start[s] + (targets[s] - start[s]) * k / n).astype(int))
            time.sleep(0.02)

    log = []
    try:
        STOP.unlink(missing_ok=True)
        for arm in arms.values():
            arm.prepare()
            arm.goal(arm.read_raw().astype(int))
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
        print("at start pose, clamp begins", flush=True)
        k, last_phase, t0 = 0, None, time.perf_counter()
        while not plan.done:
            if STOP.exists():
                raise KeyboardInterrupt("STOP file")
            tick = time.perf_counter()
            u = read_units()
            to_shadow(u)
            if plan.phase != last_phase:
                print(f"  t={k / 25:5.2f}s {plan.phase}", flush=True)
                last_phase = plan.phase
            tg = plan.next_targets(sim._ee_targets.copy())
            cmd = {}
            for i, s in enumerate(SIDES):
                arm = sim.robot[s]
                sim._ee_targets[i] = np.clip(tg[i], sim._workspace_low[i], sim._workspace_high[i])
                target, _ = sim._solve_ik(arm, sim._ee_targets[i], None)
                cur = d.ctrl[arm.actuator_ids].copy()
                new = cur + np.clip(target - cur, -sim._rate_limit, sim._rate_limit)
                d.ctrl[arm.actuator_ids] = new
                cmd[s] = np.r_[np.rad2deg(new), 0.0]
            low = hh.lowest(np.r_[cmd["left"], cmd["right"]])
            blocked = [s for s in SIDES if low[s] * 100 < a.floor_cm]
            for s in SIDES:
                if s not in blocked:
                    arms[s].goal(arms[s].to_raw(cmd[s]))
            if blocked:
                print(f"  floor guard held {blocked} (lowest {min(low.values()) * 100:.1f} cm)", flush=True)
            log.append({"t": round(time.perf_counter() - t0, 3), "phase": plan.phase, "units": {s: u[s].round(2).tolist() for s in SIDES},
                        "ee_target": sim._ee_targets.round(4).tolist(), "cmd": {s: cmd[s].round(2).tolist() for s in SIDES}})
            if k % 5 == 0:
                Image.fromarray(np.concatenate([np.asarray(Image.fromarray(cams[c].latest()).resize((320, 240))) for c in CAMS], 1)) \
                    .save(run / "frames" / f"t{k:04d}_{plan.phase}.jpg", quality=80)
            k += 1
            time.sleep(max(0.0, 0.04 - (time.perf_counter() - tick)))
        Image.fromarray(cams["overhead"].latest()).save(run / "end_overhead.jpg", quality=90)
        print("clamp sequence finished", flush=True)
    except BaseException as e:
        print(f"stopping: {e!r}", flush=True)
    finally:
        (run / "log.json").write_text(json.dumps(log))
        print(f"log: {run} ({len(log)} ticks)", flush=True)
        STOP.unlink(missing_ok=True)
        print("returning to rest (raise, wrist first, then fold)", flush=True)
        try:
            back = {}
            for s in SIDES:
                uu = arms[s].to_units(arms[s].read_raw())
                back[s] = arms[s].to_raw(np.r_[uu[0], -45.8, 45.9, 90.0, uu[4], 0.0])
            ramp_raw(back, 3.0, check_stop=False)
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
