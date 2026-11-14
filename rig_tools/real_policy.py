"""The learned RL clamp policy (an MLP, NOT the scripted IK plan) on the REAL SO-101 arms.

Policy: squeeze/runs_clamp_v3/detector_policy/best.pt, trained in sim by DAgger from the RL teacher; 100 % success,
98 % contact-free on 48 disturbed sim episodes. Every 40 ms it gets exactly what it was trained on:
  per arm: joint angles / 90, gripper opening, tool point (forward kinematics) x5, its own commanded tool point x5
  the cube reading (x / 5 cm, (y - nominal) / 5 cm) + "seen", its previous action, the clock (tick / 450).
  The cube reading is TRIANGULATED from all three cameras (cube_locator.py: red blob in the overhead and both wrist
  images; wrist camera poses from the measured joints), then converted to what the overhead detector would read
  (the policy's training input). 2026-11-12 first live run: the overhead view was blocked by the right arm from 2.4 s
  and its last, partly hidden reading was 1.2 cm off; both wrist cameras saw the cube the whole time.
and outputs a tool-point step per arm (x 0.5 x delta_scale, as in training). IK in the kinematic shadow turns that into
joint targets.

Same protection as the scripted runs: safety guard every tick (camera mounts / other arm / post / box / real joint
ranges / stuck joint / 10 deg lead limit), per-joint torque caps, servo gain P 32 for the run only (back to 16 after),
capped joint correction toward each target while the tips are > 3 cm from the cube (frozen after), floor guard,
STOP file. Any stop folds the arms back to rest, torque off. Trained speed (a learned policy cannot be slowed down).

    python real_policy.py --dry       # find the cube, run the policy once on the current state: NOTHING moves
    python real_policy.py --dry-loop  # the whole loop for --seconds, torque OFF, no goal written: timing only
    python real_policy.py --live

Real-time standard (2026-11-13, rt_control.py + so101_control.json; the old loop ran ~17 Hz, one snapped goal per
tick -> shaking): servo goals streamed at 100 Hz between policy commands; camera red-blob detection on its own
thread; frames saved on a thread; the policy step, IK rate limit, gentle-carry limit and the clock advance by
elapsed / 0.04 s; start / fold moves are minimum-jerk S-curves at 100 Hz, P 16 before the fold.
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

import mujoco                      # noqa: E402
import numpy as np                 # noqa: E402
import torch                       # noqa: E402
import torch.nn as nn              # noqa: E402
from PIL import Image              # noqa: E402

import clamp_control as CC        # noqa: E402  (servo gain)
import safety as SF                # noqa: E402
import so101_real as R             # noqa: E402
from real_clamp import camera_model   # noqa: E402  (unused now; kept for the same import side effects as before)
sys.path.insert(0, str(HERE / "so101_actuation"))
import tracking as TR              # noqa: E402
import cube_locator as CL          # noqa: E402
import rt_control as RT            # noqa: E402

STOP = HERE / "STOP"
SIDES = ("left", "right")
CAMS = ("overhead", "left_wrist", "right_wrist")
OBS_DIM = 34
ACTION_SCALE = 0.5
CLOCK_STEPS = 450
STEPS = 550                        # 22 s at 25 Hz (the clamp + place takes ~16 s)
POLICY_FILE = HERE / "squeeze_rt" / "detector_policy_v3.pt"
NEAR_CUBE = 0.03
SQUEEZE_MIN_GAP = 0.001            # m: the policy may not command the grippers closer than this (squeeze limiter)
# GENTLE CARRY (2026-11-12 drop): while the cube is held, the policy's forward / vertical action may change by at most
# HOLD_ACCEL per tick and stay within +-HOLD_SPEED. The real grip is weaker than the sim's (plastic on plastic); the cube
# tilted and fell out when the policy braked over the box. Sim with a weak grip (friction 0.5, squeeze 1/4): 2/12 ->
# 10/12 delivered; normal grip still 12/12. Squeezing (x) is untouched.
HOLD_ACCEL, HOLD_SPEED = 0.04, 0.35


def build_net():                   # identical to squeeze/detector_policy.py build_net()
    return nn.Sequential(nn.Linear(OBS_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 6), nn.Tanh())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--dry-loop", action="store_true", help="full loop timing test, torque off, nothing written")
    ap.add_argument("--floor-cm", type=float, default=-0.5)
    ap.add_argument("--seconds", type=float, default=STEPS / 25)
    # OFF by default: the policies are trained on the sim's fitted servo sag (it matches the real sag: lift / elbow
    # -1..-2.5 deg in both while holding); removing it grips ~7 mm higher than trained and the cube slips (sim-74,
    # 2026-11-12: corrector on, ideal servos 92% -> 46%). In the live runs so far it never engaged anyway.
    ap.add_argument("--correct", action="store_true", help="enable the joint TrackingCorrector (not recommended)")
    a = ap.parse_args()
    live = a.live and not a.dry and not a.dry_loop
    cfg = RT.load_config()
    print(f"control config {cfg['_path']}: policy {cfg['control']['hz']} Hz, servo stream {cfg['servo_stream']['hz']} Hz, "
          f"P {cfg['servo_gain']['run_p']} during the run", flush=True)
    torch.set_num_threads(2)

    import clamp_teacher as C
    import env as E
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    # the shadow needs the CALIBRATED camera rig: the wrist cameras' poses come from it + the measured joints
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
    _ = camera_model()                                             # (overhead back-projection now via cube_locator)

    net = build_net()
    net.load_state_dict(torch.load(POLICY_FILE, map_location="cpu"))
    net.eval()

    rig = json.loads((HERE / "rig.json").read_text())
    off = rig.get("joint_offsets_deg", {})
    arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in SIDES}
    for s, arm in arms.items():
        if arm.fingerprint != R.FINGERPRINTS[s]:
            raise SystemExit(f"{rig['arms'][s]} is not the {s} arm")
    cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}
    run = HERE / "runs" / f"real_policy_{time.strftime('%Y%m%d_%H%M%S')}"
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

    det = {"last": None, "seen": 0.0, "info": "", "pending": [], "since_tri": 99}
    JUMP, SMOOTH = 0.02, 0.4          # reject a reading > 2 cm from the estimate unless 3 in a row agree; smoothing
    rigj = json.loads(rig_cams.read_text())["cameras"]
    intr = {c: CL.cam_intrinsics(rigj[c]) for c in CAMS}

    def detect(frames=None, dets=None):
        """Triangulated cube from every camera that sees red -> the overhead-equivalent reading the policy expects.
        Uses the shadow's current joints for the wrist camera poses (call after to_shadow). dets: red blobs from the
        perception thread ({"seq", "red"}); the same seq again -> the last reading is held, not re-counted."""
        if dets is not None and dets.get("seq") == det.get("seq"):
            return det["last"], det["seen"]
        if dets is not None:
            det["seq"] = dets.get("seq")
        frames = frames or {c: cams[c].latest() for c in CAMS}
        rays, seen_by = [], []
        for c in CAMS:
            uv, n = dets["red"][c] if dets is not None else CL.detect_red(frames[c])
            if uv is not None:
                rays.append(CL.ray(intr[c], *CL.camera_pose(m, d, c), *uv))
                seen_by.append(f"{c}({n}px)")
        xyz, how = CL.locate(rays, prefer=next((i for i, s_ in enumerate(seen_by) if s_.startswith("overhead")), 0))
        oh_pose = CL.camera_pose(m, d, "overhead")
        # PLAUSIBILITY (2026-11-12 live run: while the cube was hidden in the grippers, red servo wires gave readings
        # at (+14, -12) / (-24, -18) cm and the arms wandered):
        #   - outside the table area the policy knows -> rejected
        #   - while the fingertips hold the cube (close together, around the current estimate) the cube must lie
        #     between them: readings > 2.5 cm from the fingertips' midpoint are rejected and the midpoint is used
        tl, tr = sim.robot["left"].position.copy(), sim.robot["right"].position.copy()
        mid = 0.5 * (tl + tr)
        mid_xy = CL.overhead_equivalent(mid, intr["overhead"], *oh_pose)
        holding = (np.linalg.norm(tl[:2] - tr[:2]) < 0.06 and det["last"] is not None and mid_xy is not None
                   and np.linalg.norm(mid_xy - det["last"]) < 0.035)
        if xyz is not None and not (-0.12 <= xyz[0] <= 0.12 and C.CUBE_Y - 0.16 <= xyz[1] <= C.CUBE_Y + 0.10):
            xyz, how = None, how + " (rejected: outside the table area)"
        if holding and (xyz is None or np.linalg.norm(xyz[:2] - mid[:2]) > 0.025):
            xyz, how = mid, "holding: cube taken between the fingertips" + ("" if xyz is None else " (reading rejected)")
            det["since_tri"] = 0
        xy = CL.overhead_equivalent(xyz, intr["overhead"], *oh_pose) if xyz is not None else None
        # FILTER (2026-11-12 live run: the reading hopped ~1 cm whenever cameras gained / lost the cube, 18 cm once on a
        # false wrist detection, and the closed-loop policy twitched with it). Prefer triangulated fixes; reject jumps
        # unless they persist (the cube really moved); smooth what is accepted. The policy was trained on a steady
        # reading with ~3 mm noise.
        tri = how.startswith("triangulated") or how.startswith("holding")
        det["since_tri"] = 0 if tri else det["since_tri"] + 1
        if xy is not None and not tri and det["since_tri"] < 8:
            xy, how = None, how + " (ignored: triangulation seen recently)"
        if xy is None:
            det["seen"] = 0.0
        elif det["last"] is None:
            det["last"], det["seen"] = np.array(xy, float), 1.0
        elif np.linalg.norm(xy - det["last"]) > JUMP:
            det["pending"].append(np.array(xy, float))
            pend = det["pending"][-3:]
            if len(pend) == 3 and max(np.linalg.norm(q - pend[-1]) for q in pend) < 0.01:
                det["last"], det["seen"], det["pending"] = np.mean(pend, axis=0), 1.0, []
                how += " (jump confirmed by 3 readings)"
            else:
                det["seen"] = 0.0
                how += " (jump rejected)"
        else:
            det["last"], det["seen"], det["pending"] = det["last"] + SMOOTH * (np.array(xy) - det["last"]), 1.0, []
        det["info"] = f"{how}; seen by {', '.join(seen_by) or 'none'}" + (
            f"; 3D ({xyz[0] * 100:+.1f}, {(xyz[1] - C.CUBE_Y) * 100:+.1f}, {xyz[2] * 100:.1f}) cm" if xyz is not None else "")
        return det["last"], det["seen"]

    def observe(prev, k, dets=None):
        xy, seen = detect(dets=dets)
        parts = []
        for i, s in enumerate(SIDES):
            arm = sim.robot[s]
            parts += [np.rad2deg(d.qpos[arm.qpos_ids]) / 90.0, [arm.gripper_opening], arm.position * 5.0,
                      sim._ee_targets[i] * 5.0]
        parts += [[xy[0] / 0.05, (xy[1] - C.CUBE_Y) / 0.05, seen], prev, [k / CLOCK_STEPS]]
        return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])

    def act(obs):
        with torch.no_grad():
            return net(torch.from_numpy(obs[None])).numpy()[0].astype(np.float32)

    time.sleep(2.0)                                                # let auto-exposure settle
    to_shadow(read_units())
    xy = None
    for _ in range(15):
        xy, _ = detect()
        if xy is not None:
            break
        time.sleep(0.2)
    Image.fromarray(cams["overhead"].latest()).save(run / "detect_overhead.jpg", quality=90)
    if xy is None:
        stop_cams()
        raise SystemExit("cube not found in the overhead camera")
    cx, cy = float(xy[0]), float(xy[1])
    dev = (cx * 100, (cy - C.CUBE_Y) * 100)
    print(f"cube located: {det['info']}", flush=True)
    print(f"cube at x {cx * 100:+.1f} cm, y {cy * 100:+.1f} cm; offset from nominal {dev[0]:+.1f}, {dev[1]:+.1f} cm "
          f"(trained +-1.5)", flush=True)
    if abs(dev[0]) > 1.5 or abs(dev[1]) > 1.5:
        if a.dry_loop:                                 # timing test only: torque never comes on
            print("WARNING: cube outside the trained area (dry loop continues: nothing moves)", flush=True)
        else:
            stop_cams()
            raise SystemExit("cube outside the trained area: move it into the green zone")
    for s in SIDES:
        sim.pad_quat[s] = C.clamp_quat(s, (cx, cy))                # as at reset in training
    box = (cx, cy + C.BOX_DY - 0.005)
    try:
        guard = SF.Guard(m, SF.real_limits(arms), box_xy=box)
    except SF.SafetyStop as e:
        stop_cams()
        raise SystemExit(f"SAFETY: {e}")
    print("safety guard ready (self-test OK); real joint ranges (deg): " + " | ".join(
        f"{s} " + " ".join(f"{lo:.0f}..{hi:.0f}" for lo, hi in zip(*guard.limits[s])) for s in SIDES), flush=True)
    (run / "meta.json").write_text(json.dumps({"cube": [cx, cy], "box": list(box), "live": live, "policy": POLICY_FILE.name},
                                              indent=1))

    if a.dry:
        u = read_units()
        to_shadow(u)
        for i, s in enumerate(SIDES):
            sim._ee_targets[i] = sim.robot[s].position.copy()
        obs = observe(np.zeros(6, np.float32), 0)
        t0 = time.perf_counter()
        outs = [act(obs) for _ in range(50)]
        ms = (time.perf_counter() - t0) / 50 * 1000
        print(f"MLP policy runs in {ms:.2f} ms per step (budget 40 ms); its inputs here: cube reading "
              f"({obs[24] * 5:+.1f}, {obs[25] * 5:+.1f}) cm seen={obs[26]:.0f}; first action (arms where they are now, "
              f"NOT the start pose) {np.round(outs[0], 2).tolist()}", flush=True)
        print("DRY: nothing moved", flush=True)
        stop_cams()
        return

    fold_log = []
    P_RUN, P_REST = cfg["servo_gain"]["run_p"], cfg["servo_gain"]["rest_p"]

    def ramp_raw(targets, secs, check_stop=True, log=None):
        """Scripted move: minimum-jerk S-curve at the stream rate (rt_control.smooth_ramp)."""
        return RT.smooth_ramp(arms, targets, cfg, secs, check_stop=STOP.exists if check_stop else None, log=log)

    def perceive():                                   # perception thread: red blobs in all three cameras
        f = {c: cams[c].latest() for c in CAMS}
        return {"frames": f, "red": {c: CL.detect_red(f[c]) for c in CAMS}}

    log, stream, perc, saver = [], None, None, None
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
        clock = RT.TickClock(cfg)
        saver = RT.FrameSaver(run / "frames", cfg)
        stream = RT.ServoStreamer(arms, cfg, dry=not live)
        stream.start(last_goal)
        print("at start pose: the MLP policy is driving" if live else "dry loop running", flush=True)
        corr = TR.TrackingCorrector(TR.load_profile(HERE / "so101_actuation" / "so101_profile.json"))
        prev, t0, k, kf, dry_stops, next_print = np.zeros(6, np.float32), time.perf_counter(), 0, 0.0, 0, 0.0
        while time.perf_counter() - t0 < a.seconds:
            if STOP.exists():
                raise KeyboardInterrupt("STOP file")
            tick = time.perf_counter()
            adv = clock.tick()
            u = read_units()
            to_shadow(u)
            t_read = time.perf_counter()
            res, seq, t_det = perc.get()
            obs = observe(prev, kf, dets=dict(res, seq=seq))
            action = np.clip(act(obs), -1, 1)
            tl, tr = sim.robot["left"].position, sim.robot["right"].position     # real fingertips (measured joints)
            holding_now = np.linalg.norm(tl[:2] - tr[:2]) < 0.06 and min(tl[2], tr[2]) > 0.02
            if holding_now:
                v, lv = action.reshape(2, 3).copy(), prev.reshape(2, 3)
                v[:, 1:] = np.clip(lv[:, 1:] + np.clip(v[:, 1:] - lv[:, 1:], -HOLD_ACCEL * adv, HOLD_ACCEL * adv),
                                   -HOLD_SPEED, HOLD_SPEED)
                action = v.reshape(-1)
            delta = action.reshape(2, 3) * ACTION_SCALE * sim.delta_scale * adv      # per training tick x ticks elapsed
            ee_before, ctrl_before = sim._ee_targets.copy(), {s: d.ctrl[sim.robot[s].actuator_ids].copy() for s in SIDES}

            def command(dl):
                sim._ee_targets[:] = ee_before
                c_ = {}
                for i, s in enumerate(SIDES):
                    arm = sim.robot[s]
                    sim._ee_targets[i] = np.clip(ee_before[i] + dl[i], sim._workspace_low[i], sim._workspace_high[i])
                    target, _ = sim._solve_ik(arm, sim._ee_targets[i], None)
                    cur = ctrl_before[s]
                    new = cur + np.clip(target - cur, -sim._rate_limit * adv, sim._rate_limit * adv)
                    d.ctrl[arm.actuator_ids] = new
                    c_[s] = np.r_[np.rad2deg(new), 0.0]
                return c_
            cmd = command(delta)
            # squeeze limiter: the policy squeezes by aiming its grippers into the cube; never let the COMMANDED
            # grippers come closer than SQUEEZE_MIN_GAP (drop the inward x part of this step, keep lift / carry)
            gap = guard.distances(cmd)["arm-arm"][0]
            limited = gap < SQUEEZE_MIN_GAP
            if limited:
                dl = delta.copy()
                dl[0, 0] = min(dl[0, 0], 0.0)          # left arm: no further +x (toward the right arm)
                dl[1, 0] = max(dl[1, 0], 0.0)          # right arm: no further -x
                cmd = command(dl)
                if guard.distances(cmd)["arm-arm"][0] < SQUEEZE_MIN_GAP:
                    cmd = command(np.zeros_like(delta) + np.c_[np.zeros(2), np.zeros(2), dl[:, 2]])   # only vertical
            near = min(np.linalg.norm(sim.robot[s].position[:2] - np.array([cx, cy])) for s in SIDES) < NEAR_CUBE
            if near:
                corr.freeze()
            expect = cmd
            if a.correct:
                cmd = corr.step(cmd, {s: u[s][:5] for s in SIDES}, active=not near)
            t_ctl = time.perf_counter()
            try:
                cmd = guard.step(u, cmd, expect=expect)   # raises SafetyStop -> fold to rest
            except SF.SafetyStop as e:
                if live:
                    raise
                dry_stops += 1                            # dry loop: the arms do not follow, the lead limit trips
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
            prev = action.astype(np.float32)
            t = time.perf_counter() - t0
            log.append({"t": round(t, 3), "adv": round(adv, 2), "det_age_ms": round((tick - t_det) * 1000, 1),
                        "ms": {"read": round((t_read - tick) * 1000, 1), "ctl": round((t_ctl - t_read) * 1000, 1),
                               "guard": round((t_guard - t_ctl) * 1000, 1)},
                        "action": action.round(3).tolist(),
                        "cube_reading": [round(float(obs[24]) * 5, 2), round(float(obs[25]) * 5, 2), int(obs[26])],
                        "cube_how": det["info"],
                        "units": {s: u[s].round(2).tolist() for s in SIDES}, "ee_target": sim._ee_targets.round(4).tolist(),
                        "cmd": {s: cmd[s].round(2).tolist() for s in SIDES},
                        "joint_corr_deg": {s: np.round(corr.corr[s], 2).tolist() for s in SIDES},
                        "squeeze_limited": bool(limited), "gentle_carry": bool(holding_now), "cmd_gap_mm": round(float(gap) * 1000, 2),
                        "closest_mm": {w: dict(v) for w, v in guard.closest.items()}})
            if t >= next_print:
                next_print += 1.0
                print(f"  t={t:4.1f}s  {1 / max(clock.period(), 1e-3):4.1f} Hz  [{det['info']}]  tool L {np.round(sim._ee_targets[0] * 100, 1).tolist()} cm  "
                      f"R {np.round(sim._ee_targets[1] * 100, 1).tolist()} cm | cube reading "
                      f"({obs[24] * 5:+.1f}, {obs[25] * 5:+.1f}) cm seen={obs[26]:.0f}"
                      f"{'  [correction frozen]' if corr.frozen else ''}{'  [squeeze limited]' if limited else ''}", flush=True)
            saver.maybe(t, res["frames"], f"t{k:04d}.jpg")
            k += 1
            kf += adv
            clock.sleep_rest(tick)
        if live:
            Image.fromarray(cams["overhead"].latest()).save(run / "end_overhead.jpg", quality=90)
        dts = np.diff([r["t"] for r in log]) * 1000 if len(log) > 2 else np.array([0.0])
        st = {w: round(float(np.median([r["ms"][w] for r in log])), 1) for w in ("read", "ctl", "guard")} if log else {}
        print(f"MLP policy finished: {len(log)} ticks, period median {np.median(dts):.0f} ms / p90 "
              f"{np.percentile(dts, 90):.0f} ms ({100 * np.mean(dts > 44):.0f} % over 40 ms); stage medians {st} ms; "
              f"perception {np.median(perc.ms):.0f} ms/frame-set"
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
            stop_cams()


if __name__ == "__main__":
    main()
