"""Servo dynamics test for system identification: both arms, scripted joint-space commands at 50 Hz, logging the
commanded and measured position of all 12 joints every tick. Runs ON dzkchen with monitor.py stopped.

Schedule (both arms at once, same direction so they stay parallel), all from SIM_HOME (upright, hand high):
  per arm joint (pan, lift, elbow, wrist_flex, wrist_roll): steps +15 / 0 / -15 / 0 deg (1.2 s each), then
      sine sweeps of 10 deg at 0.5, 1 and 2 Hz (2 cycles each)
  gravity holds: 4 extended poses, ramped in over 2 s and held 2 s
  gripper (no object): 100 -> 0 -> 100 -> 50 -> 100 (1 s each)
Every target is checked with so101_real.HandHeight (the rig's MuJoCo model) against --floor-cm before moving.

Output: runs/dyn_<ts>/dyn.npz with t (s), cmd (N,12) and meas (N,12) in rig units (deg incl. rig.json offsets,
gripper 0..100), loop_ms (N), phase (N, int) + phase_names; meta.json with rig + servo torque limits.

    python dyn_test.py --plan      # print schedule length + clearance, no hardware
    python dyn_test.py --live
Abort: create C:\\robot\\real\\STOP: arms return to rest, torque off.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(r"C:\robot\real")
sys.path[:0] = [str(HERE), r"C:\robot\pi"]
sys.meta_path[:] = [f for f in sys.meta_path if "__editable___pi_sim" not in getattr(f, "__module__", "")]
os.chdir(HERE)

import numpy as np                      # noqa: E402

import so101_real as R                  # noqa: E402
import rt_control as RT   # noqa: E402  (min-jerk ramps at 100 Hz, 2026-11-13 motor rule)

DT = 0.02
STOP = HERE / "STOP"
HOME = R.SIM_HOME.astype(float).copy()                 # 12 values, both arms
HOLDS = [  # one arm's (pan, lift, elbow, wflex, roll, grip); mirrored pan for the right arm
    (0.0, -20.5, 16.8, 10.9, -2.8, 100.0),
    (0.0, 6.4, -14.4, 20.0, -2.8, 100.0),
    (15.0, -21.4, -1.7, 40.8, -2.8, 100.0),
    (0.0, -65.8, 66.8, 40.8, -2.8, 100.0),
]


def both(p6):
    """One arm's pose -> 12 values. Pan is NOT mirrored: both arms turn the same way, so they stay parallel
    and never close in on each other (mirrored pan swings both hands toward the middle at once)."""
    p = np.asarray(p6, float)
    return np.r_[p, p.copy()]


BASE = both(HOLDS[0])                                   # step / sine centre pose (hand ~21 cm up)


def schedule():
    """List of (phase_name, (N,12) command array at 50 Hz)."""
    out = []

    def hold(name, u, secs):
        out.append((name, np.repeat(np.asarray(u, float)[None], int(round(secs / DT)), 0)))

    def ramp(name, a, b, secs):
        n = int(round(secs / DT))
        f = np.linspace(0, 1, n + 1)[1:, None]
        out.append((name, a + (b - a) * f))

    hold("settle_home", HOME, 1.0)
    ramp("to_base", HOME, BASE, 2.0)          # arm forward, hand ~21 cm up: room for +-15 deg on every joint
    for j, jn in enumerate(("pan", "lift", "elbow", "wflex", "roll")):
        sign = np.zeros(12)
        sign[j], sign[6 + j] = 1.0, 1.0                 # same direction on both arms (parallel, see both())
        for v in (15.0, 0.0, -15.0, 0.0):
            hold(f"step_{jn}_{v:+.0f}", BASE + sign * v, 1.2)
        for hz in (0.5, 1.0, 2.0):
            n = int(round(2 / hz / DT))
            t = np.arange(1, n + 1) * DT
            out.append((f"sine_{jn}_{hz}Hz", BASE + sign[None] * (10.0 * np.sin(2 * np.pi * hz * t))[:, None]))
        hold(f"rest_{jn}", BASE, 0.6)
    last = BASE
    for i, h in enumerate(HOLDS):
        tgt = both(h)
        ramp(f"to_hold{i}", last, tgt, 2.0)
        hold(f"hold{i}", tgt, 2.0)
        last = tgt
    ramp("to_base2", last, BASE, 2.0)
    for g in (0.0, 100.0, 50.0, 100.0):
        u = BASE.copy()
        u[5] = u[11] = g
        hold(f"grip_{g:.0f}", u, 1.0)
    ramp("to_home", BASE, HOME, 2.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--allow-steps", action="store_true",
                    help="REQUIRED with --live: this test commands deliberate +-15 deg step responses (system id)")
    ap.add_argument("--floor-cm", type=float, default=4.0)
    a = ap.parse_args()
    if a.live and not a.allow_steps:
        # user rule 2026-11-13: no policy or script may drive the servos in a way that harms the motors. This test's
        # +-15 deg instantaneous steps are exactly that by design, so it is locked unless explicitly unlocked.
        raise SystemExit("dyn_test --live commands instantaneous +-15 deg steps (harsh on the motors) and is locked; "
                         "add --allow-steps only if a new system-id run is really needed")

    sched = schedule()
    allcmd = np.concatenate([c for _, c in sched])
    hh = R.HandHeight()
    low = {s: min(hh.lowest(u)[s] for u in allcmd[::5]) for s in R.SIDES}
    print(f"{len(sched)} phases, {len(allcmd)} ticks = {len(allcmd) * DT:.0f} s; lowest hand point "
          f"left {low['left'] * 100:.1f} cm, right {low['right'] * 100:.1f} cm (floor {a.floor_cm:.0f} cm)", flush=True)
    if min(low.values()) * 100 < a.floor_cm:
        raise SystemExit("ABORT: a scheduled pose goes below the floor")
    if a.plan or not a.live:
        return

    rig = json.loads((HERE / "rig.json").read_text())
    off = rig.get("joint_offsets_deg", {})
    arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in R.SIDES}
    for s, arm in arms.items():
        if arm.fingerprint != R.FINGERPRINTS[s]:
            raise SystemExit(f"{rig['arms'][s]} is not the {s} arm")
    run = HERE / "runs" / f"dyn_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps({"rig": rig, "dt": DT, "torque_limit": 700, "grip_limit": 450,
                                               "source": "dyn_test.py"}, indent=1))

    def units():
        return np.concatenate([arms[s].to_units(arms[s].read_raw()) for s in R.SIDES])

    def send(u12):
        for i, s in enumerate(R.SIDES):
            arms[s].goal(arms[s].to_raw(u12[i * 6:(i + 1) * 6]))

    def ramp_raw(targets, secs):
        start = {s: arms[s].read_raw() for s in R.SIDES}
        n = int(secs * 100)
        for k in range(1, n + 1):
            for s in R.SIDES:
                arms[s].goal(np.round(start[s] + (targets[s] - start[s]) * RT.min_jerk(k / n)).astype(int))
            time.sleep(0.01)

    T, CMD, MEAS, LOOP, PH = [], [], [], [], []
    names = [n for n, _ in sched]
    try:
        STOP.unlink(missing_ok=True)
        for arm in arms.values():
            arm.prepare()
            arm.goal(arm.read_raw().astype(int))
            arm.torque(True)
        tgt = {s: arms[s].to_raw(HOME[i * 6:(i + 1) * 6]) for i, s in enumerate(R.SIDES)}
        stage1 = {}
        for s in R.SIDES:
            mid = arms[s].read_raw().copy()
            mid[1], mid[2] = tgt[s][1], tgt[s][2]
            stage1[s] = np.round(mid).astype(int)
        ramp_raw(stage1, 3.0)
        ramp_raw(tgt, 2.5)
        print("at SIM_HOME, starting schedule", flush=True)
        t0 = time.perf_counter()
        k = 0
        for pi, (name, cmds) in enumerate(sched):
            for c in cmds:
                if STOP.exists():
                    raise KeyboardInterrupt("STOP file")
                tick = time.perf_counter()
                send(c)
                m = units()
                T.append(tick - t0)
                CMD.append(c)
                MEAS.append(m)
                LOOP.append((time.perf_counter() - tick) * 1000)
                PH.append(pi)
                k += 1
                time.sleep(max(0.0, t0 + k * DT - time.perf_counter()))
            if name.startswith(("rest_", "hold")):
                err = np.abs(np.asarray(MEAS[-1]) - np.asarray(CMD[-1]))
                print(f"  {name:12s} end error max {err[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]].max():5.1f} deg | "
                      f"loop {np.mean(LOOP[-len(cmds):]):4.1f} ms", flush=True)
        print("schedule done", flush=True)
    except BaseException as e:
        print(f"stopping early: {e!r}", flush=True)
    finally:
        if T:
            np.savez(run / "dyn.npz", t=np.array(T), cmd=np.array(CMD), meas=np.array(MEAS), loop_ms=np.array(LOOP),
                     phase=np.array(PH), phase_names=np.array(names))
            print(f"saved {len(T)} ticks -> {run / 'dyn.npz'}", flush=True)
        try:
            back = {}
            for s in R.SIDES:
                mid = arms[s].read_raw().copy()
                rest = arms[s].neutral_raw()
                mid[3], mid[0], mid[4], mid[5] = rest[3], rest[0], rest[4], rest[5]
                back[s] = np.round(mid).astype(int)
            ramp_raw(back, 2.5)
            ramp_raw({s: arms[s].neutral_raw() for s in R.SIDES}, 3.0)
            time.sleep(0.4)
        finally:
            for arm in arms.values():
                arm.torque(False)
            print("rest, torque OFF", flush=True)


if __name__ == "__main__":
    main()
