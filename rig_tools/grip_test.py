"""Gripper force / friction test on the RIGHT arm with the real cube. Runs ON dzkchen with monitor.py stopped.

  1. right arm -> SIM_HOME -> forward test pose (hand ~21 cm up), gripper fully open; left arm untouched (torque off)
  2. waits for C:\\robot\\real\\GO (a person holds the cube between the open jaws, then GO is created)
  3. closes the gripper (100 -> 0 over 1 s) and holds 3 s; the person lets go once it has the cube
  4. shakes: pan +-10 deg at 1 Hz and 2 Hz (2 cycles each), lift +-10 deg steps; photos before / after
  5. opens the gripper (the person catches the cube), returns to rest, torque off

Logs every 20 ms: t, cmd (6), pos (6, rig units), right gripper load % (signed), phase. Output runs/grip_<ts>/.
Abort any time: create C:\\robot\\real\\STOP.
"""
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
from PIL import Image                   # noqa: E402

import so101_real as R                  # noqa: E402

DT = 0.02
STOP, GO = HERE / "STOP", HERE / "GO"
BASE = np.array([0.0, -20.5, 16.8, 10.9, -2.8, 100.0])      # right arm test pose (hand ~21 cm up)


def signed(v, bit):
    return -(v & ((1 << bit) - 1)) if v & (1 << bit) else v


def main():
    rig = json.loads((HERE / "rig.json").read_text())
    off = rig.get("joint_offsets_deg", {})
    arm = R.Arm("right", rig["arms"]["right"], off.get("right"))
    if arm.fingerprint != R.FINGERPRINTS["right"]:
        raise SystemExit("COM port is not the right arm")
    hh = R.HandHeight()
    probe = np.r_[R.SIM_HOME[:6], BASE]
    for d_pan, d_lift in ((10, 0), (-10, 0), (0, 10), (0, -10)):
        q = probe.copy(); q[6] += d_pan; q[7] += d_lift
        if hh.lowest(q)["right"] < 0.05:
            raise SystemExit("ABORT: shake pose below 5 cm")
    cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in ("overhead", "right_wrist")}
    run = HERE / "runs" / f"grip_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps({"rig": rig, "grip_limit": 450, "torque_limit": 700, "dt": DT}, indent=1))

    T, CMD, POS, LOAD, PH = [], [], [], [], []
    t0 = time.perf_counter()

    def read():
        rows = [arm.bus.read(i, R.REG_PRESENT, 6) for i in range(1, 7)]
        raw = np.array([R.u16(b, 0) for b in rows], float)
        return arm.to_units(raw), signed(R.u16(rows[5], 4), 10) / 10.0

    def tick(cmd6, phase):
        if STOP.exists():
            raise KeyboardInterrupt("STOP file")
        t = time.perf_counter()
        arm.goal(arm.to_raw(cmd6))
        pos, load = read()
        T.append(t - t0); CMD.append(np.array(cmd6, float)); POS.append(pos); LOAD.append(load); PH.append(phase)
        time.sleep(max(0.0, DT - (time.perf_counter() - t)))

    def ramp(a, b, secs, phase):
        n = int(secs / DT)
        for k in range(1, n + 1):
            tick(a + (b - a) * k / n, phase)

    def hold(c, secs, phase):
        for _ in range(int(secs / DT)):
            tick(c, phase)

    def snap(tag):
        for n, c in cams.items():
            Image.fromarray(c.latest()).save(run / f"{tag}_{n}.jpg", quality=90)

    try:
        STOP.unlink(missing_ok=True)
        GO.unlink(missing_ok=True)
        arm.prepare()
        arm.goal(arm.read_raw().astype(int))
        arm.torque(True)
        cur = arm.to_units(arm.read_raw()).astype(float)
        home = R.SIM_HOME[6:].astype(float)
        mid = cur.copy(); mid[1], mid[2] = home[1], home[2]
        ramp(cur, mid, 3.0, 0)
        ramp(mid, home, 2.5, 0)
        ramp(home, BASE, 2.0, 0)
        print("READY: gripper open at the test pose. Hold the cube between the jaws, then create GO.", flush=True)
        t_wait = time.time()
        while not GO.exists():
            if time.time() - t_wait > 600:
                raise TimeoutError("no GO within 10 min")
            hold(BASE, 0.5, 1)
        print("GO: closing", flush=True)
        closed = BASE.copy(); closed[5] = 0.0
        ramp(BASE, closed, 1.0, 2)
        hold(closed, 3.0, 3)
        snap("gripped")
        print(f"gripped: jaw {POS[-1][5]:.1f} % open, load {LOAD[-1]:+.1f} %", flush=True)
        for hz, ph in ((1.0, 4), (2.0, 5)):
            n = int(round(2 / hz / DT))
            for k in range(1, n + 1):
                c = closed.copy(); c[0] += 10 * np.sin(2 * np.pi * hz * k * DT)
                tick(c, ph)
        for v in (10.0, -10.0, 0.0):
            c = closed.copy(); c[1] += v
            hold(c, 1.0, 6)
        snap("after_shake")
        print(f"after shake: jaw {POS[-1][5]:.1f} % open, load {LOAD[-1]:+.1f} %", flush=True)
        ramp(closed, BASE, 0.8, 7)
        hold(BASE, 2.0, 7)
        print("released", flush=True)
    except BaseException as e:
        print(f"stopping early: {e!r}", flush=True)
    finally:
        if T:
            np.savez(run / "grip.npz", t=np.array(T), cmd=np.array(CMD), pos=np.array(POS), load=np.array(LOAD),
                     phase=np.array(PH))
            print("saved", run, flush=True)
        try:
            raw = arm.read_raw().copy()
            rest = arm.neutral_raw()
            mid = raw.copy(); mid[0], mid[3], mid[4], mid[5] = rest[0], rest[3], rest[4], rest[5]
            for tgt in (mid, rest):
                r0 = arm.read_raw()
                for k in range(1, 126):
                    arm.goal(np.round(r0 + (tgt - r0) * k / 125).astype(int))
                    time.sleep(0.02)
            time.sleep(0.4)
        finally:
            arm.torque(False)
            GO.unlink(missing_ok=True)
            print("rest, torque OFF", flush=True)
            for c in cams.values():
                c.stop.set()


if __name__ == "__main__":
    main()
