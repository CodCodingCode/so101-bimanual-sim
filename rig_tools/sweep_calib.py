"""Robot-driven calibration sweep for ONE arm: no hands in frame. Runs ON dzkchen (needs the COM ports + cameras,
so monitor.py must be stopped first).

Targets are poses a person already put the arm in by hand (record_calib.py run), so they are physically reachable,
plus wrist-roll / wrist-flex / gripper variants. Every target AND every interpolated step of the path to it is
checked with so101_real.HandHeight (the rig's MuJoCo model): the hand's lowest point must stay above --floor-cm.
Path between targets: raise (shoulder/elbow/wrist to the upright SIM_HOME values) -> turn pan/roll/gripper -> lower.
The other arm is opened read-only (its state is logged, torque never touched).

Output (calib_fit.py layout): runs/<name>/run_01/events.jsonl + frames/cKK_raw_{overhead,left_wrist,right_wrist}.jpg

    python sweep_calib.py --from-run runs\\calib_right_...\\run_01 --plan     # print the plan, no hardware
    python sweep_calib.py --from-run runs\\calib_right_...\\run_01 --live     # move the arm

Abort: create C:\\robot\\real\\STOP (or Ctrl+C): the arm returns to rest, torque off.
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
from PIL import Image                   # noqa: E402

import so101_real as R                  # noqa: E402

CAMS = ("overhead", "left_wrist", "right_wrist")
RAISED = np.array([-45.8, 45.9, 90.0])  # lift, elbow, wrist_flex of SIM_HOME: arm upright, hand high
STOP = HERE / "STOP"


def plan_targets(src_states, si, hh, other12, floor_m, max_n):
    """Distinct reachable targets for the arm at state index si (0 left, 6 right), with roll/flex/grip variety."""
    base = []
    for s in src_states:
        p = s[si:si + 6].astype(float)
        if p[1] < -90 and p[2] > 80:          # folded rest pose: nothing to learn
            continue
        if all(np.abs(p[:5] - q[:5]).max() >= 8 for q in base):
            base.append(p)
    out = []
    rolls, flexes, grips = (-170.0, -90.0, -10.0), (-30.0, 20.0, 70.0), (5.0, 60.0)
    for i, p in enumerate(base):
        for v in (p,
                  np.r_[p[:4], rolls[i % 3], grips[i % 2]],
                  np.r_[p[:3], flexes[i % 3], rolls[(i + 1) % 3], grips[(i + 1) % 2]]):
            v = v.copy()
            if all(np.abs(v[:5] - q[:5]).max() >= 8 for q in out) and lowest(hh, v, si, other12) >= floor_m:
                out.append(v)
    rng = np.random.default_rng(0)
    if len(out) > max_n:
        out = [out[i] for i in sorted(rng.choice(len(out), max_n, replace=False))]
    return out


def lowest(hh, p6, si, other12):
    s = other12.copy()
    s[si:si + 6] = p6
    return hh.lowest(s)["left" if si == 0 else "right"]


def waypoints(cur, tgt):
    """cur -> raised (keep pan/roll/grip) -> raised with target pan/roll/grip -> target."""
    up1 = cur.copy(); up1[1:4] = RAISED
    up2 = tgt.copy(); up2[1:4] = RAISED
    return [up1, up2, tgt.copy()]


def path_ok(hh, cur, tgt, si, other12, floor_m, n=12):
    a = cur
    for b in waypoints(cur, tgt):
        for f in np.linspace(0, 1, n):
            if lowest(hh, a + (b - a) * f, si, other12) < floor_m:
                return False
        a = b
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", required=True, help="record_calib.py run dir with hand-placed poses")
    ap.add_argument("--side", default="right", choices=("left", "right"))
    ap.add_argument("--floor-cm", type=float, default=6.0, help="min height of the hand's lowest point")
    ap.add_argument("--max-poses", type=int, default=30)
    ap.add_argument("--deg-s", type=float, default=30.0, help="joint speed cap during ramps")
    ap.add_argument("--plan", action="store_true", help="print the plan only, no hardware")
    ap.add_argument("--live", action="store_true", help="enable torque and move the arm")
    a = ap.parse_args()
    si = 0 if a.side == "left" else 6
    other = "left" if a.side == "right" else "right"
    floor_m = a.floor_cm / 100

    src = [np.array(json.loads(l)["state_in"], float) for l in open(Path(a.from_run) / "events.jsonl", encoding="utf-8")]
    other12 = src[0].copy()                                   # other arm as recorded at rest (only matters for logging)
    hh = R.HandHeight()
    targets = plan_targets(src, si, hh, other12, floor_m, a.max_poses)
    cur = src[0][si:si + 6].copy()
    plan = []
    for t in targets:
        if path_ok(hh, cur, t, si, other12, floor_m):
            plan.append(t)
            cur = t
    print(f"{len(plan)} poses planned (floor {a.floor_cm:.0f} cm):", flush=True)
    for k, p in enumerate(plan):
        print(f"  {k:2d}: " + " ".join(f"{v:7.1f}" for v in p) + f"   hand low {lowest(hh, p, si, other12) * 100:5.1f} cm", flush=True)
    if a.plan or not a.live:
        return

    rig = json.loads((HERE / "rig.json").read_text())
    off = rig.get("joint_offsets_deg", {})
    arm = R.Arm(a.side, rig["arms"][a.side], off.get(a.side))
    oth = R.Arm(other, rig["arms"][other], off.get(other))
    for x, s in ((arm, a.side), (oth, other)):
        if x.fingerprint != R.FINGERPRINTS[s]:
            raise SystemExit(f"{rig['arms'][s]} is not the {s} arm: fix rig.json")
    cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}

    name = f"calib_{a.side}_robot_{time.strftime('%Y%m%d_%H%M%S')}"
    run = HERE / "runs" / name / "run_01"
    (run / "frames").mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps({"rig": rig, "side": a.side, "source": "sweep_calib.py (robot-driven)",
                                               "floor_cm": a.floor_cm, "from_run": str(a.from_run)}, indent=1))
    ev = open(run / "events.jsonl", "a", encoding="utf-8")
    print(f"logging to {run}", flush=True)

    def units():
        out = np.zeros(12)
        out[si:si + 6] = arm.to_units(arm.read_raw())
        oi = 6 - si
        out[oi:oi + 6] = oth.to_units(oth.read_raw())
        return out

    def ramp(target_units):
        start = arm.to_units(arm.read_raw()).astype(float)
        delta = np.abs(np.asarray(target_units) - start)
        secs = max(1.2, float(delta[:5].max()) / a.deg_s, float(delta[5]) / 80)
        n = int(secs * 50)
        r0, r1 = arm.to_raw(start), arm.to_raw(target_units)
        for k in range(1, n + 1):
            if STOP.exists():
                raise KeyboardInterrupt("STOP file")
            arm.goal(np.round(r0 + (r1 - r0) * k / n).astype(int))
            time.sleep(0.02)

    def settle(timeout=3.0):
        t0, prev = time.time(), arm.to_units(arm.read_raw())
        still_since = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.1)
            u = arm.to_units(arm.read_raw())
            if np.abs(u[:5] - prev[:5]).max() > 0.3:
                still_since = time.time()
            prev = u
            if time.time() - still_since >= 0.5:
                return

    def go_rest():
        u = arm.to_units(arm.read_raw()).astype(float)
        up = u.copy(); up[1:4] = RAISED
        ramp(up)
        rest = arm.neutral_raw()
        mid = arm.read_raw().copy()
        mid[0], mid[3], mid[4], mid[5] = rest[0], rest[3], rest[4], rest[5]
        for tgt in (mid, rest):
            r0 = arm.read_raw()
            for k in range(1, 101):
                arm.goal(np.round(r0 + (tgt - r0) * k / 100).astype(int))
                time.sleep(0.025)
        time.sleep(0.4)

    try:
        STOP.unlink(missing_ok=True)
        arm.prepare()
        arm.goal(arm.read_raw().astype(int))                 # hold where it is before torque comes on
        arm.torque(True)
        print("torque ON (" + a.side + " arm only)", flush=True)
        for k, tgt in enumerate(plan):
            cur = arm.to_units(arm.read_raw()).astype(float)
            for wp in waypoints(cur, tgt):
                ramp(wp)
            settle()
            time.sleep(0.4)                                  # camera frames newer than the stop
            s = units()
            for n in CAMS:
                Image.fromarray(cams[n].latest()).save(run / "frames" / f"c{k:02d}_raw_{n}.jpg", quality=92)
            ev.write(json.dumps({"event": "model_call", "chunk": k, "state_in": [round(float(x), 3) for x in s],
                                 "target": [round(float(x), 2) for x in tgt]}) + "\n")
            ev.flush()
            err = np.abs(s[si:si + 5] - tgt[:5]).max()
            print(f"pose {k:2d}/{len(plan) - 1}: reached within {err:4.1f} deg", flush=True)
        print("sweep done, returning to rest", flush=True)
        go_rest()
    except BaseException as e:                                # incl. KeyboardInterrupt / STOP
        print(f"stopping early: {e!r}; returning to rest", flush=True)
        try:
            go_rest()
        except Exception as e2:
            print(f"go_rest failed: {e2!r}", flush=True)
    finally:
        arm.torque(False)
        print("torque OFF", flush=True)
        for c in cams.values():
            c.stop.set()
        arm.close()
        oth.close()


if __name__ == "__main__":
    main()
