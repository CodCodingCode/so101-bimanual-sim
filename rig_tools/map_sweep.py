"""Robot-driven sweep of BOTH arms for real -> sim mapping: still poses all over each arm's half of the table, three
cameras + all 12 joints logged at every pose, so each pose can be rendered in the sim and paired with the real frames.
Runs ON dzkchen (needs the COM ports + cameras: stop monitor.py first). Built on sweep_calib.py and keeps its safety:
  - one arm moves at a time; the other arm's torque is never touched (it is only read);
  - every target and every interpolated step of every path is checked with so101_real.HandHeight (the rig's MuJoCo
    model): the hand's lowest point stays above --floor-cm, and the moving hand stays on its own side of the centre
    line (--midline-cm), so the arms cannot meet;
  - joint speed capped at --deg-s; abort with C:\\robot\\real\\STOP or Ctrl+C -> arm folds to rest, torque off.
Targets come from poses the arms already reached safely in logged runs (hand-placed calibration runs and policy
runs, incl. reaching for the cube), plus wrist-roll / wrist-flex / gripper variants and blends of two such poses.
Output (calib_fit.py layout, so the same run also calibrates the joints/cameras):
  runs/map_<side>_<time>/run_01/events.jsonl  {"event": "model_call", "chunk": k, "side", "state_in": [12], "objects"}
  runs/map_<side>_<time>/run_01/frames/cKK_raw_{overhead,left_wrist,right_wrist}.jpg

    python map_sweep.py --plan                        # print both arms' plans; no hardware
    python map_sweep.py --side left --live            # move the left arm (the user starts this, hand on the e-stop)
    python map_sweep.py --side right --live
Place the red cube and the grey box where they will be during runs before starting; the overhead camera locates the
cube at every pose (logged under "objects").
"""
import argparse
import csv
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
import rt_control as RT   # noqa: E402  (min-jerk ramps at 100 Hz, 2026-11-13 motor rule)

CAMS = ("overhead", "left_wrist", "right_wrist")
RAISED = np.array([-45.8, 45.9, 90.0])  # lift, elbow, wrist_flex of SIM_HOME: arm upright, hand high
STOP = HERE / "STOP"


def logged_states(runs_root, max_runs):
    """12-value states from logged runs: events.jsonl state_in, and steps.csv measured joints (every 25th row)."""
    out = []
    runs = sorted(runs_root.rglob("run_*"), key=lambda p: p.stat().st_mtime)[-max_runs:]
    for run in runs:
        ev = run / "events.jsonl"
        if ev.exists():
            for line in open(ev, encoding="utf-8"):
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if "state_in" in e and len(e["state_in"]) == 12:
                    out.append(np.asarray(e["state_in"], float))
        st = run / "steps.csv"
        if st.exists():
            rows = list(csv.DictReader(open(st, encoding="utf-8")))
            names = [f"meas_{s}_{j}" for s in ("L", "R") for j in R.JOINTS]
            for r in rows[::25]:
                try:
                    out.append(np.asarray([float(r[n]) for n in names], float))
                except (KeyError, ValueError):
                    continue
    return out, [str(r) for r in runs]


class Guard:
    def __init__(self, hh, si, rest12, floor_m, mid_m):
        self.hh, self.si, self.rest12, self.floor_m, self.mid_m = hh, si, rest12, floor_m, mid_m
        self.side = "left" if si == 0 else "right"

    def full(self, p6):
        s = self.rest12.copy()
        s[self.si:self.si + 6] = p6
        return s

    def ok(self, p6):
        s = self.full(p6)
        if self.hh.lowest(s)[self.side] < self.floor_m:
            return False
        x = self.hh.grip_point(s, self.side)[0]
        return x <= -self.mid_m if self.side == "left" else x >= self.mid_m

    def low(self, p6):
        return self.hh.lowest(self.full(p6))[self.side]


def plan_targets(states, g, max_n, rng):
    si = g.si
    base = []
    for s in states:
        p = s[si:si + 6].astype(float)
        if p[1] < -90 and p[2] > 80:          # folded rest pose
            continue
        if all(np.abs(p[:5] - q[:5]).max() >= 6 for q in base) and g.ok(p):
            base.append(p)
    rolls, flexes, grips = (-150.0, -80.0, -10.0, 60.0), (-30.0, 20.0, 70.0), (5.0, 50.0, 95.0)
    cand = list(base)
    for i, p in enumerate(base):
        cand.append(np.r_[p[:4], rolls[i % 4], grips[i % 3]])
        cand.append(np.r_[p[:3], flexes[i % 3], rolls[(i + 2) % 4], grips[(i + 1) % 3]])
    for _ in range(4 * len(base)):            # blends of two reached poses: new places between known-safe ones
        if len(base) < 2:
            break
        a, b = rng.choice(len(base), 2, replace=False)
        f = rng.uniform(0.25, 0.75)
        v = base[a] + f * (base[b] - base[a])
        v[5] = rng.choice(grips)
        cand.append(v)
    out = []
    for v in cand:
        if all(np.abs(v[:5] - q[:5]).max() >= 6 for q in out) and g.ok(v):
            out.append(v)
    if len(out) > max_n:
        out = [out[i] for i in sorted(rng.choice(len(out), max_n, replace=False))]
    # greedy nearest-neighbour order (shorter moves)
    order, left = [], list(range(len(out)))
    cur = np.zeros(6)
    while left:
        j = min(left, key=lambda k: np.abs(out[k][:5] - cur[:5]).max())
        order.append(out[j])
        cur = out[j]
        left.remove(j)
    return order, len(base)


def waypoints(cur, tgt, g):
    """direct when every step of the straight joint path is safe, else up (keep pan/roll/grip) -> turn -> down."""
    if all(g.ok(cur + (tgt - cur) * f) for f in np.linspace(0, 1, 16)):
        return [tgt.copy()]
    up1 = cur.copy(); up1[1:4] = RAISED
    up2 = tgt.copy(); up2[1:4] = RAISED
    return [up1, up2, tgt.copy()]


def path_ok(g, cur, tgt, n=12):
    a = cur
    for b in waypoints(cur, tgt, g):
        for f in np.linspace(0, 1, n):
            if g.low(a + (b - a) * f) < g.floor_m:
                return False
        a = b
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="both", choices=("left", "right", "both"))
    ap.add_argument("--floor-cm", type=float, default=None, help="min hand height (default rig table_floor_cm + 4)")
    ap.add_argument("--midline-cm", type=float, default=3.0, help="moving hand stays this far on its own side of x=0")
    ap.add_argument("--max-poses", type=int, default=100)
    ap.add_argument("--max-runs", type=int, default=40, help="newest logged runs to take reached poses from")
    ap.add_argument("--deg-s", type=float, default=30.0)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    rig = json.loads((HERE / "rig.json").read_text())
    floor_cm = a.floor_cm if a.floor_cm is not None else float(rig.get("table_floor_cm", 1.0)) + 4.0
    states, runs = logged_states(HERE / "runs", a.max_runs)
    rest12 = np.asarray(R.SIM_HOME, float).copy()
    rest_like = [s for s in states if s[1] < -90 and s[2] > 80 and s[7] < -90 and s[8] > 80]
    if rest_like:
        rest12 = rest_like[0].copy()
    hh = R.HandHeight()
    rng = np.random.default_rng(0)
    sides = ("left", "right") if a.side == "both" else (a.side,)
    plans = {}
    for side in sides:
        si = 0 if side == "left" else 6
        g = Guard(hh, si, rest12, floor_cm / 100, a.midline_cm / 100)
        targets, nbase = plan_targets(states, g, a.max_poses, rng)
        cur = rest12[si:si + 6].copy()
        plan = []
        for t in targets:
            if path_ok(g, cur, t):
                plan.append(t)
                cur = t
        plans[side] = (plan, g)
        est = sum(max(1.2, np.abs(b[:5] - c[:5]).max() / a.deg_s) for c, b in zip([rest12[si:si + 6]] + plan[:-1], plan)) + 2.0 * len(plan)
        print(f"{side}: {len(plan)} poses from {nbase} reached poses in {len(runs)} runs; floor {floor_cm:.1f} cm, "
              f"midline {a.midline_cm:.0f} cm; about {est / 60:.0f} min", flush=True)
        if a.plan:
            for k, p in enumerate(plan[:12]):
                print(f"  {k:3d}: " + " ".join(f"{v:7.1f}" for v in p) + f"   hand low {g.low(p) * 100:5.1f} cm", flush=True)
            if len(plan) > 12:
                print(f"  ... {len(plan) - 12} more", flush=True)
    if a.plan or not a.live:
        return
    if a.side == "both":
        raise SystemExit("--live moves one arm per run: use --side left, then --side right")
    side = a.side
    plan, g = plans[side]
    si = g.si
    other = "left" if side == "right" else "right"
    off = rig.get("joint_offsets_deg", {})
    arm = R.Arm(side, rig["arms"][side], off.get(side))
    oth = R.Arm(other, rig["arms"][other], off.get(other))
    for x, s in ((arm, side), (oth, other)):
        if x.fingerprint != R.FINGERPRINTS[s]:
            raise SystemExit(f"{rig['arms'][s]} is not the {s} arm: fix rig.json")
    cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}
    name = f"map_{side}_{time.strftime('%Y%m%d_%H%M%S')}"
    run = HERE / "runs" / name / "run_01"
    (run / "frames").mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps({"rig": rig, "side": side, "source": "map_sweep.py (robot-driven)",
                                               "floor_cm": floor_cm, "midline_cm": a.midline_cm, "poses": len(plan),
                                               "from_runs": runs}, indent=1))
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
        n = int(secs * 100)
        r0, r1 = arm.to_raw(start), arm.to_raw(target_units)
        for k in range(1, n + 1):
            if STOP.exists():
                raise KeyboardInterrupt("STOP file")
            arm.goal(np.round(r0 + (r1 - r0) * RT.min_jerk(k / n)).astype(int))
            time.sleep(0.01)

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
            for k in range(1, 251):
                arm.goal(np.round(r0 + (tgt - r0) * RT.min_jerk(k / 250)).astype(int))
                time.sleep(0.01)
        time.sleep(0.4)

    try:
        STOP.unlink(missing_ok=True)
        arm.prepare()
        arm.goal(arm.read_raw().astype(int))
        arm.torque(True)
        print(f"torque ON ({side} arm only)", flush=True)
        t_start = time.time()
        for k, tgt in enumerate(plan):
            cur = arm.to_units(arm.read_raw()).astype(float)
            for wp in waypoints(cur, tgt, g):
                ramp(wp)
            settle()
            time.sleep(0.4)
            s = units()
            imgs = {n: cams[n].latest() for n in CAMS}
            for n in CAMS:
                Image.fromarray(imgs[n]).save(run / "frames" / f"c{k:03d}_raw_{n}.jpg", quality=92)
            try:
                found = R.track_objects(R.preprocess("overhead", imgs["overhead"], rig))
                objects = {kk: found.get(kk) for kk in ("cube", "cube_yaw_deg", "cube_px")}
            except Exception as e:  # tracking is a bonus; never stop the sweep for it
                objects = {"error": repr(e)}
            ev.write(json.dumps({"event": "model_call", "chunk": k, "side": side, "t": round(time.time() - t_start, 2),
                                 "state_in": [round(float(x), 3) for x in s],
                                 "target": [round(float(x), 2) for x in tgt], "objects": objects}) + "\n")
            ev.flush()
            err = np.abs(s[si:si + 5] - tgt[:5]).max()
            print(f"pose {k:3d}/{len(plan) - 1}: reached within {err:4.1f} deg", flush=True)
        print("sweep done, returning to rest", flush=True)
        go_rest()
    except BaseException as e:
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
