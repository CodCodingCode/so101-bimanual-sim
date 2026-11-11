"""Bring both arms back to the rest (folded) pose from wherever they are, then torque off. Runs ON dzkchen with
monitor.py stopped. Clears any STOP file first (it would otherwise abort the move).

  1. torque on, holding the current pose (no jump)
  2. raise: shoulder_lift / elbow / wrist_flex to the upright SIM_HOME values (pan, roll unchanged), 4 s
  3. pan / roll to SIM_HOME, 2 s
  4. rest: wrist, pan, roll, gripper to rest, then fold shoulder + elbow (the rig's go_rest order), 5.5 s
  5. torque off
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
import numpy as np        # noqa: E402
import so101_real as R    # noqa: E402

(HERE / "STOP").unlink(missing_ok=True)
rig = json.loads((HERE / "rig.json").read_text())
off = rig.get("joint_offsets_deg", {})
arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in R.SIDES}


def ramp(targets, secs):
    start = {s: arms[s].read_raw() for s in R.SIDES}
    n = int(secs * 50)
    for k in range(1, n + 1):
        for s in R.SIDES:
            arms[s].goal(np.round(start[s] + (targets[s] - start[s]) * k / n).astype(int))
        time.sleep(0.02)


try:
    for a in arms.values():
        a.prepare()
        a.goal(a.read_raw().astype(int))
        a.torque(True)
    print("torque on, holding", {s: np.round(arms[s].to_units(arms[s].read_raw()), 1).tolist() for s in R.SIDES}, flush=True)
    home = {s: arms[s].to_raw(R.SIM_HOME[i * 6:(i + 1) * 6]) for i, s in enumerate(R.SIDES)}
    up = {}
    for s in R.SIDES:
        r = arms[s].read_raw().copy()
        r[1], r[2], r[3] = home[s][1], home[s][2], home[s][3]
        up[s] = np.round(r).astype(int)
    ramp(up, 4.0)
    print("raised", flush=True)
    ramp(home, 2.0)
    print("at SIM_HOME", flush=True)
    mid = {}
    for s in R.SIDES:
        r = arms[s].read_raw().copy()
        rest = arms[s].neutral_raw()
        r[0], r[3], r[4], r[5] = rest[0], rest[3], rest[4], rest[5]
        mid[s] = np.round(r).astype(int)
    ramp(mid, 2.5)
    ramp({s: arms[s].neutral_raw() for s in R.SIDES}, 3.0)
    time.sleep(0.5)
    print("at rest", {s: np.round(arms[s].to_units(arms[s].read_raw()), 1).tolist() for s in R.SIDES}, flush=True)
finally:
    for a in arms.values():
        a.torque(False)
    print("torque OFF", flush=True)
