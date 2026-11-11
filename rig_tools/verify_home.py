"""Verification pose: both arms to SIM_HOME (same ramps as so101_real --pose-check), save RAW frames + measured state
in calib_fit layout, return to rest, torque off. Runs ON dzkchen with monitor.py stopped.

    python verify_home.py
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

CAMS = ("overhead", "left_wrist", "right_wrist")


def main():
    rig = json.loads((HERE / "rig.json").read_text())
    off = rig.get("joint_offsets_deg", {})
    arms = {s: R.Arm(s, rig["arms"][s], off.get(s)) for s in R.SIDES}
    for s, a in arms.items():
        if a.fingerprint != R.FINGERPRINTS[s]:
            raise SystemExit(f"{rig['arms'][s]} is not the {s} arm")
    cams = {n: R.Camera(rig["cameras"][n], rig.get("camera_backend", "dshow")) for n in CAMS}
    run = HERE / "runs" / f"verify_home_{time.strftime('%Y%m%d_%H%M%S')}" / "run_01"
    (run / "frames").mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps({"rig": rig, "source": "verify_home.py"}, indent=1))

    def units():
        return np.concatenate([arms[s].to_units(arms[s].read_raw()) for s in R.SIDES])

    def to_raw(u12):
        return {s: arms[s].to_raw(u12[i * 6:(i + 1) * 6]) for i, s in enumerate(R.SIDES)}

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
        tgt = to_raw(R.SIM_HOME)
        stage1 = {}
        for s in R.SIDES:
            mid = arms[s].read_raw().copy()
            mid[1], mid[2] = tgt[s][1], tgt[s][2]
            stage1[s] = np.round(mid).astype(int)
        ramp(stage1, 3.0)
        ramp(tgt, 2.5)
        time.sleep(1.5)
        st = units()
        for n in CAMS:
            Image.fromarray(cams[n].latest()).save(run / "frames" / f"c00_raw_{n}.jpg", quality=92)
        with open(run / "events.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "model_call", "chunk": 0, "state_in": [round(float(x), 3) for x in st],
                                "target": R.SIM_HOME.tolist()}) + "\n")
        print("target  " + " ".join(f"{v:6.1f}" for v in R.SIM_HOME))
        print("reached " + " ".join(f"{v:6.1f}" for v in st))
        print("saved", run)
        time.sleep(0.5)
    finally:
        try:
            back = {}
            for s in R.SIDES:
                mid = arms[s].read_raw().copy()
                rest = arms[s].neutral_raw()
                mid[3], mid[0], mid[4], mid[5] = rest[3], rest[0], rest[4], rest[5]
                back[s] = np.round(mid).astype(int)
            ramp(back, 2.5)
            ramp({s: arms[s].neutral_raw() for s in R.SIDES}, 3.0)
            time.sleep(0.5)
        finally:
            for a in arms.values():
                a.torque(False)
            print("rest, torque OFF")
            for c in cams.values():
                c.stop.set()


if __name__ == "__main__":
    main()
