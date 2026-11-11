"""Record a calibration run for calib_fit.py from the live SO-101 monitor (no COM ports, no torque).

Runs ON dzkchen. Polls the monitor (http://127.0.0.1:8104) for joint state (so101_real units, rig offsets applied)
and the three camera streams, and keeps a sample only when the arm has been still for `--still-s` seconds, so the
person moving the arm by hand only has to pause in each pose. Output matches so101_real's run layout, which is what
calib_fit.py reads:

    runs/<out>/run_01/events.jsonl   {"event": "model_call", "chunk": k, "state_in": [12 units]} per kept pose
    runs/<out>/run_01/frames/cKK_raw_{overhead,left_wrist,right_wrist}.jpg
    runs/<out>/run_01/meta.json

    python record_calib.py --seconds 240 --side right
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

MON = "http://127.0.0.1:8104"
CAMS = ("overhead", "left_wrist", "right_wrist")
SIDES = ("left", "right")


def state():
    d = json.load(urllib.request.urlopen(f"{MON}/api/state", timeout=2))
    units = np.concatenate([np.asarray(d["arms"][s]["state"]["units"], float) for s in SIDES])
    torque = any(any(d["arms"][s]["state"]["torque"]) for s in SIDES)
    return units, torque


def frame(cam):
    """One JPEG from the monitor's MJPEG stream."""
    with urllib.request.urlopen(f"{MON}/cam/{cam}.mjpg?t={time.time()}", timeout=3) as r:
        buf = b""
        while True:
            chunk = r.read(16384)
            if not chunk:
                raise IOError(f"{cam}: stream ended")
            buf += chunk
            s = buf.find(b"\xff\xd8")
            e = buf.find(b"\xff\xd9", s + 2) if s >= 0 else -1
            if e > 0:
                return buf[s:e + 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=240)
    ap.add_argument("--side", default="right", choices=SIDES)
    ap.add_argument("--still-s", type=float, default=0.8, help="arm must be this still for this long")
    ap.add_argument("--still-deg", type=float, default=0.6, help="max joint change (deg) counted as still")
    ap.add_argument("--min-new-deg", type=float, default=6.0, help="a kept pose must differ this much from every kept pose")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    real = Path(r"C:\robot\real") if (Path(r"C:\robot\real") / "rig.json").exists() else here
    name = a.out or f"calib_{a.side}_{time.strftime('%Y%m%d_%H%M%S')}"
    run = real / "runs" / name / "run_01"
    (run / "frames").mkdir(parents=True, exist_ok=True)
    si = 0 if a.side == "left" else 6
    (run / "meta.json").write_text(json.dumps({"rig": json.loads((real / "rig.json").read_text()), "side": a.side,
                                               "source": "record_calib.py (monitor, hand-moved, torque off)"}, indent=1))
    ev = open(run / "events.jsonl", "a", encoding="utf-8")
    print(f"recording {a.side} arm for {a.seconds:.0f} s -> {run}", flush=True)

    hist, kept, k, t0 = [], [], 0, time.time()
    while time.time() - t0 < a.seconds:
        u, torque = state()
        if torque:
            print("torque is ON: not recording while something drives the arms", flush=True)
            time.sleep(1.0)
            continue
        now = time.time()
        hist = [(t, v) for t, v in hist if now - t <= a.still_s] + [(now, u)]
        span = now - hist[0][0]
        arm = np.array([v[si:si + 5] for _, v in hist])
        still = span >= a.still_s * 0.9 and np.ptp(arm, axis=0).max() <= a.still_deg
        new = all(np.abs(u[si:si + 5] - p[si:si + 5]).max() >= a.min_new_deg for p in kept)
        if still and new:
            jpgs = {c: frame(c) for c in CAMS}
            u2, _ = state()
            if np.abs(u2[si:si + 5] - u[si:si + 5]).max() <= a.still_deg:      # still the same pose after the grab
                for c, b in jpgs.items():
                    (run / "frames" / f"c{k:02d}_raw_{c}.jpg").write_bytes(b)
                ev.write(json.dumps({"event": "model_call", "chunk": k, "t": now - t0,
                                     "state_in": [round(float(x), 3) for x in u]}) + "\n")
                ev.flush()
                kept.append(u)
                print(f"pose {k:2d} kept at {now - t0:5.1f}s: " + " ".join(f"{x:7.1f}" for x in u[si:si + 6]), flush=True)
                k += 1
        time.sleep(0.1)
    print(f"done: {k} poses in {run}", flush=True)


if __name__ == "__main__":
    main()
