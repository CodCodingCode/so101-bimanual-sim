"""Validate joint offsets on frames the fit never saw (map_left / map_right sweeps, 100 poses each).

Camera, arm spacing and base poses come from a fit json (fit_overhead_232.py). For the moving arm of each run we
report the mean IoU with no correction (the offsets the run was recorded with), with candidate corrections, and a
1-D scan per joint, everything else held fixed. A correction that is real should peak near the same value here.

    sim/pi_sim_so101/.venv/bin/python rig_tools/validate_offsets.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_overhead_232 as F    # noqa: E402

FIT = HERE / "data" / "joint_fit_base" / "fit232_both.json"
JN = ("pan", "lift", "elbow", "wflex", "roll")


def main():
    res = json.loads(FIT.read_text())
    x = np.array(res["x"])
    for side, run, cands in (
        ("right", HERE / "data" / "map_right", {"recorded (old: 0,0,0,0)": [0, 0, 0, 0, 0],
                                                "DEPLOYED (+4.0,-0.9,-6.7,-1.2)": [4.0, -0.9, -6.7, -1.2, 0]}),
        ("left", HERE / "data" / "map_left", {"recorded (current left)": [0, 0, 0, 0, 0],
                                              "fit (+13 lift)": res["joint_corr_deg"]["left"],
                                              "rest-stop check (lift -5.9)": [0, -5.9, 0, 0, 0]})):
        C = F.Ctx([run], HERE / "data" / "rig.json", None)
        si = 0 if side == "left" else 6

        def iou(corr5):
            xx = x.copy()
            xx[12:22] = 0.0                                     # both arms: no correction beyond the run's offsets
            if side == "left":
                xx[12:17] = np.asarray(corr5) / 2.0
            else:
                xx[17:22] = np.asarray(corr5) / 2.0
            return C.score(xx)[0]

        print(f"\n== {side} arm on {run.name} ({len(C.states)} unseen frames)")
        for name, c in cands.items():
            print(f"  {name:34s} IoU {iou(c):.4f}")
        base = np.zeros(5)
        for j, n in enumerate(JN[:4]):
            vals = np.arange(-14, 15, 2.0)
            s = [iou(np.where(np.arange(5) == j, v, base)) for v in vals]
            k = int(np.argmax(s))
            print(f"  scan {n:5s}: best {vals[k]:+5.1f} deg (IoU {s[k]:.4f}; at 0: {s[7]:.4f})  " +
                  " ".join(f"{v:+.0f}:{q:.3f}" for v, q in zip(vals[::2], s[::2])))


if __name__ == "__main__":
    main()
