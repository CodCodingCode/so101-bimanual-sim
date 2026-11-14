"""Joint-offset calibration from one robot-driven sweep (ik_sweep.py / map_sweep.py run), overhead silhouettes only.

Holds the DEPLOYED overhead camera fixed (lens from data/camera_fit.json on the CAD mount pose, as in pi_sim/assets/so101_camera_rig.json; CAD
232 mm spacing) and fits per-joint degree corrections (pan, lift, elbow, wrist_flex) for the sweep's moving arm, so
that the sim arm rendered from the logged joints overlaps the real arm in the overhead frame (IoU, same masks as
fit_overhead_232.py). Even frames fit, odd frames are held out; a correction is only worth deploying if the
held-out IoU rises too. Wrist roll is not observable from overhead and is left alone.

Nothing is deployed: prints the suggested rig.json joint_offsets_deg and writes <run>/calib_fit.json.

    pi_sim_so101/.venv/bin/python rig_tools/calib_from_sweep.py rig_tools/data/ik_left_<time>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_overhead_232 as F    # noqa: E402

JN = ("pan", "lift", "elbow", "wflex")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, help="run dir with events.jsonl, meta.json, frames/")
    ap.add_argument("--camera", type=Path, default=HERE / "data" / "camera_fit.json")
    ap.add_argument("--span", type=float, default=12.0, help="coarse scan +- degrees per joint")
    a = ap.parse_args()

    meta = json.loads((a.run / "meta.json").read_text())
    side = meta["side"]
    rig = meta["rig"]
    rig_file = a.run / "rig_at_record.json"
    rig_file.write_text(json.dumps(rig))
    C = F.Ctx([a.run], rig_file, None)
    n = len(C.states)
    fit_idx, held_idx = list(range(0, n, 2)), list(range(1, n, 2))
    x0 = np.zeros(22)
    x0[0:11] = json.loads(a.camera.read_text())["x"]
    x0[0:6] = 0.0          # DEPLOYED camera = that fit's lens on the CAD mount pose (so101_camera_rig.json), not its pose
    j0 = 12 if side == "left" else 17

    def x_of(c4):
        x = x0.copy()
        x[j0:j0 + 4] = np.asarray(c4) / 2.0                     # fit_overhead_232 units: 2 deg per unit
        return x

    def iou(c4, idx):
        return C.score(x_of(c4), idx)[0]

    base_fit, base_held = iou(np.zeros(4), fit_idx), iou(np.zeros(4), held_idx)
    print(f"{side} arm, {n} poses ({len(fit_idx)} fit / {len(held_idx)} held out); recorded with offsets "
          f"{rig['joint_offsets_deg'][side]}")
    print(f"  as recorded: IoU fit {base_fit:.4f}, held-out {base_held:.4f}", flush=True)

    c = np.zeros(4)
    best = base_fit
    for span, step in ((a.span, 2.0), (2.0, 0.5), (2.0, 0.5)):           # coordinate descent, coarse then fine
        for j in range(4):
            vals = c[j] + np.arange(-span, span + 1e-9, step)
            s = []
            for v in vals:
                t = c.copy(); t[j] = v
                s.append(iou(t, fit_idx))
            k = int(np.argmax(s))
            if s[k] > best:
                best, c[j] = s[k], vals[k]
        print(f"  pass (+-{span:g} step {step:g}): corr " + " ".join(f"{nm} {v:+.1f}" for nm, v in zip(JN, c))
              + f"  IoU fit {best:.4f}", flush=True)
    held = iou(c, held_idx)
    print(f"  fitted:      IoU fit {best:.4f}, held-out {held:.4f} (was {base_held:.4f})")

    # per-joint sensitivity on held-out frames around the fit: a real correction peaks near the same value here
    for j in range(4):
        vals = c[j] + np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        s = []
        for v in vals:
            t = c.copy(); t[j] = v
            s.append(iou(t, held_idx))
        print(f"  held-out scan {JN[j]:5s}: " + "  ".join(f"{v:+5.1f}:{q:.4f}" for v, q in zip(vals, s)))

    old = list(rig["joint_offsets_deg"][side])
    new = [round(o + d, 1) for o, d in zip(old[:4], c)] + old[4:]
    worth = held - base_held > 0.005
    print(f"\nsuggested {side} joint_offsets_deg: {new}  (now {old})")
    print("  held-out IoU improves by %.4f -> %s" % (held - base_held, "WORTH deploying" if worth else
                                                     "too small to matter: keep the current offsets"))
    out = dict(run=str(a.run), side=side, poses=n, camera=str(a.camera), corr_deg=dict(zip(JN, c.tolist())),
               iou=dict(recorded_fit=base_fit, recorded_held=base_held, fitted_fit=best, fitted_held=held),
               offsets_old=old, offsets_suggested=new, worth_deploying=bool(worth))
    (a.run / "calib_fit.json").write_text(json.dumps(out, indent=1))
    print("wrote", a.run / "calib_fit.json")


if __name__ == "__main__":
    main()
