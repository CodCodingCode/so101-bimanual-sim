#!/usr/bin/env python3
"""Build a camera rig file (so101_camera_rig.json format, what pi_sim and the rig runners load) from the calibration
folder robots/so101/calibration/cameras/ -- the single place the camera calibration lives (README there, 2026-11-13).

  lens  cameras/lens_measured_2026-11-13.json  (ChArUco, OpenCV 5-coefficient; wrists good, overhead PROVISIONAL)
  pose  cameras/placement.json                 ('refit' slot, filled by inference-testing-04 once validated; 'current'
                                                 = the 11-10 poses, which were fitted WITH THE OLD (wrong) LENSES)

A lens and a pose must come from the same fit, so per camera:
  wrist     measured lens + refit pose       (training mode refuses if the refit is missing)
  overhead  refit lens+pose if present, else the CURRENT lens + pose unchanged (they were fitted together; the README:
            "refit the overhead position and angle together with its lens")

    python build_rig_from_calibration.py                  # training rig -> data/so101_camera_rig.calibration_folder.json
    python build_rig_from_calibration.py --preview        # measured wrist lenses on the OLD wrist poses: for the cube
                                                          # check only (shows what the refit has to fix), NOT for training
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAL = HERE.parents[1] / "so101" / "calibration" / "cameras"
OUT = HERE / "data"
CAMS = ("overhead", "left_wrist", "right_wrist")
CALIBRATION_TAG = "cams 6349a0 + kinematics bb1b28"   # pi_sim/cameras.py warns on rigs without it


def build(preview=False, cal=CAL):
    lens = json.loads((cal / "lens_measured_2026-11-13.json").read_text())["cameras"]
    place = json.loads((cal / "placement.json").read_text())
    base = json.loads((cal / "rig_in_use" / "so101_camera_rig.json").read_text())
    refit = place.get("refit") or {}
    cams, notes = {}, []
    for c in CAMS:
        old = base["cameras"][c]
        cur = place["current"][c]
        r = refit.get(c)
        entry = {k: old[k] for k in ("width", "height", "fps", "rotation")}
        entry["hardware_model"] = "InnoMaker U20CAM-1080P-S1 (640x480 = centre 4:3 crop of the 1080p sensor)"
        if c == "overhead":
            if r:
                src_lens = r.get("lens") or lens[c]
                entry["intrinsics"], entry["distortion"] = src_lens["intrinsics"], src_lens["distortion"]
                entry["pose"] = {"parent_body": r.get("parent_body", cur["parent_body"]), "pos": r["pos"],
                                 "quat_wxyz": r["quat_wxyz"]}
                notes.append("overhead: REFIT lens + pose")
            else:
                entry["intrinsics"], entry["distortion"] = old["intrinsics"], old["distortion"]
                entry["pose"] = old["pose"]
                notes.append("overhead: CURRENT lens + pose (fitted together on 11-10; measured lens still provisional)")
        else:
            entry["intrinsics"], entry["distortion"] = lens[c]["intrinsics"], lens[c]["distortion"]
            if r:
                entry["pose"] = {"parent_body": r.get("parent_body", cur["parent_body"]), "pos": r["pos"],
                                 "quat_wxyz": r["quat_wxyz"]}
                notes.append(f"{c}: measured lens + REFIT pose")
            elif preview:
                entry["pose"] = {"parent_body": cur["parent_body"], "pos": cur["pos"], "quat_wxyz": cur["quat_wxyz"]}
                notes.append(f"{c}: measured lens + OLD pose (PREVIEW: the old pose was fitted with the old lens)")
            else:
                raise SystemExit(f"{c}: placement.json has no refit yet -- the old pose was fitted with the old (wrong) "
                                 f"lens and must not be combined with the measured one for training. Use --preview for "
                                 f"the cube check only, or wait for inference-testing-04's refit.")
        entry["calibration_status"] = f"built from {cal} ({notes[-1]})"
        cams[c] = entry
    out = {"schema_version": base.get("schema_version", 1),
           "note": "Built by rig_tools/build_rig_from_calibration.py from robots/so101/calibration/cameras/ "
                   f"({'PREVIEW, not for training' if preview else 'training rig'}): " + "; ".join(notes),
           "cameras": cams}
    if any("REFIT" in n for n in notes):
        out["hide_wrist_mounts"] = True     # refit wrist poses sit inside the modelled mount (pi_sim/env.py)
        out["calibration_tag"] = CALIBRATION_TAG
    return out, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rig, notes = build(a.preview)
    out = Path(a.out) if a.out else OUT / ("so101_camera_rig.calibration_folder" + ("_PREVIEW" if a.preview else "") + ".json")
    out.write_text(json.dumps(rig, indent=1))
    print("\n".join(notes))
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
