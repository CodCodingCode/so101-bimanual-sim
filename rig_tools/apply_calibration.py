#!/usr/bin/env python3
"""Adopt the real-rig calibration (robots/so101/calibration) in the sims and on the robot PC -- cameras AND arm
kinematics together, so the sims match the real arms (user, 2026-11-13: "we need to adopt the real calibrations").

  python apply_calibration.py check     # build the training rig (needs the wrist refit in placement.json) and run the
                                        # cube check with the refit kinematics: PASS / FAIL per wrist
  python apply_calibration.py local     # (after a PASS) switch every sim of this repo: the rig becomes
                                        # data/so101_camera_rig.active.json (distill / gen_vla_sim / vla_sim_eval /
                                        # detector_policy load it by default) and the pedestal shifts go to
                                        # pi_sim/assets/so101_kinematics.json (pi_sim scene)
  python apply_calibration.py laptop    # stage the robot-PC update in data/laptop_calibration_update/ (nothing deployed):
                                        # rig.json joint_offsets_deg + refit offsets, the camera rig, the kinematics file

Kinematics file (written by the refit next to the placement): calibration/actuators/kinematics_refit.json
  {"left": {"joint_offsets_deg": [5], "pedestal_shift_cm": [3]}, "right": {...}}   sim = real reading + offset
The joint offsets go to the REAL side (rig.json), the pedestal shifts to the SIM scene (sim and the robot PC's shadow).
Everything trained before the switch was trained on the old kinematics and cameras: retrain after adopting.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROBOTS = HERE.parents[1]
CAL = ROBOTS / "so101" / "calibration"
KIN = CAL / "actuators" / "kinematics_refit.json"
SIM = HERE.parent / "pi_sim_so101"
ACTIVE = HERE / "data" / "so101_camera_rig.active.json"
CANDIDATE = HERE / "data" / "so101_camera_rig.calibration_folder.json"
RUNS = Path("/private/tmp/claude-501/-Users-owner-Downloads-whatif-sim/91ecc604-fde6-4144-892d-9cd4dc1fb82b/scratchpad/real_runs2")
AREA_OK, OFFSET_OK_PX = (0.8, 1.25), 10.0
PY = sys.executable


def build_candidate():
    sys.path.insert(0, str(HERE))
    import build_rig_from_calibration as B
    rig, notes = B.build(preview=False)
    CANDIDATE.write_text(json.dumps(rig, indent=1))
    return notes


def check():
    notes = build_candidate()
    print("\n".join(notes))
    cmd = [PY, "-W", "ignore", str(SIM / "squeeze" / "rig_cube_check.py"), "--rig", str(CANDIDATE), "--runs", str(RUNS)]
    if KIN.exists():
        cmd += ["--kinematics", str(KIN)]
    else:
        print(f"(no {KIN.name}: cameras only)")
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=SIM, env={**os.environ, "SQUEEZE_TASK": "clamp"})
    if out.returncode:
        print(out.stderr[-2000:])
    out = out.stdout
    print(out)
    ok = True
    for c in ("left_wrist", "right_wrist"):
        mt = re.search(rf"{c}: (\d+) frames.*?median ([\d.]+).*?offset median (\d+) px", out)
        if not mt:
            print(f"{c}: no result -> FAIL")
            ok = False
            continue
        n, area, off = int(mt.group(1)), float(mt.group(2)), float(mt.group(3))
        good = n >= 20 and AREA_OK[0] <= area <= AREA_OK[1] and off <= OFFSET_OK_PX
        ok &= good
        print(f"{c}: {n} frames, area x{area:.2f} (want {AREA_OK[0]}-{AREA_OK[1]}), offset {off:.0f} px (want <= "
              f"{OFFSET_OK_PX:.0f}) -> {'PASS' if good else 'FAIL'}")
    print("CHECK:", "PASS" if ok else "FAIL")
    return ok


def local():
    if not CANDIDATE.exists():
        raise SystemExit("run 'check' first")
    ACTIVE.write_text(CANDIDATE.read_text())
    kin_out = SIM / "pi_sim" / "assets" / "so101_kinematics.json"
    if KIN.exists():
        k = json.loads(KIN.read_text())
        kin_out.write_text(json.dumps({s: {"pedestal_shift_cm": k[s].get("pedestal_shift_cm")} for s in k if s in ("left", "right")},
                                      indent=1))
    print(f"sims switched: rig -> {ACTIVE}; pedestal shifts -> {kin_out if KIN.exists() else '(none)'}")
    print("clamp_wide (sim-74) reads its own RIG: tell it to point at the active rig too. Then retrain.")


def laptop():
    out = HERE / "data" / "laptop_calibration_update"
    out.mkdir(parents=True, exist_ok=True)
    if not CANDIDATE.exists():
        raise SystemExit("run 'check' first")
    (out / "so101_camera_rig.json").write_text(CANDIDATE.read_text())
    cur = json.loads((CAL / "actuators" / "joint_offsets.json").read_text())["joint_offsets_deg"]
    new = {s: list(v) for s, v in cur.items()}
    if KIN.exists():
        k = json.loads(KIN.read_text())
        for s in ("left", "right"):
            d = (k.get(s) or {}).get("joint_offsets_deg")
            if d:
                new[s] = [round(a + b, 3) for a, b in zip(cur[s][:5], d)] + list(cur[s][5:])
        (out / "so101_kinematics.json").write_text(json.dumps(
            {s: {"pedestal_shift_cm": k[s].get("pedestal_shift_cm")} for s in k if s in ("left", "right")}, indent=1))
    (out / "joint_offsets_new.json").write_text(json.dumps({"current": cur, "new": new}, indent=1))
    print(f"staged in {out} (NOT deployed):")
    for s in ("left", "right"):
        print(f"  rig.json joint_offsets_deg[{s}]: {cur[s]} -> {new[s]}")
    print("  camera rig -> C:\\robot\\pi\\pi_sim\\assets\\so101_camera_rig.json (runners' shadow + triangulation)")
    print("  kinematics -> C:\\robot\\real\\sim_mac\\pi_sim\\assets\\so101_kinematics.json (+ the new pi_sim/scene.py)")


if __name__ == "__main__":
    {"check": check, "local": local, "laptop": laptop}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
