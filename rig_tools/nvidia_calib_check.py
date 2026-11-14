"""NVIDIA SO-101 calibration check, run on our own bimanual rig (READ-ONLY: never moves or writes to the arms).

Implements the "Check Your Work" step of
https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/07-calibrating-so101.html
using NVIDIA's own so101_check_calibration.py (vendored in rig_tools/nvidia_calib/).

`lerobot-calibrate` stores its result in each servo's EEPROM: Homing_Offset (reg 31), Min/Max_Position_Limit
(regs 9/11). Our runners read those registers directly instead of a lerobot JSON. This tool:
  1. reads them over the Feetech bus (read instruction 0x02 only; no torque, no goal, no register writes);
  2. writes the equivalent lerobot calibration JSON per arm (<out>/<side>.json);
  3. runs NVIDIA's static_check (motion range vs their 15-robot stats, +-2 sigma; homing offset sanity) and
     NVIDIA's live_check (present encoder position inside the calibrated range) and prints their report.

It only reads, so the safety Guard (rig_tools/safety.py) does not apply: the Guard gates commands, and this sends none.
It needs the COM port free: stop the station / any runner first (it fails with "access denied" otherwise).

Run on the laptop:  python nvidia_calib_check.py [--arms left,right] [--rig C:\\robot\\real\\rig.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]  # servo ids 1..6
REG_MIN_LIMIT, REG_MAX_LIMIT, REG_HOMING, REG_PRESENT = 9, 11, 31, 56
READ = 0x02


class ReadOnlyBus:
    """Minimal Feetech STS reader. Deliberately has no write method."""

    def __init__(self, port: str):
        import serial
        self.sp = serial.Serial(port, 1_000_000, timeout=0.02)
        self.port = port

    def read(self, sid: int, addr: int, n: int, retries: int = 3) -> bytes:
        body = [sid, 4, READ, addr, n]
        pkt = bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])
        for _ in range(retries + 1):
            self.sp.reset_input_buffer()
            self.sp.write(pkt)
            raw = self.sp.read(6 + n)
            for i in range(len(raw) - 5 - n):
                if raw[i] == 0xFF and raw[i + 1] == 0xFF and raw[i + 2] == sid and raw[i + 3] == n + 2:
                    frame = raw[i + 2:i + 5 + n]
                    if ((~sum(frame)) & 0xFF) == raw[i + 5 + n]:
                        return raw[i + 5:i + 5 + n]
        raise IOError(f"{self.port}: servo {sid} did not answer (reg {addr})")

    def u16(self, sid: int, addr: int) -> int:
        b = self.read(sid, addr, 2)
        return b[0] | (b[1] << 8)

    # the one method NVIDIA's live_check() calls on robot.bus
    def sync_read(self, register: str, normalize: bool = False) -> dict[str, int]:
        assert register == "Present_Position" and not normalize
        return {j: self.u16(i + 1, REG_PRESENT) for i, j in enumerate(JOINTS)}

    def close(self):
        self.sp.close()


def sign_magnitude(v: int, sign_bit: int = 11) -> int:
    """Feetech Homing_Offset encoding (as lerobot decodes it)."""
    mag = v & ((1 << sign_bit) - 1)
    return -mag if v & (1 << sign_bit) else mag


def calibration_from_eeprom(bus: ReadOnlyBus) -> dict:
    cal = {}
    for i, j in enumerate(JOINTS):
        sid = i + 1
        cal[j] = dict(id=sid, drive_mode=0, homing_offset=sign_magnitude(bus.u16(sid, REG_HOMING)),
                      range_min=bus.u16(sid, REG_MIN_LIMIT), range_max=bus.u16(sid, REG_MAX_LIMIT))
    return cal


def load_nvidia():
    # NVIDIA's script targets lerobot 0.4.3; 0.6.x moved lerobot.robots.so101_follower. That import is only used
    # by their main() (which we don't call), so stub it rather than edit their file.
    try:
        import lerobot.robots.so101_follower  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("lerobot.robots.so101_follower")
        stub.SO101FollowerConfig = None
        sys.modules["lerobot.robots.so101_follower"] = stub
    path = HERE / "nvidia_calib" / "so101_check_calibration.py"
    spec = importlib.util.spec_from_file_location("nv_so101_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rig", default=r"C:\robot\real\rig.json")
    ap.add_argument("--arms", default="left,right")
    ap.add_argument("--out", default=str(HERE / "calib_lerobot"))
    ap.add_argument("--stats", default=str(HERE / "nvidia_calib" / "calibration_stats.json"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # their report prints σ ✓ ⚠ (Windows console is cp1252)

    nv = load_nvidia()
    ports = json.load(open(a.rig))["arms"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats_path = Path(a.stats)
    stats = nv.load_stats(stats_path)

    summary = {}
    for side in [s.strip() for s in a.arms.split(",") if s.strip()]:
        port = ports[side]
        print(f"\n##### {side} arm on {port}")
        try:
            bus = ReadOnlyBus(port)
        except Exception as e:  # noqa: BLE001
            print(f"  cannot open {port}: {e}\n  (is the station or another runner holding it?)")
            summary[side] = "NO PORT"
            continue
        try:
            cal_raw = calibration_from_eeprom(bus)
            calib_path = out / f"{side}.json"
            calib_path.write_text(json.dumps(cal_raw, indent=4))
            calibration = nv.load_calibration(calib_path)          # NVIDIA's loader (lerobot MotorCalibration)
            static = nv.static_check(calibration, stats)
            live = nv.live_check(type("R", (), {"bus": bus})(), calibration)
        finally:
            bus.close()
        nv.print_report(calib_path, stats_path, static, live)
        ok = all(r["status"] == "PASS" for r in static) and all(r["in_range"] for r in live)
        summary[side] = "PASS" if ok else "WARN"
    print("Summary:", ", ".join(f"{k}={v}" for k, v in summary.items()))
    return 0 if summary and all(v == "PASS" for v in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
