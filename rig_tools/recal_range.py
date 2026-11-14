"""Widen a few joints' calibrated range on one real SO-101 arm without changing its units (runs ON the laptop).

Why: the NVIDIA calibration check (nvidia_calib_check.py) flagged left elbow_flex / wrist_flex ranges as short — the
original lerobot-calibrate sweep stopped before the mechanical end stops (the folded rest pose even sits past the
recorded elbow max). A full lerobot-calibrate would re-home every joint and change the arm's units, invalidating
rig.json offsets, so instead:

  record  — torque must be OFF. Watches the chosen joints while a person moves them BY HAND to both end stops.
            Starts the clock at the first motion, records `--seconds`, saves calib_lerobot/recal_<side>_latest.json.
            Read-only (no bus writes at all).
  apply   — only WIDENS: new_min = min(old_min, seen_min), new_max = max(old_max, seen_max); refuses if a side grows
            by more than --max-grow counts (implausible) or the sweep obviously did not reach both ends. Writes ONLY
            Min/Max_Position_Limit (+ the EEPROM Lock register) on those servos, with torque off, reads them back,
            then shifts rig.json joint_offsets_deg[side][j] by (new_center - old_center)*360/4095 so that
            units = (raw - center)*360/4095 + offset is unchanged for every pose. Backs up rig.json first.
  restore — writes the old limits from the record file back and restores the rig.json backup.

Homing offsets are not touched, so raw encoder readings for a given pose do not change.

No motion is ever commanded (no torque/goal writes); writes are whitelisted to regs 9, 11, 55.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]  # ids 1..6
REG_MIN, REG_MAX, REG_TORQUE, REG_LOCK, REG_PRESENT = 9, 11, 40, 55, 56
WRITABLE = {REG_MIN, REG_MAX, REG_LOCK}
K = 360 / 4095


class Bus:
    def __init__(self, port):
        import serial
        self.port, self.sp = port, serial.Serial(port, 1_000_000, timeout=0.02)

    def _txn(self, sid, instr, params, n):
        body = [sid, len(params) + 2, instr, *params]
        pkt = bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])
        for _ in range(4):
            self.sp.reset_input_buffer()
            self.sp.write(pkt)
            raw = self.sp.read(6 + n)
            for i in range(len(raw) - 5 - n):
                if raw[i] == 0xFF and raw[i + 1] == 0xFF and raw[i + 2] == sid and raw[i + 3] == n + 2:
                    frame = raw[i + 2:i + 5 + n]
                    if ((~sum(frame)) & 0xFF) == raw[i + 5 + n]:
                        return raw[i + 5:i + 5 + n]
        raise IOError(f"{self.port}: servo {sid} did not answer (instr {instr}, reg {params[0]})")

    def u16(self, sid, addr):
        b = self._txn(sid, 0x02, [addr, 2], 2)
        return b[0] | (b[1] << 8)

    def u8(self, sid, addr):
        return self._txn(sid, 0x02, [addr, 1], 1)[0]

    def write(self, sid, addr, data):
        assert addr in WRITABLE, f"refusing to write register {addr}"
        self._txn(sid, 0x03, [addr, *data], 0)

    def close(self):
        self.sp.close()


def open_arm(rig, side):
    bus = Bus(rig["arms"][side])
    torque = [bus.u8(i, REG_TORQUE) for i in range(1, 7)]
    if any(torque):
        bus.close()
        sys.exit(f"{side}: torque is ON {torque} — bring the arm to rest / torque off first. Nothing done.")
    return bus


def cmd_record(a, rig, rec_path):
    ids = {j: JOINTS.index(j) + 1 for j in a.joints}
    bus = open_arm(rig, a.side)
    try:
        old = {j: dict(min=bus.u16(s, REG_MIN), max=bus.u16(s, REG_MAX)) for j, s in ids.items()}
        start = {j: bus.u16(s, REG_PRESENT) for j, s in ids.items()}
        seen = {j: dict(min=v, max=v) for j, v in start.items()}
        print(f"{a.side} torque off. Calibrated now: " + ", ".join(f"{j} {o['min']}..{o['max']}" for j, o in old.items()))
        print(f"Waiting up to {a.wait:.0f} s for the first hand motion, then recording {a.seconds:.0f} s.", flush=True)
        t0, t_move, last = time.time(), None, 0.0
        while True:
            now = time.time()
            for j, s in ids.items():
                p = bus.u16(s, REG_PRESENT)
                seen[j]["min"], seen[j]["max"] = min(seen[j]["min"], p), max(seen[j]["max"], p)
                if t_move is None and abs(p - start[j]) > 40:
                    t_move = now
                    print(f"motion seen on {j} — recording {a.seconds:.0f} s", flush=True)
            if t_move is None and now - t0 > a.wait:
                print("no motion seen; nothing recorded")
                return 2
            if t_move is not None and now - t_move > a.seconds:
                break
            if now - last > 3:
                last = now
                print("  " + "   ".join(
                    f"{j}: {seen[j]['min']}..{seen[j]['max']} (cal {old[j]['min']}..{old[j]['max']})" for j in ids),
                    flush=True)
            time.sleep(0.01)
    finally:
        bus.close()
    rec = dict(side=a.side, port=rig["arms"][a.side], time=time.strftime("%Y-%m-%d %H:%M:%S"),
               joints={j: dict(id=ids[j], old_min=old[j]["min"], old_max=old[j]["max"],
                               seen_min=seen[j]["min"], seen_max=seen[j]["max"]) for j in ids})
    rec_path.write_text(json.dumps(rec, indent=2))
    print("\nRecorded:")
    for j, r in rec["joints"].items():
        print(f"  {j:12s} calibrated {r['old_min']}..{r['old_max']} (range {r['old_max'] - r['old_min']})   "
              f"swept {r['seen_min']}..{r['seen_max']} (range {r['seen_max'] - r['seen_min']})")
    print("saved", rec_path)
    return 0


def plan(rec, max_grow):
    out, problems = {}, []
    for j, r in rec["joints"].items():
        lo, hi = min(r["old_min"], r["seen_min"]), max(r["old_max"], r["seen_max"])
        g_lo, g_hi = r["old_min"] - lo, hi - r["old_max"]
        if max(g_lo, g_hi) > max_grow:
            problems.append(f"{j}: grows {g_lo}/{g_hi} counts (> {max_grow}) — suspicious, refusing")
        old_span = r["old_max"] - r["old_min"]
        if (r["seen_max"] - r["seen_min"]) < 0.8 * old_span:
            problems.append(f"{j}: sweep covered only {r['seen_max'] - r['seen_min']} of {old_span} counts — "
                            "move it all the way to BOTH end stops")
        d_center = ((lo + hi) - (r["old_min"] + r["old_max"])) / 2
        out[j] = dict(id=r["id"], old=(r["old_min"], r["old_max"]), new=(lo, hi), grow=(g_lo, g_hi),
                      d_offset_deg=d_center * K)
    return out, problems


def write_limits(bus, sid, lo, hi):
    bus.write(sid, REG_LOCK, [0])
    try:
        bus.write(sid, REG_MIN, [lo & 0xFF, lo >> 8])
        bus.write(sid, REG_MAX, [hi & 0xFF, hi >> 8])
    finally:
        bus.write(sid, REG_LOCK, [1])
    got = (bus.u16(sid, REG_MIN), bus.u16(sid, REG_MAX))
    if got != (lo, hi):
        raise IOError(f"servo {sid}: wrote {lo}..{hi}, read back {got[0]}..{got[1]}")


def cmd_apply(a, rig, rig_path, rec_path):
    rec = json.loads(rec_path.read_text())
    side = rec["side"]
    p, problems = plan(rec, a.max_grow)
    for j, q in p.items():
        print(f"  {j:12s} {q['old'][0]}..{q['old'][1]} -> {q['new'][0]}..{q['new'][1]}  "
              f"(+{q['grow'][0]} / +{q['grow'][1]} counts)   rig.json offset {rig['joint_offsets_deg'][side][JOINTS.index(j)]:+.4f}"
              f" -> {rig['joint_offsets_deg'][side][JOINTS.index(j)] + q['d_offset_deg']:+.4f} deg")
    if problems:
        print("\n".join(["REFUSED:"] + problems))
        return 2
    todo = {j: q for j, q in p.items() if q["new"] != q["old"]}
    if not todo:
        print("Sweep stayed inside the current calibration — nothing to widen (the range is this arm's real travel).")
        return 0
    if not a.yes:
        print("(plan only — pass --yes to write)")
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = rig_path.with_name(f"rig.before_recal_{ts}.json")
    shutil.copy2(rig_path, backup)
    bus = open_arm(rig, side)
    try:
        cur = {j: bus.u16(q["id"], REG_PRESENT) for j, q in todo.items()}
        for j, q in todo.items():
            write_limits(bus, q["id"], *q["new"])
            print(f"  wrote {j} (servo {q['id']}) {q['new'][0]}..{q['new'][1]} — read back OK")
    finally:
        bus.close()
    offs = rig["joint_offsets_deg"][side]
    note = []
    for j, q in todo.items():
        i = JOINTS.index(j)
        old_u = (cur[j] - sum(q["old"]) / 2) * K + offs[i]
        offs[i] = round(offs[i] + q["d_offset_deg"], 4)
        new_u = (cur[j] - sum(q["new"]) / 2) * K + offs[i]
        print(f"  {j}: current pose {old_u:+.3f} deg before, {new_u:+.3f} deg after (units unchanged)")
        note.append(f"{j} {q['old'][0]}..{q['old'][1]}->{q['new'][0]}..{q['new'][1]}")
    rig[f"recal_note_{side}"] = (f"{time.strftime('%Y-%m-%d')} recal_range.py widened {', '.join(note)} after the NVIDIA "
                                 f"calibration check; offsets shifted by the center change so units are unchanged. "
                                 f"Backup {backup.name}, record {rec_path.name}.")
    rig_path.write_text(json.dumps(rig, indent=2))
    rec["applied"] = dict(time=ts, rig_backup=str(backup), plan={j: q for j, q in todo.items()})
    rec_path.write_text(json.dumps(rec, indent=2))
    print("rig.json updated; backup", backup)
    return 0


def cmd_restore(a, rig, rig_path, rec_path):
    rec = json.loads(rec_path.read_text())
    if "applied" not in rec:
        sys.exit("record was never applied; nothing to restore")
    bus = open_arm(rig, rec["side"])
    try:
        for j, q in rec["applied"]["plan"].items():
            write_limits(bus, q["id"], *q["old"])
            print(f"  restored {j} {q['old'][0]}..{q['old'][1]}")
    finally:
        bus.close()
    shutil.copy2(rec["applied"]["rig_backup"], rig_path)
    print("rig.json restored from", rec["applied"]["rig_backup"])
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["record", "apply", "restore"])
    ap.add_argument("--side", default="left")
    ap.add_argument("--joints", default="elbow_flex,wrist_flex")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--wait", type=float, default=180, help="max seconds to wait for the first motion")
    ap.add_argument("--max-grow", type=int, default=150, help="max counts a limit may move outward (~13 deg)")
    ap.add_argument("--rig", default=r"C:\robot\real\rig.json")
    ap.add_argument("--record", default=None)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    a.joints = [j.strip() for j in a.joints.split(",") if j.strip()]
    assert all(j in JOINTS[:5] for j in a.joints), a.joints
    rig_path = Path(a.rig)
    rig = json.loads(rig_path.read_text())
    rec_path = Path(a.record) if a.record else HERE / "calib_lerobot" / f"recal_{a.side}_latest.json"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "record":
        return cmd_record(a, rig, rec_path)
    if a.mode == "apply":
        return cmd_apply(a, rig, rig_path, rec_path)
    return cmd_restore(a, rig, rig_path, rec_path)


if __name__ == "__main__":
    sys.exit(main())
