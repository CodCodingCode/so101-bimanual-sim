#!/usr/bin/env python3
"""Real runs (C:\\robot\\real\\runs\\real_*) -> VLA episodes in the same format as gen_vla_sim.py.

Per tick (25 Hz): state = measured joints (log "units", 5 per arm), action = commanded joints (log "cmd", 5 per arm).
Images: each saved frame is 960x240 = overhead | left_wrist | right_wrist at 320x240, saved every 5th tick; a tick uses
the latest frame at or before it (held up to 4 ticks, 0.16 s).

    python squeeze/convert_vla_real.py <folder with the extracted run folders>
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs_clamp_v3" / "vla_real"
CAMS = ("overhead", "left_wrist", "right_wrist")
TASK = "squeeze the red cube between both grippers, lift it and drop it into the grey box"
# runs kept and the tick range used (None = all). Left out: dry runs (arms did not move), the 11-11 midday scripted run
# whose commands knocked off the camera holder (135033) and the pre-redesign scripted runs (004540, 004759).
RUNS = {
    "real_clamp_20261111_185600": None,
    "real_clamp_20261111_191358": None,
    "real_clamp_20261111_192301": None,       # scripted grab + place into the box (worked)
    "real_policy_20261112_164153": None,
    "real_policy_20261112_165142": None,
    "real_policy_20261112_165355": None,
    "real_policy_20261112_165729": None,
    "real_policy_20261112_170635": None,
    "real_policy_20261112_171300": (0, 265),  # the cube fell out at ~10.7 s: keep the grab + carry before it
    "real_policy_20261112_172136": None,
    "real_policy_20261112_175335": None,      # delivered the cube into the box
}


def convert(src: Path, name: str, rng):
    L = json.loads((src / name / "log.json").read_text())
    if rng:
        L = L[rng[0]:rng[1]]
    frames = {}
    for f in (src / name / "frames").glob("t*.jpg"):
        frames[int(re.match(r"t(\d+)", f.stem).group(1))] = f
    ticks = sorted(frames)
    jpg = {c: [] for c in CAMS}
    for k in ticks:
        im = Image.open(frames[k]).convert("RGB")
        assert im.size == (960, 240), (name, k, im.size)
        for i, c in enumerate(CAMS):
            buf = io.BytesIO()
            im.crop((320 * i, 0, 320 * (i + 1), 240)).save(buf, "JPEG", quality=90)
            jpg[c].append(buf.getvalue())
    start = rng[0] if rng else 0
    S, A, F = [], [], []
    for j, r in enumerate(L):
        t = start + j
        avail = [i for i, k in enumerate(ticks) if k <= t]
        if not avail:
            continue
        S.append(np.r_[r["units"]["left"][:5], r["units"]["right"][:5]])
        A.append(np.r_[r["cmd"]["left"][:5], r["cmd"]["right"][:5]])
        F.append(avail[-1])
    np.savez(OUT / f"ep_{name}.npz", state=np.asarray(S, np.float32), action=np.asarray(A, np.float32),
             frame_idx=np.asarray(F, np.int32), cube=np.zeros((len(S), 3), np.float32),
             **{f"jpg_{c}": np.array(jpg[c], dtype=object) for c in CAMS}, task=TASK, fps=25, source=name)
    return len(S), len(ticks)


def main():
    src = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    tot = 0
    for name, rng in RUNS.items():
        n, f = convert(src, name, rng)
        tot += n
        print(f"{name}: {n} ticks, {f} frames")
    print(f"total {tot} real ticks -> {OUT}")


if __name__ == "__main__":
    main()
