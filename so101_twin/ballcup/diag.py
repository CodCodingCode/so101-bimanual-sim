"""Per-frame diagnostics of a fitted twin: IoU by phase / joint speed, gain from the best per-frame 2D shift of the
twin mask (+-16 px: a pose error vs a shape error), worst frames. Odd (held-out) episodes only by default.

    .venv/bin/python so101_twin/ballcup/diag.py --fit out/<run>_fit.json --mask-tag _t1 [--occ] [--all]
"""
import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402


def best_shift(s, r, R=12, step=3):
    best = (s & r).sum() / max((s | r).sum(), 1)
    for dy in range(-R, R + 1, step):
        for dx in range(-R, R + 1, step):
            t = np.roll(np.roll(s, dy, 0), dx, 1)
            v = (t & r).sum() / max((t | r).sum(), 1)
            best = max(best, v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--mask-tag", required=True)
    ap.add_argument("--occ", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shift", action="store_true")
    a = ap.parse_args()
    sc = F.load_scene("ballcup")
    fj = json.loads(pathlib.Path(a.fit).read_text())
    sc.ARM = dataclasses.replace(sc.ARM, model=fj.get("model", sc.ARM.model))
    sc.MASK_OCCLUDERS = a.occ
    P = F.reconcile(sc, fj["params"])
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    fr = F.fullrate(sc)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    sel = np.arange(fs.n) if a.all else np.nonzero(ep % 2 == 1)[0]
    phase = np.asarray(fs.labels["phase"])
    for c in sc.CAMERAS:
        real = F.load_masks("ballcup", c, a.mask_tag)
        iou, sh, spd = [], [], []
        for i in sel:
            if real[i].mean() < F.VISIBLE:
                iou.append(np.nan); sh.append(np.nan); spd.append(np.nan)
                continue
            s = rig.mask(c, "all", F.frame_state(sc, P, c, int(i)), 640, 480, i=int(i))
            iou.append((s & real[i]).sum() / max((s | real[i]).sum(), 1))
            sh.append(best_shift(s, real[i]) if a.shift else np.nan)
            spd.append(fr.speed(int(i)))
        iou, sh, spd = map(np.asarray, (iou, sh, spd))
        print(f"== {c}: median {np.nanmedian(iou):.3f} p10 {np.nanpercentile(iou, 10):.3f} n {np.isfinite(iou).sum()}"
              + (f" | best-shift median {np.nanmedian(sh):.3f}" if a.shift else ""))
        for ph in sorted(set(phase[sel])):
            m = phase[sel] == ph
            print(f"   phase {ph:14s} n {m.sum():3d} median {np.nanmedian(iou[m]):.3f}")
        for lo, hi in ((0, 0.5), (0.5, 2), (2, 99)):
            m = (spd >= lo) & (spd < hi)
            if m.any():
                print(f"   speed {lo}-{hi} deg/frame n {m.sum():3d} median {np.nanmedian(iou[m]):.3f}")
        order = np.argsort(np.where(np.isfinite(iou), iou, 9))[:8]
        print("   worst:", ", ".join(f"{sel[k]}(ep{ep[sel[k]]},{phase[sel[k]]}) {iou[k]:.2f}" for k in order))


if __name__ == "__main__":
    main()
