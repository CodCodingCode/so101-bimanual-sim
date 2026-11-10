"""Score a pos5 twin (fit json or init json) against one or more real mask sets with the agreed metric.

    .venv/bin/python so101_twin/pos5/score.py out/pos5_r1_fit.json --masks _c1,_lsam [--excl _occ] [--save-twin TAG]

Strict reading: IoU(twin arm-only silhouette, real mask) per frame on the real 480 x 640 grid, frames with the real arm
visible (>= 0.1 % of the image). Held-out = odd episodes. With --excl TAG, also the occluder-excluded reading: the
pixels of pos5/masks/pos5_<cam>TAG.npy (real-frame object masks, twin-free) are removed from both masks, plus the
excluded fraction |O & (T | R)| / |T | R|. Writes out/<fit stem>_score<masks>.json (per-frame IoU included).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import p5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit")
    ap.add_argument("--masks", default="_lsam")
    ap.add_argument("--excl", default="")
    ap.add_argument("--save-twin", default="", help="also save the twin silhouettes as masks/pos5_<cam><TAG>.npy")
    ap.add_argument("--cams", default="top,side")
    a = ap.parse_args()
    fp = pathlib.Path(a.fit)
    fp = fp if fp.is_absolute() else p5.HERE / fp
    P = p5.load_params(fp)
    cams = a.cams.split(",")
    tw = p5.twin_masks(P, cams)
    if a.save_twin:
        for c in cams:
            p5.save(c, a.save_twin, tw[c])
    res = {}
    for tag in a.masks.split(","):
        res[tag] = {}
        for c in cams:
            real = p5.load(c, tag)
            vis = p5.visible(real)
            v = p5.iou(tw[c], real)
            r = dict(strict=p5.report(v, vis), per_frame=[None if not np.isfinite(x) else round(float(x), 4) for x in
                                                         np.where(vis, v, np.nan)])
            if a.excl:
                O = p5.load(c, a.excl)
                ve = p5.iou(tw[c], real, O)
                u = (tw[c] | real).reshape(len(real), -1).sum(1)
                fx = (O & (tw[c] | real)).reshape(len(real), -1).sum(1) / np.maximum(u, 1)
                r["excluded"] = p5.report(ve, vis)
                r["excluded_fraction"] = dict(
                    held_out=dict(mean=round(float(fx[vis & p5.odd()].mean()), 4),
                                  median=round(float(np.median(fx[vis & p5.odd()])), 4)),
                    all_frames=dict(mean=round(float(fx[vis].mean()), 4), median=round(float(np.median(fx[vis])), 4)))
                r["per_frame_excluded"] = [None if not np.isfinite(x) else round(float(x), 4) for x in
                                           np.where(vis, ve, np.nan)]
            res[tag][c] = r
            s = r["strict"]
            line = (f"{tag:>10} {c:>4}: strict held_out med {s['held_out']['median']:.3f} p10 {s['held_out']['p10']:.3f} "
                    f"mean {s['held_out']['mean']:.3f} n={s['held_out']['n']} | all med {s['all_frames']['median']:.3f} "
                    f"p10 {s['all_frames']['p10']:.3f} mean {s['all_frames']['mean']:.3f} n={s['all_frames']['n']}")
            if a.excl:
                e = r["excluded"]
                line += (f"\n{'':>10} {c:>4}: excl   held_out med {e['held_out']['median']:.3f} p10 "
                         f"{e['held_out']['p10']:.3f} mean {e['held_out']['mean']:.3f} | all med "
                         f"{e['all_frames']['median']:.3f} | excluded frac held_out mean "
                         f"{r['excluded_fraction']['held_out']['mean']:.3f} median "
                         f"{r['excluded_fraction']['held_out']['median']:.3f}")
            print(line, flush=True)
    out = fp.parent / f"{fp.stem}_score{a.masks.replace(',', '')}{a.excl}.json"
    out.write_text(json.dumps(dict(fit=str(fp), masks=a.masks, excl=a.excl, results=res), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
