"""JAR: score a fitted jar twin with the agreed metric (independent re-computation on the full 640x480 real grid),
after ballcup/score.py + ballcup/score_occl.py.

    .venv/bin/python so101_twin/jar/score.py --fit out/<run>_fit.json --mask-tag _t2 [--occ-tag _occ] [--out out/s.json]

Strict reading: the twin's ARM-ONLY mask (every geom of the arm subtree, base included; padded pinhole render warped by
the fitted lens onto the real grid; pixels the render cannot fill count as background) vs the SAM mask of the real arm
(never resampled). Frames where the real arm covers < 0.1 % of the image are excluded (arm not in view).
held_out = ODD episodes (never used by the fit), all_frames = every included frame.
Occluder-excluded reading (--occ-tag): the same masks after removing the REAL occluder pixels (masks/jar_<cam><occ>.npy,
real frames only: DINO + SAM, no twin input) from BOTH; excluded fraction = |O & (T | R)| / |T | R| per frame.
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


def stat(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0, median=float("nan"), p10=float("nan"), mean=float("nan"))
    return dict(n=int(len(v)), median=round(float(np.median(v)), 4), p10=round(float(np.percentile(v, 10)), 4),
                mean=round(float(np.mean(v)), 4))


def frac(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return dict(mean=round(float(np.mean(v)), 4), median=round(float(np.median(v)), 4)) if len(v) else None


def score(fit_json, mask_tag, occ_tag="", model=""):
    sc = F.load_scene("jar")
    fj = json.loads(pathlib.Path(fit_json).read_text())
    sc.ARM = dataclasses.replace(sc.ARM, model=model or fj.get("model", sc.ARM.model))
    P = F.reconcile(sc, fj["params"])
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    odd = np.asarray(fs.labels["episode"]).astype(int) % 2 == 1
    out, per = {}, {}
    for c in sc.CAMERAS:
        R_all = F.load_masks("jar", c, mask_tag)
        O_all = F.load_masks("jar", c, occ_tag) if occ_tag else None
        vis = R_all.reshape(fs.n, -1).mean(1) >= F.VISIBLE
        strict = np.full(fs.n, np.nan)
        excl = np.full(fs.n, np.nan)
        fu = np.full(fs.n, np.nan)
        ft = np.full(fs.n, np.nan)
        for i in np.nonzero(vis)[0]:
            T = rig.mask(c, "all", F.frame_state(sc, P, c, int(i)), 640, 480)
            R = R_all[i]
            U = T | R
            strict[i] = (T & R).sum() / max(U.sum(), 1)
            if O_all is not None:
                O = O_all[i]
                T2, R2 = T & ~O, R & ~O
                u2 = (T2 | R2).sum()
                excl[i] = (T2 & R2).sum() / u2 if u2 else np.nan
                fu[i] = (U & O).sum() / max(U.sum(), 1)
                ft[i] = (T & O).sum() / max(T.sum(), 1)
        sel = {"held_out": odd, "all_frames": np.ones(fs.n, bool)}
        out[c] = dict(strict_arm_only={k: stat(np.where(m, strict, np.nan)) for k, m in sel.items()},
                      fit_split_even=stat(np.where(~odd, strict, np.nan)),
                      excluded_not_in_view=dict(all=int((~vis).sum()), held_out=int((~vis & odd).sum())))
        if O_all is not None:
            out[c].update(occluder_excluded={k: stat(np.where(m, excl, np.nan)) for k, m in sel.items()},
                          excluded_fraction_of_union={k: frac(np.where(m & vis, fu, np.nan)) for k, m in sel.items()},
                          excluded_fraction_of_twin_mask={k: frac(np.where(m & vis, ft, np.nan)) for k, m in sel.items()},
                          frames_empty_after_exclusion=int((vis & ~np.isfinite(excl)).sum()))
        per[c] = dict(strict=strict, excl=excl, frac_union=fu)
    return out, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--mask-tag", required=True)
    ap.add_argument("--occ-tag", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out, per = score(a.fit, a.mask_tag, a.occ_tag, a.model)
    for c, v in out.items():
        print(f"{c} STRICT held_out {v['strict_arm_only']['held_out']} | all {v['strict_arm_only']['all_frames']} | "
              f"even {v['fit_split_even']} | excluded {v['excluded_not_in_view']}")
        if a.occ_tag:
            print(f"{c} EXCL   held_out {v['occluder_excluded']['held_out']} | all {v['occluder_excluded']['all_frames']}"
                  f" | frac union {v['excluded_fraction_of_union']}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(dict(
            fit=a.fit, mask_tag=a.mask_tag, occ_tag=a.occ_tag, stats=out,
            per_frame={c: {k: [None if not np.isfinite(x) else round(float(x), 4) for x in arr] for k, arr in d.items()}
                       for c, d in per.items()}), indent=1))


if __name__ == "__main__":
    main()
