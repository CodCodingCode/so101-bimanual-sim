"""Real-occluder-excluded reading (agreed with the parent / other session): the twin's ARM-ONLY mask (no twin objects)
vs the real SAM arm mask, after removing the REAL occluder pixels (cup / ball masks from the real frames only:
segment.py --mode occ, DINO + SAM, no twin input) from BOTH masks. The fit is frozen (final_fit.json, never refitted
on this reading). Also reports the strict arm-only numbers on the same frames and the fraction of pixels excluded.

    .venv/bin/python so101_twin/ballcup/score_occl.py --fit out/final_fit.json --mask-tag _t2o --occ-tag _occ [--meta]

Frames: those of every other reading (real arm mask >= 0.1 % of the image before exclusion). Per frame
IoU_excl = |(T - O) & (R - O)| / |(T - O) | (R - O)|; excluded fraction = |O & (T | R)| / |T | R| (share of the strict
IoU union removed); twin_excluded = |O & T| / |T|. --meta adds the numbers to the twin set's meta.json under
held_out_real_occluder_excluded (existing fields untouched).
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

SET_META = F.ROOT / "neuroscope" / "molmoact2" / "store" / "sets" / "so101_ballcup_twin2" / "meta.json"


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
    return dict(mean=round(float(np.mean(v)), 4), median=round(float(np.median(v)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="out/final_fit.json")
    ap.add_argument("--mask-tag", default="_t2o")
    ap.add_argument("--occ-tag", default="_occ")
    ap.add_argument("--meta", action="store_true")
    a = ap.parse_args()
    sc = F.load_scene("ballcup")
    fj = json.loads((HERE / a.fit).read_text())
    sc.ARM = dataclasses.replace(sc.ARM, model=fj["model"])
    sc.MASK_OCCLUDERS = False   # ARM-ONLY twin mask: no twin cup / ball anywhere
    sc.CUP_W = 0.0
    P = F.reconcile(sc, fj["params"])
    rig = F.Rig(sc, P)
    assert not rig.occ and not rig.has_obj
    fs = F.real_set(sc)
    odd = np.asarray(fs.labels["episode"]).astype(int) % 2 == 1
    res, per = {}, {}
    for c in sc.CAMERAS:
        R_all = F.load_masks("ballcup", c, a.mask_tag)
        O_all = F.load_masks("ballcup", c, a.occ_tag)
        strict = np.full(fs.n, np.nan)
        excl = np.full(fs.n, np.nan)
        fu = np.full(fs.n, np.nan)
        ft = np.full(fs.n, np.nan)
        vis = R_all.reshape(fs.n, -1).mean(1) >= F.VISIBLE
        for i in np.nonzero(vis)[0]:
            T = rig.mask(c, "all", F.frame_state(sc, P, c, int(i)), 640, 480)
            R, O = R_all[i], O_all[i]
            U = T | R
            strict[i] = (T & R).sum() / max(U.sum(), 1)
            T2, R2 = T & ~O, R & ~O
            u2 = (T2 | R2).sum()
            excl[i] = (T2 & R2).sum() / u2 if u2 else np.nan
            fu[i] = (U & O).sum() / max(U.sum(), 1)
            ft[i] = (T & O).sum() / max(T.sum(), 1)
        sel = {"held_out": odd, "all_frames": np.ones(fs.n, bool)}
        res[c] = dict(
            strict_arm_only={k: stat(np.where(m, strict, np.nan)) for k, m in sel.items()},
            occluder_excluded={k: stat(np.where(m, excl, np.nan)) for k, m in sel.items()},
            excluded_fraction_of_union={k: frac(np.where(m & vis, fu, np.nan)) for k, m in sel.items()},
            excluded_fraction_of_twin_mask={k: frac(np.where(m & vis, ft, np.nan)) for k, m in sel.items()},
            frames_not_in_view=int((~vis).sum()), frames_empty_after_exclusion=int((vis & ~np.isfinite(excl)).sum()))
        per[c] = dict(strict=strict, excl=excl, frac_union=fu)
        v = res[c]
        print(f"{c}: STRICT held_out {v['strict_arm_only']['held_out']} all {v['strict_arm_only']['all_frames']}")
        print(f"{c}: EXCL   held_out {v['occluder_excluded']['held_out']} all {v['occluder_excluded']['all_frames']}")
        print(f"{c}: excluded fraction of union {v['excluded_fraction_of_union']}, of twin mask "
              f"{v['excluded_fraction_of_twin_mask']}")
    out = dict(fit=a.fit, mask_tag=a.mask_tag, occ_tag=a.occ_tag, stats=res,
               per_frame={c: {k: [None if not np.isfinite(x) else round(float(x), 4) for x in arr]
                              for k, arr in d.items()} for c, d in per.items()})
    (HERE / "out" / "score_final_realocc.json").write_text(json.dumps(out, indent=1))
    if a.meta:
        m = json.loads(SET_META.read_text())
        m["held_out_real_occluder_excluded"] = dict(
            protocol=("twin ARM-ONLY mask (no twin cup / ball) vs the real SAM arm mask (so101_twin/ballcup/masks/"
                      f"ballcup_<cam>{a.mask_tag}.npy), after removing the REAL occluder pixels (cup | ball masks from the "
                      f"real frames only, so101_twin/ballcup/masks/ballcup_<cam>{a.occ_tag}.npy: Grounding DINO 'a paper "
                      "cup.' / 'a blue ball.' every box >= 0.3, SAM 2.1 box prompt, kept if bright (median max-channel "
                      ">= 110) and < 20 % of the image; no twin input, no fitted parameter) from BOTH masks; same frames as "
                      "the strict reading (real arm >= 0.1 % of the image); fit frozen (not refitted on this reading); "
                      "held_out = odd episodes"),
            excluded_fraction_definition="per frame |occluder & (twin | real)| / |twin | real| (share of the strict IoU "
                                         "union removed); also of the twin arm mask",
            fit_json=a.fit if a.fit.startswith("so101_twin") else f"so101_twin/ballcup/{a.fit}",
            cameras={c: dict(strict_arm_only=v["strict_arm_only"], occluder_excluded=v["occluder_excluded"],
                             excluded_fraction_of_union=v["excluded_fraction_of_union"],
                             excluded_fraction_of_twin_mask=v["excluded_fraction_of_twin_mask"],
                             frames_not_in_view=v["frames_not_in_view"],
                             frames_empty_after_exclusion=v["frames_empty_after_exclusion"]) for c, v in res.items()},
            score_json="so101_twin/ballcup/out/score_final_realocc.json")
        SET_META.write_text(json.dumps(m, indent=1))
        print("meta.json updated:", SET_META)


if __name__ == "__main__":
    main()
