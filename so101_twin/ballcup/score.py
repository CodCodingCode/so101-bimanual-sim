"""Score a fitted ballcup twin with the agreed metric (independent re-computation, full 640x480 real pixel grid).

    .venv/bin/python so101_twin/ballcup/score.py --fit out/<run>_fit.json --mask-tag _t1 [--occ] [--out out/score.json]

Per camera: whole-arm silhouette IoU between the twin's rendered arm mask (padded pinhole render warped by the fitted
lens onto the real 640x480 grid; pixels the render cannot fill count as background) and the SAM mask of the real arm
(never resampled). Frames where the real arm covers < 0.1 % of the image are excluded (arm not in view). held_out =
ODD episodes (never used by the fit), all_frames = every included frame. --occ: the twin's cup / ball (objects.json)
hide arm pixels behind them (visible-arm mask); without it the twin arm mask is the unoccluded arm silhouette.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402


def stat(v):
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0, median=float("nan"), p10=float("nan"), mean=float("nan"))
    return dict(n=int(len(v)), median=round(float(np.median(v)), 4), p10=round(float(np.percentile(v, 10)), 4),
                mean=round(float(np.mean(v)), 4))


def score(fit_json: str, mask_tag: str, occ: bool, model: str = ""):
    sc = F.load_scene("ballcup")
    fj = json.loads(pathlib.Path(fit_json).read_text())
    import dataclasses

    sc.ARM = dataclasses.replace(sc.ARM, model=model or fj.get("model", sc.ARM.model))
    sc.MASK_OCCLUDERS = occ
    P = F.reconcile(sc, fj["params"])
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    odd = np.asarray(fs.labels["episode"]).astype(int) % 2 == 1
    out, per = {}, {}
    for c in sc.CAMERAS:
        real = F.load_masks("ballcup", c, mask_tag)
        area = real.reshape(fs.n, -1).mean(1)
        vis = area >= F.VISIBLE
        iou = np.full(fs.n, np.nan)
        for i in np.nonzero(vis)[0]:
            m = rig.mask(c, "all", F.frame_state(sc, P, c, int(i)), 640, 480, i=int(i))
            iou[i] = (m & real[i]).sum() / max((m | real[i]).sum(), 1)
        out[c] = dict(held_out=stat(np.where(odd, iou, np.nan)), all_frames=stat(iou),
                      fit_split_even=stat(np.where(~odd, iou, np.nan)),
                      excluded_not_in_view=dict(all=int((~vis).sum()), held_out=int((~vis & odd).sum())))
        per[c] = iou
    return out, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--mask-tag", required=True)
    ap.add_argument("--occ", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out, per = score(a.fit, a.mask_tag, a.occ, a.model)
    for c, v in out.items():
        print(c, "held_out", v["held_out"], "| all", v["all_frames"], "| excluded", v["excluded_not_in_view"])
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(dict(fit=a.fit, mask_tag=a.mask_tag, occluders=a.occ, stats=out,
                                                       per_frame={c: [None if not np.isfinite(x) else round(float(x), 4)
                                                                      for x in v] for c, v in per.items()}), indent=1))


if __name__ == "__main__":
    main()
