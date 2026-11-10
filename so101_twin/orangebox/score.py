"""Score a fitted twin with the agreed metric: per camera, whole-arm silhouette IoU between the twin's lens-warped arm
mask on the real 640x480 grid and the (unwarped) SAM mask of the real arm; frames with the real arm < 0.1 % of the
image are excluded; held_out = odd episodes (the fit only sees even ones), all_frames = every visible frame.

    .venv/bin/python so101_twin/orangebox/score.py <fit.json> [--masks _dino[,_c1]] [--model so101|so100]
Prints and writes <fit stem>_score<masks>.json next to the fit json (stats + per-frame IoU).
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


def score(fit_json: pathlib.Path, tag: str):
    fj = json.loads(fit_json.read_text())
    sc = F.load_scene("orangebox")
    if fj.get("model") and fj["model"] != sc.ARM.model:
        sc.ARM = dataclasses.replace(sc.ARM, model=fj["model"], wrist_mount=fj["model"] == "so101" and sc.ARM.wrist_mount)
    P = F.reconcile(sc, fj["params"])
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    out = dict(fit=str(fit_json), masks=tag, model=sc.ARM.model, cams={})
    for c in sc.CAMERAS:
        real = F.load_masks("orangebox", c, tag)
        grp = F.frame_groups(sc, c)
        area = real.reshape(fs.n, -1).mean(1)
        vis = area >= F.VISIBLE
        iou = np.full(fs.n, np.nan)
        for i in np.nonzero(vis)[0]:
            m = rig.mask(c, grp[i], F.frame_state(sc, P, c, i), 640, 480)
            iou[i] = F.iou_frames(m[None], real[i][None])[0]
        odd = ep % 2 == 1
        out["cams"][c] = dict(held_out=F.stat(np.where(odd, iou, np.nan)), all_frames=F.stat(iou),
                              fit_split_even=F.stat(np.where(~odd, iou, np.nan)),
                              excluded_not_in_view=int((~vis).sum()), excluded_held_out=int((~vis & odd).sum()),
                              iou=[None if not np.isfinite(v) else round(float(v), 4) for v in iou])
        h, a = out["cams"][c]["held_out"], out["cams"][c]["all_frames"]
        print(f"{c} [{tag}]: HELD-OUT n={h['n']} median {h['median']:.3f} p10 {h['p10']:.3f} mean {h['mean']:.3f} | "
              f"ALL n={a['n']} median {a['median']:.3f} p10 {a['p10']:.3f} mean {a['mean']:.3f} | "
              f"excluded {out['cams'][c]['excluded_not_in_view']}", flush=True)
    dst = fit_json.with_name(fit_json.stem + f"_score{tag}.json")
    dst.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fit_json")
    ap.add_argument("--masks", default="_dino")
    a = ap.parse_args()
    for t in a.masks.split(","):
        score(pathlib.Path(a.fit_json), t)
