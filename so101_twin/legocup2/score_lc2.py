"""Score a legocup2 params json with the agreed metric against one or more SAM mask sets.

    .venv/bin/python so101_twin/legocup2/score_lc2.py out/legocup_final_fit.json --tags ,_lc2 [--json out/x.json]

Metric: whole-arm silhouette IoU, twin arm mask (640x480, lens warp, state at the camera's fitted latency) vs SAM
mask of the real arm (so101_twin/masks/legocup_<cam><tag>.npy), per camera, frames where the real arm is visible
(>= 0.1 % of the image). held_out = odd episodes (never used by the fit), all_frames = every visible frame.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("params")
ap.add_argument("--tags", default=",_lc2")
ap.add_argument("--json", default="")
a = ap.parse_args()
sc = F.load_scene("legocup")
fs = F.real_set(sc)
pj = pathlib.Path(a.params)
P = F.reconcile(sc, json.loads((pj if pj.is_absolute() else HERE / pj).read_text())["params"])
rig = F.Rig(sc, P)
ep = np.asarray(fs.labels["episode"]).astype(int)
odd = ep % 2 == 1
sims = {c: np.stack([rig.mask(c, "all", F.frame_state(sc, P, c, i), 640, 480) for i in range(fs.n)]) for c in sc.CAMERAS}
prompts = {c: F.load_masks("legocup", c, sub="prompts") for c in sc.CAMERAS}
out = {}
for tag in a.tags.split(","):
    if not (F.MASKS / f"legocup_top{tag}.npy").exists():
        continue
    res = {}
    for c in sc.CAMERAS:
        real = F.load_masks("legocup", c, tag)
        vis = real.reshape(fs.n, -1).mean(1) >= F.VISIBLE
        new = np.where(vis, F.iou_frames(sims[c], real), np.nan)
        old = np.where(vis, F.iou_frames(prompts[c], real), np.nan)
        res[c] = dict(new=F.stat(new), new_heldout=F.stat(np.where(odd, new, np.nan)),
                      old=F.stat(old), old_heldout=F.stat(np.where(odd, old, np.nan)), n_visible=int(vis.sum()),
                      per_frame_new=[None if not np.isfinite(x) else round(float(x), 4) for x in new])
        h, al = res[c]["new_heldout"], res[c]["new"]
        oh = res[c]["old_heldout"]
        print(f"masks '{tag or 'orig'}' {c}: HELD-OUT n={h['n']} median {h['median']:.3f} p10 {h['p10']:.3f} mean "
              f"{h['mean']:.3f} | ALL n={al['n']} median {al['median']:.3f} p10 {al['p10']:.3f} mean {al['mean']:.3f}"
              f" | old twin held-out median {oh['median']:.3f}", flush=True)
    out[tag or "orig"] = res
if a.json:
    pathlib.Path(a.json if pathlib.Path(a.json).is_absolute() else HERE / a.json).write_text(json.dumps(out, indent=1))
