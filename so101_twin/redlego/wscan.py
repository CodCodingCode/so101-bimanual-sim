"""REDLEGO: scan one joint's offset (and sign) with everything else fixed at a fit json; mean IoU per camera at 160x120
(fitted lens) over EVEN frames chosen by farthest-point sampling.
    .venv/bin/python so101_twin/redlego/wscan.py --fit out/<x>.json --joint 3 [--mask-tag _d1k] [--step 10]"""
import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

W, H = 160, 120
ap = argparse.ArgumentParser()
ap.add_argument("--fit", required=True)
ap.add_argument("--joint", type=int, default=3)
ap.add_argument("--mask-tag", default="_d1k")
ap.add_argument("--step", type=float, default=10.0)
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--lo", type=float, default=-180)
ap.add_argument("--hi", type=float, default=180)
a = ap.parse_args()
sc = F.load_scene("redlego")
fj = json.loads(pathlib.Path(a.fit).read_text())
sc.ARM = dataclasses.replace(sc.ARM, model=fj.get("model", sc.ARM.model))
P0 = F.reconcile(sc, fj["params"])
rig = F.Rig(sc, P0)
fs = F.real_set(sc)
ep = np.asarray(fs.labels["episode"]).astype(int)
X = fs.state[:, :6] / 30.0
data = {}
for c in sc.CAMERAS:
    real = F.load_masks("redlego", c, a.mask_tag)
    small = real.reshape(fs.n, H, 4, W, 4).mean((2, 4)) >= 0.5
    vis = np.nonzero((ep % 2 == 0) & (small.reshape(fs.n, -1).mean(1) > 0.005))[0]
    sel = vis[F.fps_select(X[vis], min(a.frames, len(vis)), seed=5)]
    data[c] = (sel, small[sel])
j = a.joint
s0 = float(P0["scales"][j])
off0 = float(P0["offsets_deg"][j])
print(f"joint {j}: fitted scale {s0:.3f} offset {off0:.2f}")
res = []
for sg in (1.0, -1.0):
    for off in np.arange(a.lo, a.hi, a.step):
        P = json.loads(json.dumps(P0))
        P["scales"][j] = sg * abs(s0)
        P["offsets_deg"][j] = float(off)
        rig.set_params(P)
        v = {}
        for c, (sel, real) in data.items():
            ious = []
            for i, r in zip(sel, real):
                m = rig.mask(c, "all", F.frame_state(sc, P, c, int(i)), W, H)
                u = (m | r).sum()
                ious.append((m & r).sum() / u if u else 0.0)
            v[c] = float(np.mean(ious))
        res.append((sg, off, v))
        print(f"sign {sg:+.0f} off {off:+6.1f}: " + " ".join(f"{c} {x:.3f}" for c, x in v.items()), flush=True)
best = max(res, key=lambda r: np.mean(list(r[2].values())))
print("best", best)
