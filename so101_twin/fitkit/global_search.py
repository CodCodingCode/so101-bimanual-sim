"""Wide camera search for scenes whose starting camera is far off (drawer, bincube):
    .venv/bin/python so101_twin/fitkit/global_search.py <scene> <view> [--model so101] [--n 3000]
Random poses within +-0.3 m / +-40 deg / fovy 35-90 of the start at quarter resolution, Nelder-Mead from the best
10, then writes out/<scene>_<model>_global.json (a saved fit usable as --init)."""
import argparse
import json
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, __import__("os").path.dirname(__file__))
import armfit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("view")
ap.add_argument("--model", default="so101")
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sphere", default="", help="cx,cy,cz,rmin,rmax: sample cameras on the upper hemisphere around this "
                                             "point looking at it (no usable starting camera)")
a = ap.parse_args()
cfg = armfit.scene_config(a.scene)
F = armfit.Fit(cfg, a.model, k_fit=4, n_fit=16)
F.set_lag(-1.5)
if cfg["real_mask"] == "auto":
    F.train_colour(a.view)
rng = np.random.default_rng(a.seed)
p0 = F.p[a.view].copy()
span = np.array([0.3, 0.3, 0.3, 40, 40, 40])
t0 = time.time()
cands = []
def sphere_sample():
    cx, cy, cz, r0, r1 = map(float, a.sphere.split(","))
    az, el = rng.uniform(-180, 180), rng.uniform(3, 89)
    r = rng.uniform(r0, r1)
    e, z = np.radians(el), np.radians(az)
    pos = np.array([cx, cy, cz]) + r * np.array([np.cos(e) * np.cos(z), np.cos(e) * np.sin(z), np.sin(e)])
    look = np.array([cx, cy, cz]) + rng.uniform(-0.1, 0.1, 3)
    ypr = armfit.lookat_to_ypr(pos, look, rng.uniform(-25, 25))
    return np.r_[ypr, rng.uniform(35, 95), p0[7:]]


for k in range(a.n):
    if a.sphere:
        p = sphere_sample()
    else:
        p = p0.copy()
        p[:6] += rng.uniform(-1, 1, 6) * span
        p[6] = rng.uniform(35, 90)
    cands.append((F.loss(a.view, p), p))
    if k % 500 == 0:
        print(f"{k}/{a.n} best {1 - min(c[0] for c in cands):.3f} {time.time() - t0:.0f}s", flush=True)
cands.sort(key=lambda c: c[0])
best = cands[0]
for L, p in cands[:10]:
    res = minimize(lambda x: F.loss(a.view, np.r_[x, p[7:]]), p[:7], method="Nelder-Mead",
                   options=dict(initial_simplex=np.vstack([p[:7]] + [p[:7] + np.eye(7)[j] * [0.03, 0.03, 0.03, 4, 4, 4, 4][j] for j in range(7)]),
                                maxfev=600, xatol=1e-4, fatol=1e-5))
    print(f"refine {1 - L:.3f} -> {1 - res.fun:.3f}", flush=True)
    if res.fun < best[0]:
        best = (res.fun, np.r_[res.x, p[7:]])
F.p[a.view] = best[1]
if a.model == "so101":
    print("wrist map", F.choose_wrist_map())
st = F.state()
st["scene"] = a.scene
path = armfit.OUT / f"{a.scene}_{a.model}_global.json"
path.write_text(json.dumps(st, indent=1))
print(f"best soft IoU {1 - best[0]:.3f} -> {path}", flush=True)
