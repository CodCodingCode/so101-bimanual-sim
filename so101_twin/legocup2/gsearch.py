"""Broad CMA search for one camera (pose + fovy + lens + joint offsets / gripper) from a params json, trimmed-mean IoU
objective on even-episode frames (stratified: rest-like / moving), then writes out/<tag>.json (fit-json format).

    .venv/bin/python so101_twin/legocup2/gsearch.py CAM INIT.json TAG [--evals 600] [--sigma 1.5] [--mask-tag ""]
        [--layout pose,fovy,lens,off] [--n 64] [--trim 0.15] [--restarts 2]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cam")
ap.add_argument("init")
ap.add_argument("tag")
ap.add_argument("--evals", type=int, default=600)
ap.add_argument("--sigma", type=float, default=1.5)
ap.add_argument("--mask-tag", default="")
ap.add_argument("--layout", default="pose,fovy,lens,off")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--trim", type=float, default=0.15)
ap.add_argument("--restarts", type=int, default=2)
ap.add_argument("--res", type=int, default=2, help="downsample factor of the fit masks (2 = 320x240)")
a = ap.parse_args()
sc = F.load_scene("legocup")
sc.CAMERAS = {a.cam: sc.CAMERAS[a.cam]}
sc.OBJ_TRIM = a.trim
sc.SCALE_SIGMA = [0.25] * 5   # normalised-units dataset: per-joint deg/unit scale, loose prior around 1
F.SCALE["jscale"] = 0.06
fs = F.real_set(sc)
init = json.loads(pathlib.Path(a.init).read_text())
P0 = F.reconcile(sc, init["params"])
real = F.load_masks("legocup", a.cam, a.mask_tag)
ep = np.asarray(fs.labels["episode"]).astype(int)
area = real.reshape(fs.n, -1).mean(1)
ok = (area >= F.VISIBLE) & (ep % 2 == 0)
rest = np.abs(fs.state[:, 1] + 98) < 3
X = fs.state[:, :6] / 30.0
idx = []
for grp in (ok & rest, ok & ~rest):
    cand = np.nonzero(grp)[0]
    k = int(round(a.n * len(cand) / ok.sum()))
    idx += cand[F.fps_select(X[cand], max(k, 8))].tolist()
idx = np.array(sorted(set(idx)))
W, H = 640 // a.res, 480 // a.res


def down(m):
    n, h, w = m.shape
    return m.reshape(n, h // a.res, a.res, w // a.res, a.res).mean((2, 4)) >= 0.5


fit = {a.cam: (idx, np.array(["all"] * len(idx)), down(real[idx]))}
F.FW, F.FH = W, H
rig = F.Rig(sc, P0)
obj = F.Objective(rig, sc, fit, P0)
lay = []
for k in a.layout.split(","):
    lay.append(("pose", a.cam, "all") if k == "pose" else (k, a.cam) if k in ("fovy", "lens", "lag") else (k,))
t0 = time.time()
s0 = obj.score(P0)
print(f"{a.cam}: {len(idx)} frames ({(rest[idx]).sum()} rest-like), start trimmed {s0:.3f}", flush=True)
best = (s0, P0)
for r in range(a.restarts):
    P, s = F.optimise(obj, best[1], lay, None, a.evals, sigma0=a.sigma if r == 0 else a.sigma / 2, seed=10 + r,
                      log=lambda x: print(x, flush=True), tag=f"[G{r} {time.time() - t0:.0f}s]")
    if s > best[0]:
        best = (s, P)
P = best[1]
rig.set_params(P)
sims = np.stack([rig.mask(a.cam, "all", F.frame_state(sc, P, a.cam, i)) for i in idx])
v = F.iou_frames(sims, fit[a.cam][2])
print(f"done {time.time() - t0:.0f}s: trimmed {best[0]:.3f}, fit-frame IoU median {np.nanmedian(v):.3f} rest "
      f"{np.nanmedian(v[rest[idx]]):.3f} moving {np.nanmedian(v[~rest[idx]]):.3f}", flush=True)
print("offsets", np.round(P["offsets_deg"], 2).tolist(), "grip", round(P["grip_scale"], 3), round(P.get("grip_off", 0), 3),
      "fovy", round(P["cams"][a.cam]["fovy"], 2), "lens", {k: round(x, 4) for k, x in P["cams"][a.cam]["lens"].items()},
      flush=True)
out = dict(init)
out["params"] = json.loads(json.dumps(init["params"]))
out["params"]["cams"][a.cam] = P["cams"][a.cam]
for k in ("offsets_deg", "grip_scale", "grip_off", "scales"):
    if k in P:
        out["params"][k] = P[k]
(HERE / "out" / f"{a.tag}.json").write_text(json.dumps(out, indent=1))
print("wrote", HERE / "out" / f"{a.tag}.json")
