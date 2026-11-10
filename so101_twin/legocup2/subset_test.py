"""Diagnostic: fit the top camera (pose+fovy+offsets) on frame subsets (rest-like only / moving only) to see whether
the model can match each subset, and how far apart the two solutions are."""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

sc = F.load_scene("legocup")
sc.CAMERAS = {"top": sc.CAMERAS["top"]}
sc.OBJ_TRIM = 0.1
fs = F.real_set(sc)
real = F.load_masks("legocup", "top")
P0 = F.reconcile(sc, json.loads((HERE / "out" / sys.argv[1]).read_text())["params"])
ep = np.asarray(fs.labels["episode"]).astype(int)
area = real.reshape(fs.n, -1).mean(1)
rest = np.abs(fs.state[:, 1] + 98) < 3
ok = (area >= F.VISIBLE) & (ep % 2 == 0)
X = fs.state[:, :6] / 30.0
res = {}
for name, grp in (("rest", ok & rest), ("moving", ok & ~rest)):
    cand = np.nonzero(grp)[0]
    idx = cand[F.fps_select(X[cand], 32)]
    fit = {"top": (idx, np.array(["all"] * len(idx)), F.down2(real[idx]))}
    rig = F.Rig(sc, P0)
    obj = F.Objective(rig, sc, fit, P0)
    t0 = time.time()
    s0 = obj.score(P0, reg=False)
    P, s = F.optimise(obj, P0, [("pose", "top", "all"), ("fovy", "top"), ("off",)], None, 350, sigma0=0.8,
                      log=lambda x: None)
    rig.set_params(P)
    v = F.iou_frames(np.stack([rig.mask("top", "all", fs.state[i]) for i in idx]), fit["top"][2])
    g = P["cams"]["top"]["groups"]["all"]
    print(f"{name}: {len(idx)} frames start {s0:.3f} -> {s:.3f}, median {np.nanmedian(v):.3f}; pos {np.round(g['pos'], 3)} "
          f"fovy {P['cams']['top']['fovy']:.1f} off {np.round(P['offsets_deg'], 1).tolist()} ({time.time() - t0:.0f}s)",
          flush=True)
    res[name] = P
    (HERE / "out" / f"subset_{name}.json").write_text(json.dumps(dict(params=P), indent=1))
