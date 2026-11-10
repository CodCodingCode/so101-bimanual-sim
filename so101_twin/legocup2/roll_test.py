"""Decide the wrist-roll offset with a fitted top camera per candidate: from the ported builder camera, CMA on top pose
+ fovy (+ offsets lift/elbow/wflex) per roll candidate, trimmed-mean IoU on 48 even-episode frames."""
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
sc.OBJ_TRIM = 0.15
fs = F.real_set(sc)
real = F.load_masks("legocup", "top")
idx = F.choose_fit_frames(sc, "top", real, 48, "even")
fit = {"top": (idx, np.array(["all"] * len(idx)), F.down2(real[idx]))}
rolls = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.0, 180.0, 90.0, -90.0]
for r in rolls:
    P = F.init_params(sc)
    P["offsets_deg"] = [0.0, 0.0, 0.0, 0.0, r]
    rig = F.Rig(sc, P)
    obj = F.Objective(rig, sc, fit, P)
    t0 = time.time()
    s0 = obj.score(P)
    P1, s1 = F.optimise(obj, P, [("pose", "top", "all"), ("fovy", "top")], None, 260, coarse=40, sigma0=1.0,
                        log=lambda x: None)
    P2, s2 = F.optimise(obj, P1, [("pose", "top", "all"), ("fovy", "top"), ("off",)], None, 200, sigma0=0.5,
                        log=lambda x: None)
    rig.set_params(P2)
    sims = np.stack([rig.mask("top", "all", fs.state[i]) for i in idx])
    v = F.iou_frames(sims, fit["top"][2])
    print(f"roll {r:+.0f}: start {s0:.3f} -> pose {s1:.3f} -> +offsets {s2:.3f}; fit-frame median {np.nanmedian(v):.3f} "
          f"offsets {np.round(P2['offsets_deg'], 1).tolist()} fovy {P2['cams']['top']['fovy']:.1f} ({time.time() - t0:.0f}s)",
          flush=True)
    (HERE / "out" / f"roll_test_{int(r)}.json").write_text(json.dumps(dict(params=P2, score=s2), indent=1))
