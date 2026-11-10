"""Per-episode refinement of the "top" camera: the real overhead camera moved between episodes (the green block
shifts by up to ~50 px between episodes while the side view is static).

    .venv/bin/python so101_twin/stack/episode_cams.py       # -> out/episode_cams.json, out/episode_cams.png

For each of the 40 episodes, Nelder-Mead over a small correction (dx, dy, dz, dyaw, dpitch, droll) of the global
fitted top camera (build_scene.CAMS['top'], written by fit_cameras.py --apply), objective = mean over the episode's
8 frames of (green-block IoU + arm IoU) / 2 at 320x240. replay.py uses the per-episode camera when this file exists.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
import fit_cameras as fc  # noqa: E402
from twin import HERE, real_set  # noqa: E402

OUT = HERE / "out"
CAM = "top"


def main():
    fs = real_set()
    ep = np.asarray(fs.labels["episode"])
    p_glob = fc.p0_from_cams(CAM)[:7]
    result = {}
    t0 = time.time()
    tiles = []
    for k, e in enumerate(np.unique(ep)):
        idx = [int(i) for i in np.where(ep == e)[0]]
        f = fc.Fitter(CAM, idx, fs, small=True)
        base = f.ious(p_glob).mean()

        def loss(d):
            p = p_glob.copy()
            p[:3] += d[:3]
            p[3:6] += d[3:6]
            return 1 - f.ious(p).mean()

        d0 = np.zeros(6)
        step = np.array([0.02, 0.02, 0.02, 2.0, 2.0, 1.5])
        res = minimize(loss, d0, method="Nelder-Mead",
                       options=dict(initial_simplex=np.stack([d0] + [d0 + np.eye(6)[i] * step[i] for i in range(6)]),
                                    xatol=1e-3, fatol=1e-4, maxiter=250, maxfev=300))
        p = p_glob.copy()
        p[:3] += res.x[:3]
        p[3:6] += res.x[3:6]
        result[int(e)] = dict(params=dict(zip(fc.KEYS, map(float, p))), delta=[float(v) for v in res.x],
                              iou_global=float(base), iou=float(1 - res.fun), frames=idx)
        print(f"ep {e:3d}: IoU {base:.3f} -> {1 - res.fun:.3f}  d={np.round(res.x, 3)}  ({time.time() - t0:.0f}s)", flush=True)
        if k % 8 == 0:
            f.set_params(p)
            f.sim.set_joints(f.states[1])
            seg = f.sim.seg(CAM)
            sm = np.isin(seg, f.sim.arm_geoms) | (seg == f.sim.obj_geom["green"])
            rm = f.real[1] | f.real_green[1]
            ov = np.zeros(sm.shape + (3,), np.uint8)
            ov[rm] = (200, 60, 60); ov[sm] = (60, 60, 220); ov[rm & sm] = (240, 240, 240)
            real = np.asarray(Image.fromarray(fs.obs(idx[1])["images"][CAM]).resize(sm.shape[::-1]))
            tiles.append(np.concatenate([real, f.sim.render(CAM), ov], 1))
        f.sim.close()
    (OUT / "episode_cams.json").write_text(json.dumps(dict(cam=CAM, global_params=dict(zip(fc.KEYS, map(float, p_glob))),
                                                           episodes=result), indent=1))
    Image.fromarray(np.concatenate(tiles, 0)).save(OUT / "episode_cams.png")
    ious = np.array([r["iou"] for r in result.values()]); base = np.array([r["iou_global"] for r in result.values()])
    print(f"mean IoU (green+arm)/2 over episodes: global {base.mean():.3f} -> per-episode {ious.mean():.3f}")


if __name__ == "__main__":
    main()
