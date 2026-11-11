"""Overhead camera only (mount pose, focal length, k1/k2, principal point): arms at the CAD mount positions
(232 mm, no per-arm shift), joints at the DEPLOYED rig.json offsets. Output is written in the pi_sim camera-rig
schema (fx, fy, cx, cy, OpenCV k1 k2 p1 p2 k3, world pose), ready for assets/so101_camera_rig.json.

All runs here were recorded with the pre-19:28 offsets, so the deployed delta is added to every logged state.

    sim/pi_sim_so101/.venv/bin/python rig_tools/fit_camera_only.py
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_overhead_232 as F    # noqa: E402

D = HERE / "data"
RUNS = [D / "right_robot", D / "left_robot", D / "map_left", D / "map_right"]
DELTA = np.zeros(10)            # deployed minus recorded offsets, left pan..roll then right pan..roll (deg)
DELTA[0:4] = [-2.62, 12.79, 2.17, -6.56]
DELTA[5:9] = [4.0, -0.9, -6.7, -1.2]
INIT = D / "joint_fit_base" / "fit232_both.json"

_C, _IDX = None, None


def full_x(xc):
    x = np.zeros(22)
    x[0:11] = xc
    x[11] = 0.0                                                  # half-spacing = 0.116 (CAD)
    x[12:22] = DELTA / 2.0
    return x


def _init(idx):
    global _C, _IDX
    _C, _IDX = F.Ctx([str(r) for r in RUNS], str(D / "rig.json"), 0.116), idx


def _score(xc):
    return _C.score(full_x(np.asarray(xc)), _IDX)[0]


def main():
    C = F.Ctx(RUNS, D / "rig.json", 0.116)
    n = len(C.states)
    fit_idx, test_idx = list(range(0, n, 2)), list(range(1, n, 2))
    x0 = F.pack_from_json(json.loads(INIT.read_text()), C.M)[:11]
    s0 = C.score(full_x(x0), fit_idx)[0]
    print(f"{n} frames ({len(fit_idx)} fit / {len(test_idx)} held out); start fit-set IoU {s0:.3f}", flush=True)
    import cma
    pool = mp.get_context("spawn").Pool(8, initializer=_init, initargs=(fit_idx,))
    es = cma.CMAEvolutionStrategy(x0, 0.3, {"maxfevals": 2400, "verbose": -9, "seed": 2, "popsize": 16})
    best, t0 = (s0, x0), time.time()
    while not es.stop():
        xs = es.ask()
        ss = pool.map(_score, [np.asarray(x) for x in xs])
        for x, s in zip(xs, ss):
            if s > best[0]:
                best = (s, np.asarray(x).copy())
        es.tell(xs, [-s for s in ss])
        if es.countiter % 10 == 0:
            print(f"  it {es.countiter:3d} ({time.time() - t0:4.0f}s): fit IoU {best[0]:.3f}", flush=True)
    pool.close()
    s, xc = best
    held = C.score(full_x(xc), test_idx)[0]
    cam, _, _ = C.params(full_x(xc))
    pos = (C.M.pos0 + cam["dpos"]).round(4).tolist()
    quat = F.quat_mul(C.M.quat0, F.rotvec_quat(cam["drot"]))
    quat = (quat / np.linalg.norm(quat)).round(5).tolist()
    out = {"fit_iou": s, "held_out_iou": held, "frames": n,
           "intrinsics": {"fx": round(cam["f"], 1), "fy": round(cam["f"], 1), "cx": round(cam["cx"], 1), "cy": round(cam["cy"], 1)},
           "distortion": [round(cam["k1"], 4), round(cam["k2"], 4), 0, 0, 0],
           "pose": {"parent_body": "world", "pos": pos, "quat_wxyz": quat}, "x": np.asarray(xc).tolist()}
    (D / "camera_fit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "x"}, indent=1))


if __name__ == "__main__":
    main()
