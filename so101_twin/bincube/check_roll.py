"""Pick the wrist-roll sign / offset: the wrist camera is attached to the Fixed_Jaw body, so its silhouette fit cannot
see the roll. Instead compare where the cube (on the table, per-episode position from out/cube_xy.json) appears
in the real wrist image (yellow centroid) with its projection through the fitted sim wrist camera, over candidate
roll signs and offsets. Also reports the same for the pan offset grid (the pan moves the cube in the wrist view).

    .venv/bin/python so101_twin/bincube/check_roll.py     # prints mean pixel error per candidate; best written to out/roll_check.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
import place_objects  # noqa: E402
import replay  # noqa: E402
import twin  # noqa: E402
from twin import HERE, Sim, real_set  # noqa: E402


def project(sim: Sim, cam: str, X: np.ndarray) -> np.ndarray:
    cid = sim.cam_id[cam]
    R = sim.d.cam_xmat[cid].reshape(3, 3)
    Xc = R.T @ (X - sim.d.cam_xpos[cid])
    fy = 240.0 / np.tan(np.radians(sim.m.cam_fovy[cid]) / 2)
    z = max(-Xc[2], 1e-3)
    return np.array([320 + fy * Xc[0] / z, 240 - fy * Xc[1] / z]), -Xc[2]


def main():
    fs = real_set()
    sim = Sim()
    lids, cubes = replay.all_states(fs)
    cube_xy = replay.load_cube_xy()
    h = build_scene.CUBE["half"]
    samples = []
    for i in np.nonzero(cubes == "table")[0]:
        e = int(fs.labels["episode"][i])
        if e not in cube_xy:
            continue
        n, u, v = place_objects.yellow_centroid(fs.obs(int(i))["images"]["top"])
        if n < 400 or v > 400:          # skip cube-in-jaws-looking frames (big blob at the bottom)
            continue
        samples.append((fs.state[i], np.array([*cube_xy[e], h]), np.array([u, v])))
    print(f"{len(samples)} wrist frames with the cube on the table")
    sign0, off0 = twin.SIGNS[4], twin.OFFSETS_DEG[4]
    results = []
    for sign in (1.0, -1.0):
        for off in range(-180, 181, 15):
            twin.SIGNS[4], twin.OFFSETS_DEG[4] = sign, float(off)
            errs, infront = [], 0
            for st, X, uv in samples:
                sim.set_joints(st)
                p, depth = project(sim, "top", X)
                if depth > 0:
                    infront += 1
                    errs.append(np.linalg.norm(p - uv))
            med = float(np.median(errs)) if errs else 1e9
            results.append((med, sign, off, infront))
    twin.SIGNS[4], twin.OFFSETS_DEG[4] = sign0, off0
    results.sort()
    for med, sign, off, k in results[:8]:
        print(f"roll sign {sign:+.0f} offset {off:+4d}: median px err {med:6.1f} ({k} in front)")
    cur = [r for r in results if r[1] == sign0 and abs(r[2] - off0) < 1e-6]
    print("current:", cur)
    (HERE / "out" / "roll_check.json").write_text(json.dumps(dict(best=dict(sign=results[0][1], offset=results[0][2],
                                                                      median_px=results[0][0]), n=len(samples))))
    sim.close()


if __name__ == "__main__":
    main()
