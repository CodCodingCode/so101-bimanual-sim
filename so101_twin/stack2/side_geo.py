"""Side camera seed from the resting cube: cube centre on the disc (grasp) and on the block (release) from FK (seed joint
map), their side-image centroids and apparent sizes (the 4 cm cube: width_px = f * 0.04 / depth), EVEN episodes only.

    .venv/bin/python so101_twin/stack2/side_geo.py [--write]
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
from bundle import FK  # noqa: E402
from seed_pnp import _rotvec, project, rotvec_to_mat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    sc = F.load_scene("stack")
    seed = json.loads((HERE / "out" / "seed_cams.json").read_text())
    off, scl, lag, acube = (np.array(seed["joint"]["offsets"]), np.array(seed["joint"]["scales"]), seed["joint"]["lag"],
                            seed["joint"]["cube_a"])
    z = np.load(HERE / "out" / "fullrate_cube.npz")
    fk = FK(sc)
    pts, uv, wpx = [], [], []
    for k in z.files:
        if not k.startswith("state_"):
            continue
        e = int(k.split("_")[1])
        st, sd = z[k], z[f"side_{e}"]
        g = st[:, 5]
        # grasp: first close after open with the arm down; release: first open after holding
        tg = next((t for t in range(1, len(g)) if g[t - 1] > 18 >= g[t] and st[t, 1] > -60), None)
        tr = next((t for t in range(tg + 10 if tg else 1, len(g)) if g[t - 1] <= 18 < g[t]), None) if tg else None
        for t_ev, t_img in ((tg, (tg - 15) if tg else None), (tr, (tr + 15) if tr else None)):
            if t_ev is None or t_img is None or t_img >= len(g) or not np.isfinite(sd[t_img, 0]):
                continue
            X, Zg = fk({e: st}, np.array([[e, t_ev, 0, 0]], float), lag, off, scl)
            pts.append(X[0] + acube * Zg[0])
            uv.append(sd[t_img, :2])
            x0, y0, x1, y1 = sd[t_img, 3:]
            wpx.append(x1 - x0)
    pts, uv, wpx = map(np.asarray, (pts, uv, wpx))
    print(len(pts), "resting-cube observations; widths px", np.round(np.percentile(wpx, [10, 50, 90]), 0).tolist())

    def resid(p):
        pos, R, fovy = p[:3], rotvec_to_mat(p[3:6]), p[6]
        pr, depth = project(pos, R, fovy, pts)
        f = 0.5 * 480 / math.tan(math.radians(fovy) / 2)
        r1 = (pr - uv).ravel()
        r2 = (f * 0.04 / np.maximum(depth, 0.02) - wpx) * 0.5
        right = R[:, 0]
        r3 = [right[2] * 200]          # no roll: the table horizon is level in the side image
        return np.concatenate([r1, r2, r3])

    rng = np.random.default_rng(0)
    best = None
    blk = pts[len(pts) // 2:].mean(0) if len(pts) > 2 else pts.mean(0)
    ctr = pts.mean(0)
    for s in range(300):
        pos = ctr + rng.normal(0, 0.25, 3)
        pos[2] = rng.uniform(-0.05, 0.15)
        q = F.A.look_quat(pos, ctr + rng.normal(0, 0.03, 3), roll_deg=0)
        p0 = np.concatenate([pos, _rotvec(F.A.mat_from_quat(q)), [rng.uniform(35, 75)]])
        try:
            r = least_squares(resid, p0, loss="soft_l1", f_scale=10, bounds=([-np.inf] * 6 + [25], [np.inf] * 6 + [100]))
        except Exception:
            continue
        rr = resid(r.x)
        c = float(np.median(np.abs(rr[:2 * len(pts)])) + np.median(np.abs(rr[2 * len(pts):-1])))
        if best is None or c < best[0]:
            best = (c, r.x)
    p = best[1]
    pr, depth = project(p[:3], rotvec_to_mat(p[3:6]), p[6], pts)
    print("best cost", round(best[0], 2), "pos", np.round(p[:3], 3).tolist(), "fovy", round(p[6], 1),
          "reproj median", np.round(np.median(np.linalg.norm(pr - uv, axis=1)), 1), "depths", np.round(np.percentile(depth, [10, 90]), 3).tolist())
    if a.write:
        seed["cams"]["side"] = dict(pos=list(map(float, p[:3])), quat=list(map(float, F.A.quat_from_mat(rotvec_to_mat(p[3:6])))),
                                    fovy=float(p[6]))
        seed["method"] += "; side: resting-cube positions + apparent size (side_geo.py)"
        (HERE / "out" / "seed_cams.json").write_text(json.dumps(seed, indent=1))
        print("wrote side camera into seed_cams.json")


if __name__ == "__main__":
    main()
