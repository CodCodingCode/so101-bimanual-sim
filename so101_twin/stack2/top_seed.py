"""Top camera seed: per-session poses with ONE shared fovy, PnP on the held wooden cube (full rate, EVEN episodes).

    .venv/bin/python so101_twin/stack2/top_seed.py   # -> out/top_seed.json

Joint map fixed to the builder's fitted map expressed for the official SO-101 new_calib model (offsets lift -2.28,
elbow -25.70, wrist_flex +29.58 deg, scales 0.923 / 1.107 / 0.988), which fits the held-cube tracks far better than the
identity map (identity: no camera above the table fits; builder map: 5.6-8.3 px median per session).
"""
import json
import math
import pathlib
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bundle2 as B2  # noqa: E402
import fit as F  # noqa: E402
from bundle import FK, load_obs  # noqa: E402
from seed_pnp import _rotvec, project, rotvec_to_mat  # noqa: E402

OFF = np.array([0, -2.28, -25.70, 29.58, 0.0])
SCL = np.array([1, 0.923, 1.107, 0.988, 1.0])


def main():
    sc = F.load_scene("stack")
    obs, states = load_obs(sc, max_per_cam=100000)
    fk = FK(sc)
    rng = np.random.default_rng(0)
    rows = obs["top"]
    grp = np.array([sc.top_group(int(e)) for e in rows[:, 0]])
    keys = sorted(set(grp))
    data, init = {}, {}
    for g in keys:
        sel = rows[grp == g]
        sel = sel[np.random.default_rng(1).choice(len(sel), min(200, len(sel)), replace=False)]
        X, Z = fk(states, sel, -1.5, OFF, SCL)
        data[g] = (X, Z, sel[:, 2:4])
        b = B2.pnp("top", X, Z, sel[:, 2:4], rng, starts=40, fovy_rng=(40, 75))
        init[g] = b[1] if b else None
        print(g, len(sel), "single-session PnP", None if b is None else round(b[0], 1), flush=True)
    good = [g for g in keys if init[g] is not None]
    for g in keys:
        if init[g] is None:   # session without a solution: start from the nearest session (by episode order)
            init[g] = init[min(good, key=lambda k: abs(keys.index(k) - keys.index(g)))]
    fovy0 = float(np.median([init[g][6] for g in good]))

    def resid(p):
        fovy, a = p[0], p[1]
        out = []
        for k, g in enumerate(keys):
            X, Z, uv = data[g]
            pos, rv = p[2 + 6 * k:5 + 6 * k], p[5 + 6 * k:8 + 6 * k]
            pr, zz = project(pos, rotvec_to_mat(rv), fovy, X + a * Z)
            out.append(((pr - uv) * np.where(zz[:, None] > 0.03, 1.0, 30.0)).ravel())
            out.append(np.array([3000 * max(0.0, 0.15 - pos[2])]))
            out.append(0.3 * (pos - p[2:5]) / 0.01)   # sessions differ by small camera moves (soft tie to session 0)
        return np.concatenate(out)

    p0 = np.concatenate([[fovy0, 0.0]] + [init[g][:6] for g in keys])
    r = least_squares(resid, p0, loss="soft_l1", f_scale=8.0, max_nfev=600,
                      bounds=([30, -0.01] + [-np.inf] * 6 * len(keys), [90, 0.04] + [np.inf] * 6 * len(keys)))
    fovy, a = r.x[0], r.x[1]
    groups = {}
    for k, g in enumerate(keys):
        X, Z, uv = data[g]
        pos, rv = r.x[2 + 6 * k:5 + 6 * k], r.x[5 + 6 * k:8 + 6 * k]
        pr, _ = project(pos, rotvec_to_mat(rv), fovy, X + a * Z)
        err = np.linalg.norm(pr - uv, axis=1)
        print(f"{g}: median {np.median(err):.1f} px p90 {np.percentile(err, 90):.1f} pos {np.round(pos, 3).tolist()}")
        groups[g] = dict(pos=list(map(float, pos)), quat=list(map(float, F.A.quat_from_mat(rotvec_to_mat(rv)))))
    print(f"shared fovy {fovy:.1f}, cube offset {a:.3f}")
    (HERE / "out" / "top_seed.json").write_text(json.dumps(dict(fovy=float(fovy), groups=groups, offsets=OFF.tolist(),
                                                                scales=SCL.tolist()), indent=1))


if __name__ == "__main__":
    main()
