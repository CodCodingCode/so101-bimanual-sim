"""Per-frame object pixels from the twin-independent DINO+SAM object masks (no fitting): cup / ball centroid + area
per camera, and per episode the median top-view cup centroid over its clean frames (cup blob area within 30 % of the
episode median, i.e. not cut by the arm). -> out/obj_pixels.json

    .venv/bin/python so101_twin/ballcup/obj_pixels.py [--obj-tag _d1]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))
from sets import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-tag", default="_d1")
    a = ap.parse_args()
    fs = Store().frames("so101_ballcup_real")
    ep = np.asarray(fs.labels["episode"]).astype(int)
    out = dict(obj_tag=a.obj_tag, frames={}, episodes={})
    data = {}
    for c in ("top", "side"):
        for o in ("cup", "ball"):
            m = np.unpackbits(np.load(HERE / "masks" / f"ballcup_{c}{a.obj_tag}_{o}.npy"), axis=-1)[..., :640].astype(bool)
            area = m.reshape(fs.n, -1).sum(1)
            cen = []
            for i in range(fs.n):
                ys, xs = np.nonzero(m[i])
                cen.append([float(xs.mean()), float(ys.mean())] if len(xs) else None)
            data[(c, o)] = (area, cen)
    clean = {}
    for e in sorted(set(ep)):
        idx = np.nonzero(ep == e)[0]
        ok = np.ones(len(idx), bool)
        for c in ("top", "side"):
            ar = data[(c, "cup")][0][idx]
            med = np.median(ar[ar > 0]) if (ar > 0).any() else 0
            ok &= (ar > 0.7 * med) & (ar < 1.3 * med) & (ar > 300)
        clean[e] = idx[ok]
        cens = [data[("top", "cup")][1][i] for i in idx[ok]]
        out["episodes"][str(e)] = dict(cup_top_px=np.median(cens, 0).tolist() if cens else None,
                                       clean_frames=idx[ok].tolist())
    for i in range(fs.n):
        out["frames"][str(i)] = {f"{o}_{c}": dict(px=data[(c, o)][1][i], area=int(data[(c, o)][0][i]))
                                 for c in ("top", "side") for o in ("cup", "ball")}
        out["frames"][str(i)]["cup_clean"] = bool(i in set(clean[int(ep[i])].tolist()))
    (HERE / "out" / "obj_pixels.json").write_text(json.dumps(out))
    print("episodes with a clean top cup centroid:", sum(v["cup_top_px"] is not None for v in out["episodes"].values()))


if __name__ == "__main__":
    main()
