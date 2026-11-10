"""Check the twin set against the real set and write out/pairs_sheet.jpg (8 random pairs).

    .venv/bin/python so101_twin/pickplace/verify.py [--seed 0]

Each row: real top | twin top | real side | twin side, labelled with the real index, episode and t.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from twin import REAL_SET  # noqa: E402

sys.path.insert(0, str(HERE.parent.parent / "neuroscope" / "molmoact2"))
from sets import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    S = Store()
    real, twin = S.frames(REAL_SET), S.frames("so101_pickplace_twin")
    assert twin.n == real.n and twin.cameras == real.cameras, (twin.n, twin.cameras)
    assert np.array_equal(twin.state, real.state), "state.npy differs"
    assert twin.prompt == real.prompt, "prompt.json differs"
    for k in real.labels:
        if k in ("real_or_sim", "source"):
            continue
        assert np.array_equal(twin.labels[k], real.labels[k]), f"label {k} differs"
    assert np.array_equal(twin.labels["pair"], np.arange(real.n))
    assert set(twin.labels["real_or_sim"]) == {"sim"} and set(twin.labels["source"]) == {"pickplace_twin"}
    for c in real.cameras:
        assert twin._img[c].shape == real._img[c].shape and twin._img[c].dtype == np.uint8, c
    assert twin.meta["twin_of"] == REAL_SET
    print("layout OK:", json.dumps({k: twin.meta[k] for k in ("name", "n", "cameras", "real_or_sim", "source", "image_shape", "twin_of")}))

    rng = np.random.default_rng(a.seed)
    idx = sorted(rng.choice(real.n, a.n, replace=False).tolist())
    rows = []
    for i in idx:
        r, t = real.obs(i), twin.obs(int(twin.labels["pair"][i]))
        tiles = [r["images"]["top"], t["images"]["top"], r["images"]["side"], t["images"]["side"]]
        row = Image.fromarray(np.concatenate([np.asarray(Image.fromarray(x).resize((320, 240))) for x in tiles], 1))
        ImageDraw.Draw(row).text((6, 4), f"#{i} ep{real.labels['episode'][i]} t{real.labels['t'][i]} {real.labels['phase'][i]}",
                                 fill=(255, 0, 0))
        rows.append(np.asarray(row))
    out = HERE / "out" / "pairs_sheet.jpg"
    Image.fromarray(np.concatenate(rows, 0)).save(out, quality=88)
    print("wrote", out, "pairs", idx)


if __name__ == "__main__":
    main()
