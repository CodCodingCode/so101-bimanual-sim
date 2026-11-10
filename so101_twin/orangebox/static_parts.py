"""Static parts of the SIDE view at its left edge (the side camera is static within each episode group):
  base  = pixels orange (hue mask) in >= 85 % of the group's frames, x < 60: the arm's fixed base (the twin renders
          the base there: base_so101 + waveshare mounting plate);
  clamp = pixels dark (v < 0.3) in >= 90 % of the group's frames, x < 60, y > 330: the black stand/clamp under the base
          (not part of the arm model).
    .venv/bin/python so101_twin/orangebox/static_parts.py -> masks/orangebox_side_static.npz (base, clamp per group)
    and  apply(mask_tag, out_tag): out = (mask - clamp) | (base & orange in that frame)   [diagnostic / union mode]
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
sys.path.insert(0, str(HERE))
from sets import Store  # noqa: E402
from hue_masks import hsv  # noqa: E402

fs = Store().frames("so101_orangebox_real")
EP = fs.labels["episode"].astype(int)
GROUPS = {"e00-09": EP <= 9, "e10-48": EP > 9}


def unpack(a):
    return np.unpackbits(a, axis=-1)[..., :640].astype(bool)


def build():
    hue = unpack(np.load(HERE / "masks" / "orangebox_side_hue.npy"))
    out = {}
    X = np.arange(640)[None, :]
    Y = np.arange(480)[:, None]
    for g, m in GROUPS.items():
        dark = np.mean([hsv(np.asarray(fs.obs(int(i))["images"]["side"]))[2] < 0.3 for i in np.nonzero(m)[0]], 0)
        out[f"base_{g}"] = (hue[m].mean(0) >= 0.85) & (X < 60)
        out[f"clamp_{g}"] = (dark >= 0.9) & (X < 60) & (Y > 330)
        print(g, "base px", out[f"base_{g}"].sum(), "clamp px", out[f"clamp_{g}"].sum())
    np.savez_compressed(HERE / "masks" / "orangebox_side_static.npz", **out)


def apply(tag_in, tag_out, add_base=True, cut_clamp=True):
    st = np.load(HERE / "masks" / "orangebox_side_static.npz")
    hue = unpack(np.load(HERE / "masks" / "orangebox_side_hue.npy"))
    m = unpack(np.load(HERE / "masks" / f"orangebox_side{tag_in}.npy"))
    for g, sel in GROUPS.items():
        for i in np.nonzero(sel)[0]:
            if cut_clamp:
                m[i] &= ~st[f"clamp_{g}"]
            if add_base:
                m[i] |= st[f"base_{g}"] & hue[i]
    np.save(HERE / "masks" / f"orangebox_side{tag_out}.npy", np.packbits(m, axis=-1))
    top = HERE / "masks" / f"orangebox_top{tag_in}.npy"
    (HERE / "masks" / f"orangebox_top{tag_out}.npy").write_bytes(top.read_bytes())


if __name__ == "__main__":
    build()
    if len(sys.argv) > 2:
        apply(sys.argv[1], sys.argv[2])
