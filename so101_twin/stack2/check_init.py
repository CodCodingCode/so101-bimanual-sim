"""Twin arm outline (red) over real frames, from the scene init (seed_cams.json) or a fit json.

    .venv/bin/python so101_twin/stack2/check_init.py OUT.jpg CAM [--fit-json F] [--dt -1.5] idx...
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
from segment import erode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("cam")
ap.add_argument("idx", nargs="*", type=int)
ap.add_argument("--fit-json", default="")
ap.add_argument("--dt", type=float, default=-1.5)
ap.add_argument("--mask-tag", default="")
a = ap.parse_intermixed_args()
sc = F.load_scene("stack")
if a.fit_json:
    P = F.reconcile(sc, json.loads(pathlib.Path(a.fit_json).read_text())["params"])
else:
    P = F.init_params(sc)
    for c in P["cams"]:
        P["cams"][c]["dt_frames"] = a.dt
rig = F.Rig(sc, P)
fs = F.real_set(sc)
real = None
if a.mask_tag:
    real = np.unpackbits(np.load(F.TWIN / "masks" / f"stack_{a.cam}{a.mask_tag}.npy"), axis=-1)[..., :640].astype(bool)
idx = a.idx or [0, 1, 2, 3, 6, 9, 10, 17]
tiles = []
for i in idx:
    g = F.frame_groups(sc, a.cam)[i]
    m = rig.mask(a.cam, g, F.frame_state(sc, P, a.cam, i), 640, 480)
    im = np.asarray(fs.obs(i)["images"][a.cam]).copy()
    if real is not None:
        o = real[i] & ~erode(real[i], 2)
        im[o] = (40, 255, 40)
    im[m & ~erode(m, 2)] = (255, 0, 0)
    tiles.append(im)
rows = [np.concatenate(tiles[k:k + 4], 1) for k in range(0, len(tiles), 4)]
Image.fromarray(np.concatenate(rows, 0)).resize((1280, 240 * len(rows))).save(a.out, quality=88)
print("wrote", a.out)
