"""Twin outline (red) over real frames (+ optional mask outline, green) for both cameras: joint-map / camera check.

    .venv/bin/python so101_twin/ballcup/check.py [--fit out/X_fit.json] [--mask-tag _d1] [--roll 0] [--grip inv]
        [--frames ...] [--out out/check.jpg]
Without --fit: scene camera inits (cam_search results).
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402


def outline(m):
    return m & ~ndi.binary_erosion(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="")
    ap.add_argument("--mask-tag", default="")
    ap.add_argument("--roll", type=float, default=None)
    ap.add_argument("--model", default="")
    ap.add_argument("--frames", default="0,1,2,3,40,42,45,99,161,204,254,319")
    ap.add_argument("--out", default=str(HERE / "out" / "check.jpg"))
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    sc = F.load_scene("ballcup")
    P = json.loads(pathlib.Path(a.fit).read_text())["params"] if a.fit else F.init_params(sc)
    if a.model:
        import dataclasses
        sc.ARM = dataclasses.replace(sc.ARM, model=a.model)
    P = F.reconcile(sc, P)
    if a.roll is not None:
        P["offsets_deg"][4] = a.roll
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    real = {c: F.load_masks("ballcup", c, a.mask_tag) for c in sc.CAMERAS} if a.mask_tag else None
    idx = [int(x) for x in a.frames.split(",")]
    rows = []
    for i in idx:
        tiles = []
        for c in sc.CAMERAS:
            im = np.asarray(fs.obs(i)["images"][c]).copy()
            s = rig.mask(c, "all", F.frame_state(sc, P, c, i), 640, 480, i=i)
            txt = f"{i} ep{fs.labels['episode'][i]} {c}"
            if real is not None:
                r = real[c][i]
                im[ndi.binary_dilation(outline(r))] = (0, 255, 0)
                u = (s | r).sum()
                txt += f" IoU {(s & r).sum() / max(u, 1):.2f}"
            im[ndi.binary_dilation(outline(s))] = (255, 0, 0)
            t = Image.fromarray(im)
            ImageDraw.Draw(t).text((6, 6), txt, fill=(255, 255, 0))
            tiles.append(np.asarray(t))
        rows.append(np.concatenate(tiles, 1))
    k = 2
    grid = [np.concatenate(rows[j:j + k], 1) for j in range(0, len(rows) - len(rows) % k, k)]
    im = Image.fromarray(np.concatenate(grid, 0))
    im = im.resize((int(im.width * a.scale), int(im.height * a.scale)))
    im.save(a.out, quality=85)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
