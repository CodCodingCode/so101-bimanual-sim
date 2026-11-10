"""Contact sheet: real frames with mask outlines (green = mask A, red = mask B optional). Debug helper.

    .venv/bin/python so101_twin/ballcup/sheet.py <maskA.npy> [maskB.npy] --frames 0,40,... --out file.jpg
"""
import argparse
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))
from sets import Store  # noqa: E402


def load(p):
    return np.unpackbits(np.load(p), axis=-1)[..., :640].astype(bool)


def outline(m, w=1):
    return m & ~ndi.binary_erosion(m, iterations=w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", nargs="?", default="")
    ap.add_argument("--cam", default="")
    ap.add_argument("--frames", default="0,1,2,3,40,42,45,99,161,204,254,319")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fill", action="store_true")
    a = ap.parse_args()
    fs = Store().frames("so101_ballcup_real")
    cam = a.cam or ("top" if "_top" in a.a else "side")
    A = load(a.a)
    B = load(a.b) if a.b else None
    idx = [int(x) for x in a.frames.split(",")]
    tiles = []
    for i in idx:
        im = np.asarray(fs.obs(i)["images"][cam]).copy()
        if a.fill:
            im[A[i]] = (0.5 * im[A[i]] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        else:
            im[ndi.binary_dilation(outline(A[i]))] = (0, 255, 0)
        if B is not None:
            im[outline(B[i])] = (255, 0, 0)
        t = Image.fromarray(im).resize((320, 240))
        ImageDraw.Draw(t).text((4, 4), f"{i} ep{fs.labels['episode'][i]}", fill=(255, 255, 0))
        tiles.append(np.asarray(t))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + 4], 1) for r in range(0, len(tiles), 4)]
    Image.fromarray(np.concatenate(rows, 0)).save(a.out, quality=85)


if __name__ == "__main__":
    main()
