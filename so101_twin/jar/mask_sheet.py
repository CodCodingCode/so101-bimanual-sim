"""JAR: sheet of real frames with mask outlines: green = mask A, red = mask B (optional), 16 frames, both cameras.
    .venv/bin/python so101_twin/jar/mask_sheet.py _d1 [--b TAG_OR_PROMPTDIR] [--idx 0,5,...] [--out out/x.jpg]"""
import argparse
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))


def load(p):
    return np.unpackbits(np.load(p), axis=-1)[..., :640].astype(bool)


def outline(m, r=1):
    return m & ~ndi.binary_erosion(m, iterations=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("--b", default="")
    ap.add_argument("--idx", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--cams", default="top,side")
    a = ap.parse_args()
    from sets import Store

    fs = Store().frames("so101_jar_real")
    idx = [int(x) for x in a.idx.split(",")] if a.idx else np.linspace(0, fs.n - 1, 16).astype(int).tolist()
    cams = a.cams.split(",")

    def get(tag, c):
        p = pathlib.Path(tag)
        if p.is_dir():
            return load(p / f"jar_{c}.npy")
        return load(HERE / "masks" / f"jar_{c}{tag}.npy")

    A = {c: get(a.a, c) for c in cams}
    B = {c: get(a.b, c) for c in cams} if a.b else None
    tiles = []
    for i in idx:
        row = []
        for c in cams:
            im = np.asarray(fs.obs(i)["images"][c]).copy()
            if B is not None:
                im[ndi.binary_dilation(outline(B[c][i]), iterations=1)] = (255, 40, 40)
            im[ndi.binary_dilation(outline(A[c][i]), iterations=1)] = (40, 255, 40)
            t = Image.fromarray(im)
            ImageDraw.Draw(t).text((6, 6), f"{i} {c}", fill=(255, 255, 0))
            row.append(np.asarray(t))
        tiles.append(np.concatenate(row, 1))
    per = 2 if len(cams) == 2 else 4
    rows = [np.concatenate(tiles[r:r + per], 1) for r in range(0, len(tiles), per)]
    g = Image.fromarray(np.concatenate(rows, 0))
    g = g.resize((g.width // 2, g.height // 2), Image.BILINEAR)
    g.save(a.out or HERE / "out" / f"sheet{a.a.replace('/', '_')}.jpg", quality=85)


if __name__ == "__main__":
    main()
