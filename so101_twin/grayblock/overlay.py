"""GRAYBLOCK copy of jar/overlay.py. JAR: IoU overlays of a params json on chosen frames (white = both, red = real mask only, blue = twin only).
    .venv/bin/python so101_twin/grayblock/overlay.py out/<x>.json --mask-tag _d1c [--idx 1,3,...] [--out out/ov.jpg]"""
import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit")
    ap.add_argument("--mask-tag", default="_d1c")
    ap.add_argument("--idx", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    sc = F.load_scene("grayblock")
    fj = json.loads(pathlib.Path(a.fit).read_text())
    sc.ARM = dataclasses.replace(sc.ARM, model=a.model or fj.get("model", sc.ARM.model))
    P = F.reconcile(sc, fj["params"])
    rig = F.Rig(sc, P)
    fs = F.real_set(sc)
    idx = [int(x) for x in a.idx.split(",")] if a.idx else np.linspace(1, fs.n - 2, 8).astype(int).tolist()
    real = {c: F.load_masks("grayblock", c, a.mask_tag) for c in sc.CAMERAS}
    rows = []
    for i in idx:
        tiles = []
        for c in sc.CAMERAS:
            im = np.asarray(fs.obs(i)["images"][c]).astype(np.float32)
            s = rig.mask(c, "all", F.frame_state(sc, P, c, i), 640, 480)
            r = real[c][i]
            ov = np.zeros((480, 640, 3), np.float32)
            ov[r & s] = 255
            ov[r & ~s] = (255, 40, 40)
            ov[~r & s] = (40, 90, 255)
            al = (r | s)[..., None]
            t = Image.fromarray(np.where(al, 0.4 * im + 0.6 * ov, im).astype(np.uint8))
            iou = (s & r).sum() / max((s | r).sum(), 1)
            ImageDraw.Draw(t).text((6, 6), f"{i} ep{fs.labels['episode'][i]} {c} IoU {iou:.3f}", fill=(255, 255, 0))
            tiles.append(np.asarray(t))
        rows.append(np.concatenate(tiles, 1))
    g = Image.fromarray(np.concatenate(rows, 0))
    g = g.resize((g.width // 2, g.height // 2), Image.BILINEAR)
    g.save(a.out or HERE / "out" / f"ov_{pathlib.Path(a.fit).stem}.jpg", quality=85)


if __name__ == "__main__":
    main()
