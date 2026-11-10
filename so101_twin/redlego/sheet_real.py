"""REDLEGO: contact sheet of real frames (several episodes per camera) -> out/sheet_real_<cam>.jpg"""
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S = ROOT / "neuroscope/molmoact2/store/sets/so101_redlego_real"
lab = np.load(S / "labels.npz", allow_pickle=True)
ep = lab["episode"].astype(int)
idx = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(0, 320, 20))
for cam in ("top", "side"):
    imgs = np.load(S / f"images_{cam}.npy", mmap_mode="r")
    tiles = []
    for i in idx:
        im = Image.fromarray(np.asarray(imgs[i])).resize((320, 240))
        ImageDraw.Draw(im).text((4, 4), f"#{i} ep{ep[i]} t{lab['t'][i]}", fill=(255, 0, 0))
        tiles.append(np.asarray(im))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[j:j + 4], 1) for j in range(0, len(tiles), 4)]
    Image.fromarray(np.concatenate(rows, 0)).save(HERE / "out" / f"sheet_real_{cam}.jpg", quality=85)
print("ok")
