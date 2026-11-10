"""GRAYBLOCK: contact sheet of real frames (diagnostic). .venv python so101_twin/grayblock/sheet_real.py CAM OUT [frames...]"""
import sys, pathlib
import numpy as np
from PIL import Image, ImageDraw
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
from sets import Store
fs = Store().frames("so101_grayblock_real")
cam, out = sys.argv[1], sys.argv[2]
idx = [int(x) for x in sys.argv[3:]] or list(np.linspace(0, fs.n - 1, 12).astype(int))
tiles = []
for i in idx:
    im = Image.fromarray(np.asarray(fs.obs(i)["images"][cam])).resize((320, 240))
    d = ImageDraw.Draw(im); d.text((4, 4), f"{i} ep{fs.labels['episode'][i]} t{fs.labels['t'][i]}", fill=(255, 0, 0))
    tiles.append(np.asarray(im))
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
rows = [np.concatenate(tiles[j:j + 4], 1) for j in range(0, len(tiles), 4)]
Image.fromarray(np.concatenate(rows, 0)).save(out, quality=85)
