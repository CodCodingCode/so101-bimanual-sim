"""REDLEGO: per-episode median images for a few episodes (camera / box / lighting changes) -> out/ep_medians_<cam>.jpg"""
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))
from sets import Store  # noqa: E402

fs = Store().frames("so101_redlego_real")
ep = fs.labels["episode"].astype(int)
eps = [int(x) for x in sys.argv[1].split(",")]
for cam in ("top", "side"):
    imgs = np.load(fs.path / f"images_{cam}.npy", mmap_mode="r")
    g = np.median(np.asarray(imgs[::2]).astype(np.float32), axis=0)
    tiles = []
    for e in eps:
        m = np.median(np.asarray(imgs[np.nonzero(ep == e)[0]]).astype(np.float32), axis=0)
        im = Image.fromarray(m.astype(np.uint8)).resize((320, 240))
        ImageDraw.Draw(im).text((4, 4), f"ep{e} mean {m.mean():.1f}", fill=(255, 0, 0))
        d = np.clip(np.abs(m - g).mean(-1) * 4, 0, 255).astype(np.uint8)
        dm = Image.fromarray(d).resize((320, 240)).convert("RGB")
        tiles.append(np.concatenate([np.asarray(im), np.asarray(dm)], 0))
    Image.fromarray(np.concatenate(tiles, 1)).save(HERE / "out" / f"ep_medians_{cam}.jpg", quality=85)
print("ok")
