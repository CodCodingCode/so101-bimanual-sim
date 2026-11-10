"""Orange-arm hue masks (seed only, never scored): saturated orange pixels per frame and camera.

    .venv/bin/python so101_twin/orangebox/hue_masks.py   -> so101_twin/orangebox/masks/orangebox_<cam>_hue.npy (packed)
"""
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
sys.path.insert(0, str(HERE))
from sets import Store  # noqa: E402
from segment import dilate, erode  # noqa: E402

HUE = {"top": (8.0, 46.0), "side": (-12.0, 40.0)}


def hsv(im):
    x = im.astype(np.float32) / 255
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(-1)
    d = mx - x.min(-1) + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, d / (mx + 1e-6), mx


def hue_mask(im, cam):
    h, s, v = hsv(im)
    lo, hi = HUE[cam]
    h = np.where(h > 180, h - 360, h)
    m = (h >= lo) & (h <= hi) & (s > 0.5) & (v > 0.12)
    m = dilate(erode(m, 1), 1)
    return erode(dilate(m, 2), 2)


if __name__ == "__main__":
    fs = Store().frames("so101_orangebox_real")
    out = {}
    for cam in ("top", "side"):
        ms = np.stack([hue_mask(np.asarray(fs.obs(i)["images"][cam]), cam) for i in range(fs.n)])
        np.save(HERE / "masks" / f"orangebox_{cam}_hue.npy", np.packbits(ms, axis=-1))
        out[cam] = ms
        a = ms.reshape(fs.n, -1).mean(1)
        print(cam, "area mean %.4f min %.4f, frames < 0.1%%: %d" % (a.mean(), a.min(), (a < 0.001).sum()))
    idx = np.linspace(0, fs.n - 1, 12).astype(int)
    rows = []
    for i in idx:
        t = []
        for cam in ("top", "side"):
            im = np.asarray(fs.obs(int(i))["images"][cam]).copy()
            o = out[cam][i] & ~erode(out[cam][i], 1)
            im[dilate(o, 1)] = (0, 255, 0)
            t.append(im)
        rows.append(np.concatenate(t, 1))
    g = [np.concatenate(rows[k:k + 3], 1) for k in range(0, 12, 3)]
    Image.fromarray(np.concatenate(g, 0)).resize((1920 // 1, 1920 * 480 * 4 // (3 * 1280 * 1) // 2 * 1)).save(
        HERE / "out" / "hue_sheet.jpg", quality=85)
