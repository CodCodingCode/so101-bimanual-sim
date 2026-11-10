"""Wooden-cube centroid per real frame and camera (colour blob), used to seed the cameras by PnP on the gripper.

    .venv/bin/python so101_twin/stack2/cube_track.py      # -> out/cube_track.npz (cx, cy, area per cam), out/cube_track.jpg

The 4 cm wooden cube is the only wood-coloured object; while it is held (builder's cube timeline: 'jaws') its centre
is a few mm from the gripper's tool centre point, so (tcp(state), cube centroid) pairs are 3D-2D correspondences that
do not depend on any arm mask.
"""
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "neuroscope" / "molmoact2"))
from sets import Store  # noqa: E402


def hsv(img):
    x = img.astype(np.float32) / 255
    mx = x.max(-1)
    mn = x.min(-1)
    d = mx - mn + 1e-6
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, d / (mx + 1e-6), mx


EXCLUDE = {"top": (slice(0, 150), slice(500, 640))}   # static cardboard next to the parked structure (top view)


def wood(img, cam=None):
    h, s, v = hsv(img)
    m = (h > 12) & (h < 45) & (s > 0.28) & (v > 0.35)
    if cam in EXCLUDE:
        m[EXCLUDE[cam]] = False
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        return None
    sizes = ndi.sum(m, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < 150:
        return None
    ys, xs = np.nonzero(lab == k)
    return xs.mean(), ys.mean(), float(len(xs)), (xs.min(), ys.min(), xs.max(), ys.max())


def main():
    fs = Store().frames("so101_stack_real")
    out = {}
    tiles = []
    for c in fs.cameras:
        res = np.full((fs.n, 7), np.nan)
        for i in range(fs.n):
            w = wood(np.asarray(fs.obs(i)["images"][c]), c)
            if w is not None:
                res[i, :3] = w[:3]
                res[i, 3:] = w[3]
        out[c] = res
        print(c, "cube found in", int(np.isfinite(res[:, 0]).sum()), "frames", flush=True)
    np.savez(HERE / "out" / "cube_track.npz", **out)
    for i in np.linspace(0, fs.n - 1, 8).astype(int):
        row = []
        for c in fs.cameras:
            im = np.asarray(fs.obs(int(i))["images"][c]).copy()
            r = out[c][i]
            if np.isfinite(r[0]):
                x0, y0, x1, y1 = r[3:].astype(int)
                im[y0:y1 + 1, [x0, x1]] = (255, 0, 255)
                im[[y0, y1], x0:x1 + 1] = (255, 0, 255)
            row.append(im)
        tiles.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(tiles, 0)).resize((640, 1920)).save(HERE / "out" / "cube_track.jpg", quality=85)


if __name__ == "__main__":
    main()
