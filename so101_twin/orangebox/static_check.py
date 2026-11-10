"""Static-camera check: per-episode median image (arm mostly removed), phase correlation of its high-passed version
against the global reference, per camera.  .venv/bin/python so101_twin/orangebox/static_check.py"""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
from sets import Store

fs = Store().frames("so101_orangebox_real")
ep = fs.labels["episode"].astype(int)
eps = sorted(set(ep.tolist()))


def hp(g):
    g = g.astype(np.float32)
    k = 9
    c = np.pad(g, k, mode="edge").cumsum(0).cumsum(1)
    H, W = g.shape
    s = c[2 * k:2 * k + H, 2 * k:2 * k + W] - c[:H, 2 * k:2 * k + W] - c[2 * k:2 * k + H, :W] + c[:H, :W]
    return g - s / (2 * k) ** 2


def phase(a, b):
    w = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None]
    A, B = np.fft.fft2(a * w), np.fft.fft2(b * w)
    R = A * np.conj(B)
    r = np.fft.ifft2(R / (np.abs(R) + 1e-6)).real
    y, x = np.unravel_index(np.argmax(r), r.shape)
    H, W = r.shape
    return (x if x < W // 2 else x - W), (y if y < H // 2 else y - H), float(r.max())


for cam in ("top", "side"):
    meds = {}
    for e in eps:
        idx = np.nonzero(ep == e)[0]
        ims = np.stack([np.asarray(fs.obs(int(i))["images"][cam]).mean(-1) for i in idx])
        meds[e] = hp(np.median(ims, 0))
    ref = np.median(np.stack(list(meds.values())), 0)
    out = []
    for e in eps:
        dx, dy, pk = phase(meds[e], ref)
        out.append((e, dx, dy, round(pk, 3)))
    print(cam, out)
