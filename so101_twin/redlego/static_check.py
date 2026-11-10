"""REDLEGO copy of jar/static_check.py. Static-camera check: per-episode median image vs the global median, phase correlation shift (px) per camera."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
from sets import Store  # noqa: E402


def phase_shift(a, b):
    A = np.fft.fft2(a - a.mean())
    B = np.fft.fft2(b - b.mean())
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-9
    r = np.fft.ifft2(R).real
    y, x = np.unravel_index(np.argmax(r), r.shape)
    h, w = r.shape
    if y > h // 2:
        y -= h
    if x > w // 2:
        x -= w
    return int(y), int(x), float(r.max())


def main():
    fs = Store().frames("so101_redlego_real")
    ep = fs.labels["episode"].astype(int)
    for cam in ("top", "side"):
        imgs = np.load(fs.path / f"images_{cam}.npy", mmap_mode="r")
        g = np.median(np.asarray(imgs[::2]).astype(np.float32).mean(-1), axis=0)
        np.save(pathlib.Path(__file__).parent / "out" / f"bg_global_{cam}.npy", g)
        out = []
        for e in sorted(set(ep)):
            idx = np.nonzero(ep == e)[0]
            m = np.median(np.asarray(imgs[idx]).astype(np.float32).mean(-1), axis=0)
            # use only the outer frame (arm mostly central) to reduce the arm's influence
            dy, dx, pk = phase_shift(m, g)
            out.append((e, dy, dx, pk, float(np.abs(m - g).mean())))
        print(cam, " ".join(f"e{e}:({dy},{dx}) {d:.1f}" for e, dy, dx, pk, d in out))


if __name__ == "__main__":
    main()
