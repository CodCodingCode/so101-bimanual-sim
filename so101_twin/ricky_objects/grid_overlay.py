"""Overlay a world grid (z = z0 plane, 10 cm lines, x red / y green, labelled) on each session median."""
import json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import camgeo
fitj = sys.argv[1]; z0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
rng = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [-1.0, 1.0, -1.0, 1.0]
st = json.loads(pathlib.Path(fitj).read_text())
KEYS = ("x", "y", "z", "yaw", "pitch", "roll", "fovy", "k1", "k2", "cx", "cy")
for v, d in st["views"].items():
    p = [d[k] for k in KEYS]
    med = np.load(HERE / "work" / f"median_{v}.npy")
    h, w = med.shape[:2]
    im = Image.fromarray(med); dr = ImageDraw.Draw(im)
    for axis in (0, 1):
        for c in np.arange(rng[2 * axis], rng[2 * axis + 1] + 1e-9, 0.1):
            s = np.linspace(rng[2 * (1 - axis)], rng[2 * (1 - axis) + 1], 400)
            P = np.zeros((400, 3)); P[:, axis] = c; P[:, 1 - axis] = s; P[:, 2] = z0
            u, vv, z = camgeo.project(p, P, h, w)
            ok = (z > 0.05) & (u > -50) & (u < w + 50) & (vv > -50) & (vv < h + 50)
            col = (255, 60, 60) if axis == 0 else (60, 255, 60)
            pts = list(zip(u[ok], vv[ok]))
            if len(pts) > 1:
                dr.line(pts, fill=col, width=3 if abs(c) < 1e-9 else 1)
            # label at the middle visible point
            if ok.any():
                k = np.flatnonzero(ok)[len(np.flatnonzero(ok)) // 2]
                dr.text((u[k] + 3, vv[k] + 3), f"{'xy'[axis]}={c:+.1f}", fill=col)
    im.save(HERE / "work" / f"grid_{v}.jpg", quality=90)
    print(v)
