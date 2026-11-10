"""Fit the static scene of the ricky twin (mat rectangle, wooden table rectangle) to the per-session median real
frames, through the frozen fitted cameras: every pixel ray (1/4 grid) is cast onto the table plane and the projected
rectangles are scored by IoU against colour masks (mat = dark grey, table = wood or mat). Writes work/static_fit.json."""
import json, pathlib, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import camgeo

FIT = HERE.parents[0] / "fitkit/out/ricky_so101_sess.json"
KEYS = ("x", "y", "z", "yaw", "pitch", "roll", "fovy", "k1", "k2", "cx", "cy")
st = json.loads(FIT.read_text())
S = 4
Z_MAT = 0.002
# generous boxes (full-res rows, cols) where the bowl / arm base / cable sit in every session: scored in neither mask
EXCL = {"top": [(370, 620, 790, 1200), (380, 500, 520, 800)], "side": [(190, 360, 590, 870), (250, 470, 150, 470)]}


def masks(med, cam):
    m = med.astype(int)
    hsv = np.asarray(Image.fromarray(med).convert("HSV")).astype(int)
    mx = m.max(-1)
    dark = (mx < (80 if cam == "top" else 100)) & (m[..., 2] - m[..., 0] < 12)
    dark = ndi.binary_closing(dark, iterations=6)          # the grey logo on the mat
    wood = (hsv[..., 0] >= 4) & (hsv[..., 0] <= 30) & (hsv[..., 1] >= 30) & (hsv[..., 1] <= 150) & (hsv[..., 2] >= 90)
    return dark, wood


def rect_in(P, r):
    cx, cy, hx, hy, yaw = r
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = P[..., 0] - cx, P[..., 1] - cy
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return (np.abs(lx) <= hx) & (np.abs(ly) <= hy)


views = {}
for v, d in st["views"].items():
    p = [d[k] for k in KEYS]
    med = np.load(HERE / "work" / f"median_{v}.npy")
    H, W = med.shape[:2]
    cam = d["cam"]
    dark, wood = masks(med, cam)
    valid = np.ones((H, W), bool)
    for y0, y1, x0, x1 in EXCL[cam]:
        valid[y0:y1, x0:x1] = False
    vv, uu = np.mgrid[S // 2:H:S, S // 2:W:S]
    P = camgeo.cast_plane_z(p, uu.ravel().astype(float), vv.ravel().astype(float), H, W, Z_MAT).reshape(vv.shape + (3,))
    ok = np.isfinite(P[..., 0]) & valid[vv, uu]
    views[v] = dict(P=P, ok=ok, dark=dark[vv, uu], table=(dark | wood)[vv, uu], cam=cam)


def iou(a, b, ok):
    a, b = a & ok, b & ok
    return (a & b).sum() / max((a | b).sum(), 1)


def loss(r, key):
    return 1 - np.mean([iou(rect_in(V["P"], r), V[key], V["ok"]) for V in views.values()])


out = {}
for key, r0 in (("dark", [0.0, 0.27, 0.42, 0.24, 0.0]), ("table", [0.05, 0.1, 0.6, 0.45, 0.0])):
    best = None
    for dyaw in (0.0, 0.05, -0.05):
        x0 = np.array(r0, float); x0[4] += dyaw
        res = minimize(lambda x: loss(x, key), x0, method="Nelder-Mead",
                       options=dict(initial_simplex=np.vstack([x0] + [x0 + np.eye(5)[k] * [0.05, 0.05, 0.05, 0.05, 0.05][k] for k in range(5)]),
                                    xatol=1e-4, fatol=1e-6, maxfev=1500))
        if best is None or res.fun < best.fun:
            best = res
    per = {v: round(float(iou(rect_in(V["P"], best.x), V[key], V["ok"])), 3) for v, V in views.items()}
    out["mat" if key == "dark" else "table"] = dict(center=best.x[:2].tolist(), half=best.x[2:4].tolist(), yaw_deg=float(np.degrees(best.x[4])),
                                                     iou_per_view=per)
    print(key, np.round(best.x, 4), "yaw deg", round(float(np.degrees(best.x[4])), 2), per, flush=True)
(HERE / "work" / "static_fit.json").write_text(json.dumps(out, indent=1))
