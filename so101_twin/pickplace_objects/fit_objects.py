"""Per-episode placement of the pickplace box (orange rim) and pink brick through the NEW fitted cameras
(fitkit/out/pickplace_so101_sam1.json). Detection: colour masks on each episode's first frame (orange rim, pink
brick), with the twin arm's silhouette (stored arm_mask, dilated) removed so the orange LeRobot sticker and red wires
never count. Models: the builder's box (outer size sx x sy, rim square section 6 mm along the bottom edges) and
2x4 brick (hx, hy, hz half sizes); shapes shared by all episodes, (x, y, yaw) per episode, plus one shared support
plane height z (as for ricky). Writes work/objects_fit.json."""
import json, pathlib, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import camgeo, objfit

FIT = ROOT / "so101_twin/fitkit/out/pickplace_so101_sam1.json"
KEYS = ("x", "y", "z", "yaw", "pitch", "roll", "fovy", "k1", "k2", "cx", "cy")
st = json.loads(FIT.read_text())
P = {v: [d[k] for k in KEYS] for v, d in st["views"].items()}
REAL = ROOT / "neuroscope/molmoact2/store/sets/so101_pickplace_real"
TW = ROOT / "neuroscope/molmoact2/store/sets/so101_pickplace_twin2"
lab = np.load(REAL / "labels.npz")
EP = lab["episode"]
first = {int(e): int(np.flatnonzero(EP == e)[np.argmin(lab["t"][EP == e])]) for e in np.unique(EP)}
RIM = 0.006
ZOBJ = [0.0]


def colour_masks(i, c):
    rgb = np.asarray(np.load(REAL / f"images_{c}.npy", mmap_mode="r")[i])
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    W = rgb.shape[1]
    arm = np.unpackbits(np.asarray(np.load(TW / f"arm_mask_{c}.npy", mmap_mode="r")[i]), axis=-1)[..., :W].astype(bool)
    arm = ndi.binary_dilation(arm, iterations=8)
    orange = (h >= 8) & (h <= 35) & (s >= 90) & (v >= 100) & ~arm
    pink = ((h >= 235) | (h <= 10)) & (s >= 100) & (v >= 90) & ~arm
    out = {}
    for name, m, mn in (("rim", orange, 40), ("brick", pink, 30)):
        m = ndi.binary_opening(m, iterations=1)
        lab_, n = ndi.label(ndi.binary_closing(m, iterations=3))
        if n == 0:
            out[name] = m & False
            continue
        sz = ndi.sum(m, lab_, range(1, n + 1))
        if name == "brick":
            keep = 1 + int(np.argmax(sz))
            out[name] = m & (lab_ == keep) if sz.max() >= mn else m & False
        else:
            out[name] = m & np.isin(lab_, 1 + np.flatnonzero(sz >= mn))
    return out


H, W = 480, 640
data = {e: {c: colour_masks(i, c) for c in ("top", "side")} for e, i in first.items()}


def rim_pts_list(pose, shape):
    x, y, yaw = pose
    sx, sy = shape
    r = RIM / 2
    bars = [((-sx / 2 + r, 0), (r, sy / 2)), ((sx / 2 - r, 0), (r, sy / 2)), ((0, -sy / 2 + r), (sx / 2, r)), ((0, sy / 2 - r), (sx / 2, r))]
    c, s = np.cos(yaw), np.sin(yaw)
    out = []
    for (lx, ly), (hx, hy) in bars:
        out.append(objfit.box_pts(x + c * lx - s * ly, y + s * lx + c * ly, ZOBJ[0] + r, hx, hy, r, yaw))
    return out


def sil(v, pts_list, crop):
    m = None
    for pts in pts_list:
        a = objfit.hull_mask(P[v], pts, H, W, crop)
        m = a if m is None else (m | a)
    return m


def model(name, pose, shape):
    if name == "rim":
        return rim_pts_list(pose, shape)
    zc_ = pose[3] if len(pose) > 3 else ZOBJ[0] + shape[2]
    return [objfit.box_pts(pose[0], pose[1], zc_, shape[0], shape[1], shape[2], pose[2])]


def ep_loss(name, e, pose, shape, cams=("top", "side")):
    L = []
    for c in cams:
        m = data[e][c][name]
        if not m.any():
            continue
        crop = objfit.crop_of(m, 40, H, W)
        L.append(1 - objfit.iou(sil(c, model(name, pose, shape), crop), m[crop[0]:crop[1], crop[2]:crop[3]]))
    return float(np.mean(L)) if L else 1.0


def init(name, e):
    m = data[e]["top"][name]
    yx = np.argwhere(m).mean(0)
    X = camgeo.cast_plane_z(P["top"], [yx[1]], [yx[0]], H, W, 0.005)[0]
    return np.array([X[0], X[1], 0.0])



# ---- v2: triangulation + chamfer (the IoU fit of thin rims / small bricks stalls when the start does not overlap)
from scipy.ndimage import distance_transform_edt as edt


def tri(e):
    rows = []
    for c in ("top", "side"):
        m = data[e][c]["brick"]
        if not m.any():
            return None
        yx = np.argwhere(m).mean(0)
        d = camgeo.rays(P[c], [yx[1]], [yx[0]], H, W)[0]
        rows.append((np.asarray(P[c][:3]), d))
    A = sum(np.eye(3) - np.outer(d, d) for o, d in rows)
    b = sum((np.eye(3) - np.outer(d, d)) @ o for o, d in rows)
    X = np.linalg.solve(A, b)
    return X, [float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - o))) for o, d in rows]


shapes = {"rim": np.array([0.13, 0.11]), "brick": np.array([0.0159, 0.0079, 0.0055])}   # builder sizes
eps = sorted(data)
T3 = {e: tri(e) for e in eps}
zc = np.median([t[0][2] for t in T3.values() if t is not None])
ZOBJ[0] = float(zc - shapes["brick"][2])
print("brick triangulated in", sum(t is not None for t in T3.values()), "episodes; median centre z %.4f -> support plane z %.4f" % (zc, ZOBJ[0]), flush=True)
poses = {"rim": {}, "brick": {}}
for e in eps:
    if T3[e] is not None:
        x, y = T3[e][0][:2]
    else:
        m = data[e]["top"]["brick"]
        yx = np.argwhere(m).mean(0)
        x, y = camgeo.cast_plane_z(P["top"], [yx[1]], [yx[0]], H, W, zc)[0][:2]
    yaws = np.linspace(0, np.pi, 37)
    zc_e = float(np.clip(T3[e][0][2], shapes["brick"][2], 0.035)) if T3[e] is not None else float(zc)
    L = [ep_loss("brick", e, [x, y, a, zc_e], shapes["brick"], ("top",)) for a in yaws]
    a0 = yaws[int(np.argmin(L))]
    r = minimize(lambda q: ep_loss("brick", e, [x, y, q[0], zc_e], shapes["brick"], ("top",)), [a0], method="Nelder-Mead",
                 options=dict(initial_simplex=[[a0], [a0 + 0.05]], maxfev=60))
    zc_e = float(np.clip(T3[e][0][2], shapes["brick"][2], 0.035)) if T3[e] is not None else float(zc)
    poses["brick"][e] = np.array([x, y, float(r.x[0]), zc_e])


def chamfer(e, pose, shape, cams=("top", "side")):
    """Mean distance (px) from detected rim pixels to the model rim silhouette (one-sided: the far rim edges are
    often too faint to detect, so only the detected pixels must lie on the model)."""
    out = []
    for c in cams:
        m = data[e][c]["rim"]
        if not m.any():
            continue
        crop = objfit.crop_of(m, 80, H, W)
        mm = sil(c, rim_pts_list(pose, shape), crop)
        if not mm.any():
            out.append(80.0)
            continue
        dt = edt(~mm)
        out.append(float(dt[m[crop[0]:crop[1], crop[2]:crop[3]]].mean()))
    return float(np.mean(out)) if out else 80.0


x0 = np.array([-0.15, 0.02, 0.0])
for e in eps:
    best = None
    for s0 in (x0, x0 + [0.03, 0, 0], x0 + [0, -0.03, 0], x0 + [-0.03, 0, 0], x0 + [0, 0.03, 0]):
        r = minimize(lambda p: chamfer(e, p, shapes["rim"]), s0, method="Nelder-Mead",
                     options=dict(initial_simplex=np.vstack([s0] + [s0 + np.eye(3)[j] * [0.01, 0.01, 0.05][j] for j in range(3)]), xatol=1e-5, fatol=1e-4, maxfev=300))
        if best is None or r.fun < best.fun:
            best = r
    poses["rim"][e] = best.x
sub = eps[::5]
r = minimize(lambda sh: np.mean([chamfer(e, poses["rim"][e], sh) for e in sub]) if (sh > 0.05).all() else 99, shapes["rim"], method="Nelder-Mead",
             options=dict(initial_simplex=np.vstack([shapes["rim"], shapes["rim"] + [0.01, 0], shapes["rim"] + [0, 0.01]]), maxfev=80))
shapes["rim"] = r.x
for e in eps:
    s0 = poses["rim"][e]
    r = minimize(lambda p: chamfer(e, p, shapes["rim"]), s0, method="Nelder-Mead",
                 options=dict(initial_simplex=np.vstack([s0] + [s0 + np.eye(3)[j] * [0.005, 0.005, 0.03][j] for j in range(3)]), xatol=1e-5, fatol=1e-4, maxfev=200))
    poses["rim"][e] = r.x
print("rim size", np.round(shapes["rim"], 4), "chamfer px median", round(float(np.median([chamfer(e, poses["rim"][e], shapes["rim"]) for e in eps])), 2), flush=True)

out = dict(z_objects=ZOBJ[0], rim=RIM, shapes={n: shapes[n].tolist() for n in shapes}, method="brick: two-view centroid triangulation gives (x, y, centre z) per episode (z clipped to [hz, 3.5 cm]; top-view ray on the median height when the side view misses it), yaw from top-view IoU; box: one-sided chamfer of the detected orange rim to the model rim in both views", episodes={})
for e in eps:
    rec = {}
    for n in ("rim", "brick"):
        r = dict(pose=[float(x) for x in poses[n][e]])
        for c in ("top", "side"):
            m = data[e][c][n]
            if not m.any():
                r[c] = None
                continue
            crop = objfit.crop_of(m, 40, H, W)
            mm = sil(c, model(n, poses[n][e], shapes[n]), crop)
            mc = m[crop[0]:crop[1], crop[2]:crop[3]]
            ctr = np.concatenate(model(n, poses[n][e], shapes[n])).mean(0, keepdims=True)
            depth = float(camgeo.project(P[c], ctr, H, W)[2][0])
            f = camgeo.focal(P[c], H)
            if n == "brick":
                ce = float(np.linalg.norm(np.argwhere(mm).mean(0) - np.argwhere(mc).mean(0))) if mm.any() else float("nan")
                r[c] = dict(iou=round(float(objfit.iou(mm, mc)), 3), centroid_px=round(ce, 2), centroid_cm=round(100 * ce * depth / f, 2))
            else:
                ch = chamfer(e, poses[n][e], shapes[n], (c,))
                r[c] = dict(chamfer_px=round(ch, 2), chamfer_cm=round(100 * ch * depth / f, 2))
        if n == "brick" and T3[e] is not None:
            r["ray_gap_cm"] = [round(100 * g, 2) for g in T3[e][1]]
        rec[n] = r
    out["episodes"][int(e)] = rec
for c in ("top", "side"):
    rr = [out["episodes"][e]["brick"][c] for e in out["episodes"] if out["episodes"][e]["brick"][c]]
    print("brick", c, "n %d IoU median %.3f | centroid px median %.1f max %.1f | cm median %.2f" % (len(rr), np.median([x["iou"] for x in rr]), np.median([x["centroid_px"] for x in rr]), np.max([x["centroid_px"] for x in rr]), np.median([x["centroid_cm"] for x in rr])))
    rr = [out["episodes"][e]["rim"][c] for e in out["episodes"] if out["episodes"][e]["rim"][c]]
    print("rim", c, "n %d chamfer px median %.1f max %.1f | cm median %.2f" % (len(rr), np.median([x["chamfer_px"] for x in rr]), np.max([x["chamfer_px"] for x in rr]), np.median([x["chamfer_cm"] for x in rr])))
(HERE / "work" / "objects_fit.json").write_text(json.dumps(out, indent=1))
