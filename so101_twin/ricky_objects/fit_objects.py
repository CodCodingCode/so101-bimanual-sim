"""Per-episode placement of the ricky bowl and block from the Grounding-DINO + SAM masks of each episode's first
frame in both cameras, through the frozen fitted cameras. Bowl = truncated cone (rb bottom, rt top, h), block = box
(hx, hy, hz half sizes); shapes are shared by all episodes, positions (and block yaw) are per episode.
Writes work/objects_fit.json."""
import json, pathlib, sys
import numpy as np
from scipy.optimize import minimize
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import camgeo, objfit

FIT = HERE.parents[0] / "fitkit/out/ricky_so101_sess.json"
KEYS = ("x", "y", "z", "yaw", "pitch", "roll", "fovy", "k1", "k2", "cx", "cy")
Z0 = 0.002     # mat top (visible table plane)
ZOBJ = [0.002]  # object support plane: fitted (shared by every episode and both objects), see README
st = json.loads(FIT.read_text())
P = {v: [d[k] for k in KEYS] for v, d in st["views"].items()}
view_of = {}
for v, d in st["views"].items():
    for e in d["episodes"]:
        view_of.setdefault(d["cam"], {})[int(e)] = v
z = np.load(HERE / "work" / "seg_first.npz", allow_pickle=True)
H, W = map(int, z["shape"])
lab = np.load(HERE.parents[1] / "neuroscope/molmoact2/store/sets/so101_ricky_real/labels.npz")
ep_of = lab["episode"]
OBJ = {"bowl": "a white bowl.", "block": "a small brown block."}

data = {}
for k in range(len(z["frame"])):
    i, c, pr = int(z["frame"][k]), str(z["cam"][k]), str(z["prompt"][k])
    m = np.unpackbits(z["bits"][k], axis=-1)[..., :W].astype(bool)
    name = [o for o, s in OBJ.items() if s == pr][0]
    e = int(ep_of[i])
    data.setdefault(name, {}).setdefault(e, {})[c] = m


def model_pts(name, pose, shape):
    if name == "bowl":
        return objfit.frustum_pts(pose[0], pose[1], ZOBJ[0], shape[0], shape[1], shape[2])
    return objfit.box_pts(pose[0], pose[1], ZOBJ[0] + shape[2], shape[0], shape[1], shape[2], pose[2])


def ep_loss(name, e, pose, shape, cams=("top", "side")):
    L = []
    for c in cams:
        m = data[name][e].get(c)
        if m is None or not m.any():
            continue
        crop = objfit.crop_of(m, 60, H, W)
        L.append(1 - objfit.iou(objfit.hull_mask(P[view_of[c][e]], model_pts(name, pose, shape), H, W, crop), m[crop[0]:crop[1], crop[2]:crop[3]]))
    return float(np.mean(L)) if L else 1.0


def init_pose(name, e):
    m = data[name][e]["side"]
    yx = np.argwhere(m).mean(0)
    zc = 0.03 if name == "bowl" else 0.012
    X = camgeo.cast_plane_z(P[view_of["side"][e]], [yx[1]], [yx[0]], H, W, zc)[0]
    return [X[0], X[1]] + ([] if name == "bowl" else [0.0])


shapes = {"bowl": np.array([0.065, 0.08, 0.065]), "block": np.array([0.025, 0.025, 0.014])}
eps = sorted(data["bowl"])
poses = {n: {e: np.array(init_pose(n, e)) for e in eps} for n in OBJ}
for rnd in range(4):
    for n in OBJ:
        for e in eps:
            x0 = poses[n][e]
            stp = [0.01, 0.01] + ([0.3] if n == "block" else [])
            best = None
            starts = [x0] if n == "bowl" else [x0, x0 + [0, 0, np.pi / 4]]
            for s0 in starts:
                r = minimize(lambda x: ep_loss(n, e, x, shapes[n]), s0, method="Nelder-Mead",
                             options=dict(initial_simplex=np.vstack([s0] + [s0 + np.eye(len(s0))[j] * stp[j] for j in range(len(s0))]),
                                          xatol=1e-5, fatol=1e-6, maxfev=400))
                if best is None or r.fun < best.fun:
                    best = r
            poses[n][e] = best.x
        if rnd < 3:
            sub = eps[::3]
            f = lambda sh: np.mean([ep_loss(n, e, poses[n][e], sh) for e in sub]) if (sh > 0.003).all() else 1.0
            x0 = shapes[n]
            r = minimize(f, x0, method="Nelder-Mead",
                         options=dict(initial_simplex=np.vstack([x0] + [x0 + np.eye(3)[j] * 0.005 for j in range(3)]), xatol=1e-5, fatol=1e-6, maxfev=300))
            shapes[n] = r.x
        if rnd < 3 and n == "block":
            sub = eps[::3]
            fz = lambda zz: np.mean([ep_loss(m_, e, poses[m_][e], shapes[m_]) for m_ in OBJ for e in sub]) if (ZOBJ.__setitem__(0, float(zz[0])) is None) else 1.0
            r = minimize(fz, [ZOBJ[0]], method="Nelder-Mead", options=dict(initial_simplex=[[ZOBJ[0]], [ZOBJ[0] + 0.01]], xatol=1e-5, fatol=1e-6, maxfev=60))
            ZOBJ[0] = float(r.x[0])
            print("round", rnd, "object plane z", round(ZOBJ[0], 4), flush=True)
        print("round", rnd, n, "shape", np.round(shapes[n], 4), "mean IoU", round(1 - np.mean([ep_loss(n, e, poses[n][e], shapes[n]) for e in eps]), 3), flush=True)

# residuals: per camera IoU and centroid error (px), metric equivalent at the object's depth (cm)
out = dict(z_table=Z0, z_objects=ZOBJ[0], shapes={n: shapes[n].tolist() for n in OBJ}, episodes={})
for e in eps:
    rec = {}
    for n in OBJ:
        pose = poses[n][e]
        r = dict(pose=[float(x) for x in pose])
        for c in ("top", "side"):
            m = data[n][e][c]
            v = view_of[c][e]
            crop = objfit.crop_of(m, 60, H, W)
            pts = model_pts(n, pose, shapes[n])
            mm = objfit.hull_mask(P[v], pts, H, W, crop)
            ce = objfit.centroid_err(P[v], pts, m[crop[0]:crop[1], crop[2]:crop[3]], crop, H, W)
            depth = float(camgeo.project(P[v], pts.mean(0, keepdims=True), H, W)[2][0])
            f = camgeo.focal(P[v], H)
            r[c] = dict(iou=round(float(objfit.iou(mm, m[crop[0]:crop[1], crop[2]:crop[3]])), 3), centroid_px=round(ce, 2),
                        centroid_cm=round(100 * ce * depth / f, 2), depth_m=round(depth, 3))
        # single-view fits: how far apart the two cameras put the object (cm), a direct 3D consistency check
        sv = {}
        for c in ("top", "side"):
            x0 = pose.copy()
            stp = [0.01, 0.01] + ([0.3] if n == "block" else [])
            rr = minimize(lambda x: ep_loss(n, e, x, shapes[n], (c,)), x0, method="Nelder-Mead",
                          options=dict(initial_simplex=np.vstack([x0] + [x0 + np.eye(len(x0))[j] * stp[j] for j in range(len(x0))]), xatol=1e-5, fatol=1e-6, maxfev=300))
            sv[c] = rr.x[:2]
        r["top_vs_side_cm"] = round(float(100 * np.linalg.norm(sv["top"] - sv["side"])), 2)
        rec[n] = r
    out["episodes"][int(e)] = rec
for n in OBJ:
    for c in ("top", "side"):
        a = np.array([out["episodes"][e][n][c]["iou"] for e in out["episodes"]])
        px = np.array([out["episodes"][e][n][c]["centroid_px"] for e in out["episodes"]])
        cm = np.array([out["episodes"][e][n][c]["centroid_cm"] for e in out["episodes"]])
        print(n, c, "IoU median %.3f min %.3f | centroid px median %.1f max %.1f | cm median %.2f max %.2f" % (np.median(a), a.min(), np.nanmedian(px), np.nanmax(px), np.nanmedian(cm), np.nanmax(cm)))
    d = np.array([out["episodes"][e][n]["top_vs_side_cm"] for e in out["episodes"]])
    print(n, "top-only vs side-only fit distance cm: median %.2f max %.2f" % (np.median(d), d.max()))
(HERE / "work" / "objects_fit.json").write_text(json.dumps(out, indent=1))
