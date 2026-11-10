"""GRAYBLOCK: object placement for the RGB render of the twin set (the metric never uses objects: arm-only masks).

    .venv/bin/python so101_twin/grayblock/objects.py --fit out/final_fit.json   # -> out/objects.json, out/objects_check.jpg

Inputs: the FITTED cameras (fit json) and the twin-independent DINO+SAM object masks of the real frames (masks/
grayblock_<cam>_d1_box.npy: SAM box-prompt mask of "a blue box."; masks/grayblock_<cam>_occ_block.npy: the filtered
"a gray block." masks of segment.py --mode occ). No arm mask, no fit objective.
Box: an open cuboid (lx, ly, h), placed per episode by ray-casting through the FITTED top camera: per episode the
majority (>= 50 % of its 8 frames) of the top view's SAM box masks and of its near blue face (masks/grayblock_top_occn:
the front wall); the front wall's lowest pixels -> rays -> z = 0 give the front-bottom edge (line fit: yaw, width lx);
its highest pixels -> rays -> the vertical plane of that edge give the height h; the box mask's highest pixels (the far
wall's rim) -> rays -> z = h give the depth ly (distance from the front edge). Sizes: median over episodes (shared),
pose per episode. Check: IoU of the projected box hull vs the majority box masks, both cameras.
Block (cube of side BLOCK) by phase from the full-rate gripper signal (this dataset's gripper starts closed, opens
around the block, holds it at ~15 and opens again over the box): before the hold interval on the table, during it in
the jaws (at the tool point of the posed twin), after it on the box floor. Positions from the block-mask centroids
(triangulated from both views when the rays meet within 3 cm, else the top-view ray at z = BLOCK / 2): the frame's
own detection when it agrees with the phase (outside / inside the box footprint), else the episode's median, else
(box) the box centre; a table frame without any detection in its episode is hidden.
Layout (RGB only): the wall behind the arm in the top view, from the table-wall edge of the top background median
ray-cast onto z = 0; the chair seen in the side view as a dark box (its silhouette cast onto a vertical plane).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import minimize

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
import geom as G  # noqa: E402

CAMS = ("top", "side")


def project(P, cam, X, w=640, h=480):
    """world points (n, 3) -> pixel indices (n, 2) with the fitted lens; also depth along the view axis."""
    p = P["cams"][cam]
    g = p["groups"]["all"]
    L = p["lens"]
    R = G.mat_from_quat(g["quat"])
    xc = (np.asarray(X, float) - np.asarray(g["pos"], float)) @ R
    z = -xc[:, 2]
    xu, yu = xc[:, 0] / z, -xc[:, 1] / z
    r2 = xu * xu + yu * yu
    s = 1 + L["k1"] * r2 + L["k2"] * r2 * r2
    f = 0.5 * h / math.tan(math.radians(p["fovy"]) / 2)
    return np.stack([xu * s * f + w / 2 + L["cx"] * w - 0.5, yu * s * f + h / 2 + L["cy"] * h - 0.5], 1), z


def hull(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def box_corners(x, y, yaw, lx, ly, h):
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dx in (-lx / 2, lx / 2):
        for dy in (-ly / 2, ly / 2):
            for z in (0.0, h):
                out.append((x + c * dx - s * dy, y + s * dx + c * dy, z))
    return np.array(out)


def box_sil(P, cam, prm, scale=4):
    uv, z = project(P, cam, box_corners(*prm))
    im = Image.new("L", (640 // scale, 480 // scale), 0)
    if (z > 0.02).all():
        ImageDraw.Draw(im).polygon([(u / scale, v / scale) for u, v in hull((uv + 0.5).tolist())], fill=1)
    return np.asarray(im, bool)


def small(m, k=4):
    n = m.shape[0]
    return m.reshape(n, 480 // k, k, 640 // k, k).mean((2, 4)) >= 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--block", type=float, default=0.03)
    a = ap.parse_args()
    sc = F.load_scene("grayblock")
    fj = json.loads(pathlib.Path(a.fit).read_text())
    sc.ARM = dataclasses.replace(sc.ARM, model=fj.get("model", sc.ARM.model))
    P = F.reconcile(sc, fj["params"])
    fs = F.real_set(sc)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    eps = sorted(set(ep.tolist()))
    boxm = {c: small(F.load_masks("grayblock", c, "_d1_box")) for c in CAMS}
    maj = {c: {e: boxm[c][ep == e].mean(0) >= 0.5 for e in eps} for c in CAMS}

    def iou(A, B):
        u = (A | B).sum()
        return (A & B).sum() / u if u else 0.0

    def loss_all(v, elist):
        lx, ly, h = v[3:6]
        if min(lx, ly, h) < 0.03:
            return 1.0
        return -np.mean([iou(box_sil(P, c, (v[0], v[1], v[2], lx, ly, h)), maj[c][e]) for c in CAMS for e in elist
                         if maj[c][e].sum() > 30])
    occn = small(F.load_masks("grayblock", "top", "_occn"), 1)
    boxfull = F.load_masks("grayblock", "top", "_d1_box")
    cam_top = np.asarray(P["cams"]["top"]["groups"]["all"]["pos"], float)
    raw = {}
    for e in eps:
        sel = ep == e
        Bm = boxfull[sel].mean(0) >= 0.5
        Fm = occn[sel].mean(0) >= 0.5
        if Fm.sum() < 200 or Bm.sum() < 500:
            continue
        cols = np.nonzero(Fm.any(0))[0]
        cols = cols[(cols > cols.min() + 4) & (cols < cols.max() - 4)]
        bot = np.array([np.nonzero(Fm[:, x])[0].max() for x in cols])
        top_ = np.array([np.nonzero(Fm[:, x])[0].min() for x in cols])
        Xb = np.array([G.hit_z(*G.cam_ray(P, "top", x, y + 0.5), 0.0) for x, y in zip(cols, bot)])
        c0 = Xb[:, :2].mean(0)
        u_, s_, vt = np.linalg.svd(Xb[:, :2] - c0)
        dirv = vt[0] if vt[0][0] >= 0 else -vt[0]
        tproj = (Xb[:, :2] - c0) @ dirv
        lx_e = float(tproj.max() - tproj.min())
        mid = c0 + dirv * 0.5 * (tproj.max() + tproj.min())
        nrm = np.array([-dirv[1], dirv[0]])
        if (cam_top[:2] - mid) @ nrm > 0:
            nrm = -nrm   # inward normal: away from the top camera
        # height: rays of the front face's top pixels hit the vertical plane through the front edge
        hs = []
        for x, y in zip(cols, top_):
            o, d = G.cam_ray(P, "top", x, y - 0.5)
            den = d[:2] @ nrm
            if abs(den) > 1e-6:
                t = ((mid - o[:2]) @ nrm) / den
                hs.append((o + t * d)[2])
        h_e = float(np.median(hs))
        # depth: the box mask's highest pixels (far rim) at z = h
        bcols = np.nonzero(Bm.any(0))[0]
        btop = np.array([np.nonzero(Bm[:, x])[0].min() for x in bcols])
        Xr = np.array([G.hit_z(*G.cam_ray(P, "top", x, y + 0.5), h_e) for x, y in zip(bcols, btop)])
        dep = (Xr[:, :2] - mid) @ nrm
        ly_e = float(np.percentile(dep, 80))
        yaw = math.atan2(dirv[1], dirv[0])
        raw[e] = (mid, nrm, yaw, lx_e, ly_e, h_e)
    lx = float(np.median([r[3] for r in raw.values()]))
    ly = float(np.median([r[4] for r in raw.values()]))
    h = float(np.median([r[5] for r in raw.values()]))
    print(f"box size lx {lx:.3f} ly {ly:.3f} h {h:.3f} m from {len(raw)} episodes", flush=True)
    box = {}
    ious = []
    for e in eps:
        r_ = raw.get(e) or next(iter(raw.values()))
        cen = r_[0] + r_[1] * ly / 2
        box[str(e)] = [float(cen[0]), float(cen[1]), float(r_[2])]
        prm = (cen[0], cen[1], r_[2], lx, ly, h)
        ious.append(np.mean([iou(box_sil(P, c, prm), maj[c][e]) for c in CAMS if maj[c][e].sum() > 30]))
    print("box per-episode hull IoU (both cams) median", round(float(np.median(ious)), 3), "min",
          round(float(np.min(ious)), 3), flush=True)
    # block
    B = a.block
    blk = {c: F.load_masks("grayblock", c, "_occ_block") for c in CAMS}
    rig = F.Rig(sc, P)
    import mujoco

    sid = mujoco.mj_name2id(rig.arm.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    # GRAYBLOCK: the block moves by phase (table -> jaws -> box). This dataset's gripper starts CLOSED (state ~1.4),
    # opens (~30) around the block, holds it at ~15 and opens again over the box, so the hold interval of each episode
    # is read off the full-rate gripper signal (mid level 0.5 * (min + max)) after the first
    # opening. Before it: table; during it: held (tool point); after it: box.
    # (hold = from the first frame below the mid level after the first opening to the frame before the next opening)
    fr = F.fullrate(sc)
    hold = {}
    for e in eps:
        g = fr.seq[e][:, 5].astype(float)
        thr = 0.5 * (g.min() + g.max())
        op = np.nonzero(g > thr)[0]
        if not len(op):
            continue
        ta = np.nonzero((g < thr) & (np.arange(len(g)) > op[0]))[0]   # closes on the block after the first opening
        if not len(ta):
            continue
        tb = np.nonzero((g > thr) & (np.arange(len(g)) > ta[0]))[0]   # opens again: the release over the box
        hold[e] = (int(ta[0]), int(tb[0]) - 1 if len(tb) else len(g) - 1)
    tt = np.asarray(fs.labels["t"]).astype(int)
    det = {}
    for i in range(fs.n):
        e = int(ep[i])
        bx = box[str(e)]
        cen = {}
        for c in CAMS:
            m = blk[c][i]
            if m.sum() >= 20:
                yy, xx = np.nonzero(m)
                cen[c] = (float(xx.mean()), float(yy.mean()))
        X = None
        if "top" in cen and "side" in cen:
            o1, d1 = G.cam_ray(P, "top", *cen["top"])
            o2, d2 = G.cam_ray(P, "side", *cen["side"])
            Xt, dist = G.closest(o1, d1, o2, d2)
            if dist < 0.03:
                X = Xt
        if X is None and "top" in cen:
            X = G.hit_z(*G.cam_ray(P, "top", *cen["top"]), B / 2)
        if X is None and "side" in cen:
            X = G.hit_z(*G.cam_ray(P, "side", *cen["side"]), B / 2)
        if X is not None:
            c_, s_ = math.cos(bx[2]), math.sin(bx[2])
            lxy = np.array([[c_, s_], [-s_, c_]]) @ (np.asarray(X[:2]) - np.asarray(bx[:2]))
            inside = abs(lxy[0]) < lx / 2 - 0.01 and abs(lxy[1]) < ly / 2 - 0.01
            det[i] = (np.asarray(X, float), inside)
    out_blk, kinds = {}, {}
    for i in range(fs.n):
        e = int(ep[i])
        bx = box[str(e)]
        h0, h1 = hold.get(e, (10 ** 9, 10 ** 9))
        same = np.nonzero(ep == e)[0]
        if h0 <= tt[i] <= h1:
            out_blk[str(i)] = ["held"]
            kinds[i] = "held"
        elif tt[i] < h0:
            tab = [det[j][0] for j in same if j in det and not det[j][1] and tt[j] < h0]
            X = det[i][0] if i in det and not det[i][1] else (np.median(tab, 0) if tab else None)
            if X is not None:
                out_blk[str(i)] = ["table", [float(X[0]), float(X[1]), B / 2], 0.0]
                kinds[i] = "table"
        else:
            inb = [det[j][0] for j in same if j in det and det[j][1] and tt[j] > h1]
            X = det[i][0] if i in det and det[i][1] else (np.median(inb, 0) if inb else np.array([bx[0], bx[1], 0]))
            out_blk[str(i)] = ["box", [float(X[0]), float(X[1]), 0.002 + B / 2], float(bx[2])]
            kinds[i] = "box"
    cnt = {k: sum(1 for v in kinds.values() if v == k) for k in ("table", "held", "box")}
    cnt["hidden"] = fs.n - sum(cnt.values())
    print("block", cnt, flush=True)
    # layout: wall behind the arm in the top view: first dark->bright transition below row 150 (wall -> table) in
    # columns away from the arm and the box, ray-cast to z = 0, RANSAC line
    bg = np.load(HERE / "out" / "bg_median_top.npy").astype(np.float32).mean(-1)
    bm_all = boxm["top"].mean(0) > 0.2
    pts = []
    for u in list(range(10, 230, 10)) + list(range(360, 400, 10)) + list(range(610, 640, 10)):
        if bm_all[:, u // 4].any():
            continue
        dv = np.diff(bg[:, u])
        cand = np.nonzero(dv[150:420] > 6)[0]
        if len(cand):
            v = int(cand[0]) + 150
            o, d = G.cam_ray(P, "top", u, v + 0.5)
            if d[2] < 0:
                pts.append(G.hit_z(o, d, 0.0)[:2])
    pts = np.array(pts)
    layout = {}
    if len(pts) >= 5:
        rng = np.random.default_rng(0)
        best_in = None
        for _ in range(300):
            i1, i2 = rng.choice(len(pts), 2, replace=False)
            dv_ = pts[i2] - pts[i1]
            if np.linalg.norm(dv_) < 1e-3:
                continue
            nv = np.array([-dv_[1], dv_[0]]) / np.linalg.norm(dv_)
            inl = np.abs((pts - pts[i1]) @ nv) < 0.01
            if best_in is None or inl.sum() > best_in.sum():
                best_in = inl
        q = pts[best_in]
        k, b = np.polyfit(q[:, 0], q[:, 1], 1)   # y = k x + b
        ang = math.atan(k)
        mid = np.array([0.0, b])
        nrm = np.array([-math.sin(ang), math.cos(ang)])
        if (cam_top[:2] - mid) @ nrm < 0:
            nrm = -nrm   # towards the camera
        c0 = mid - 0.006 * nrm + 0.16 * np.array([math.cos(ang), math.sin(ang)])   # spans x ~ -0.48 .. 0.80
        layout["wall"] = dict(pos=[float(c0[0]), float(c0[1]), 0.6], size=[0.64, 0.005, 0.6],
                              quat=list(map(float, F.A.quat_from_rotvec([0, 0, ang]))), material="wall")
        print("wall: y = %.3f x + %.3f (%d of %d edge points)" % (k, b, int(best_in.sum()), len(pts)), flush=True)
    # GRAYBLOCK: the side view's own wall (left of its wall corner): the table-wall edge of the side background median
    # (first dark -> bright transition below row 200, columns 10..250) ray-cast to z = 0, RANSAC line, as above
    bgs = np.load(HERE / "out" / "bg_median_side.npy")
    gs = bgs.astype(np.float32).mean(-1)
    pts2 = []
    for u in range(10, 250, 10):
        dv = np.diff(gs[:, u])
        cand = np.nonzero(dv[200:470] > 6)[0]
        if len(cand):
            v = int(cand[0]) + 200
            o, d = G.cam_ray(P, "side", u, v + 0.5)
            if d[2] < 0:
                pts2.append(G.hit_z(o, d, 0.0)[:2])
    pts2 = np.array(pts2)
    if len(pts2) >= 5:
        k2, b2 = np.polyfit(pts2[:, 0], pts2[:, 1], 1)
        res_ = np.abs(pts2[:, 1] - (k2 * pts2[:, 0] + b2))
        ok = res_ < 0.02
        k2, b2 = np.polyfit(pts2[ok, 0], pts2[ok, 1], 1)
        ang2 = math.atan(k2)
        cam_side = np.asarray(P["cams"]["side"]["groups"]["all"]["pos"], float)
        x0 = float(np.median(pts2[ok, 0]))
        mid2 = np.array([x0, k2 * x0 + b2])
        nrm2 = np.array([-math.sin(ang2), math.cos(ang2)])
        if (cam_side[:2] - mid2) @ nrm2 < 0:
            nrm2 = -nrm2
        c2 = mid2 - 0.006 * nrm2
        dirv2 = np.array([math.cos(ang2), math.sin(ang2)])
        # extend the wall from its far end (beyond the side camera's view) to the back wall's line
        # not added: within 1-2 cm of the back wall's line (the side view sees the same wall; its vertical seam is no
        # corner), so only the back wall is modelled
        print("side-view wall edge (check only): y = %.3f x + %.3f (%d of %d edge points)" % (k2, b2, int(ok.sum()), len(pts2)), flush=True)
    # GRAYBLOCK: the black chair seen by the side camera only: a dark slab facing the side camera, spanning the rays of
    # the chair's bounding box (dark pixels of the side background median right of x = 400) at 1.0 m from the camera
    darkc = bgs.max(-1) < 60
    darkc[:, :400] = False
    ys_, xs_ = np.nonzero(darkc)
    if len(xs_) > 1000:
        gq = P["cams"]["side"]["groups"]["all"]
        Rs = G.mat_from_quat(gq["quat"])
        fwd = -Rs[:, 2]
        for D in np.arange(1.0, 2.01, 0.05):   # nearest distance at which the slab is outside the top view
            corners = []
            for u, v in ((xs_.min(), ys_.min()), (xs_.max() + 40, ys_.min()), (xs_.min(), ys_.max()),
                         (xs_.max() + 40, ys_.max())):
                o, d = G.cam_ray(P, "side", float(u), float(v))
                corners.append(o + d * (D / float(d @ fwd)))
            corners = np.array(corners)
            uv, zz = project(P, "top", corners)
            if ((uv[:, 0] > 660) | (uv[:, 0] < -20) | (zz < 0)).all() or (uv[:, 0] > 660).all():
                break
        cen = corners.mean(0)
        wv = np.linalg.norm(corners[1] - corners[0]) / 2
        hv = np.linalg.norm(corners[2] - corners[0]) / 2
        layout["chair"] = dict(pos=list(map(float, cen)), size=[float(wv), float(hv), 0.01],
                               quat=list(map(float, F.A.quat_from_mat(Rs))), material="chair")
        print("chair slab at", np.round(cen, 3).tolist(), "half size", round(wv, 3), round(hv, 3), "D", round(float(D), 2), flush=True)
    res = dict(box_size=dict(lx=lx, ly=ly, h=h, t=0.003), box=box, box_iou=dict(median=float(np.median(ious)),
               min=float(np.min(ious))), block_size=B, block=out_blk, block_counts=cnt, layout=layout,
               fit_json=a.fit)
    (HERE / "out" / "objects.json").write_text(json.dumps(res, indent=1))
    print("wrote out/objects.json")


if __name__ == "__main__":
    main()
