"""Cup / ball placement for the ballcup twin from the DINO+SAM object masks of both cameras (no arm fitting here).

    .venv/bin/python so101_twin/ballcup/objects.py --fit out/<run>_fit.json [--obj-tag _d1]   # -> out/objects.json

Cup = frustum mesh (r_top, r_bot, h) standing on the table (z = 0). Shape: grid of candidates scored by the mean IoU of
the twin cup silhouette (arm posed in front of it, lens warp) with the SAM cup masks over 10 episodes; per-episode xy:
initialised by casting the top-view cup-mask centroid through the fitted top camera onto the plane z = 0.75 h, then
refined per episode by Nelder-Mead on the cup IoU of both views (the cup does not move within an episode).
Ball = sphere of radius R. Per frame: the two views' ball-mask centroids are triangulated (closest point of the two
rays) when both views see a ball blob and the rays pass within 4 cm; otherwise top view only, cast onto z = R (on
the table) or, if that lands within 6 cm of the gripper's tool point, the tool point (held); else no ball (hidden).
Episodes and frames of BOTH halves are placed (this is scene construction from the frames; nothing here is fitted
to the arm silhouettes).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import mujoco
import numpy as np
from scipy.optimize import minimize

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A  # noqa: E402
import fit as F  # noqa: E402

N_SEG = 32


def frustum_verts(r_top, r_bot, h, n=N_SEG):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    top = np.stack([r_top * np.cos(a), r_top * np.sin(a), np.full(n, h)], 1)
    bot = np.stack([r_bot * np.cos(a), r_bot * np.sin(a), np.zeros(n)], 1)
    return np.concatenate([bot, top], 0)


def cup_extras(shape):
    import xml.etree.ElementTree as ET

    def ex(root, asset, wb):
        v = frustum_verts(*shape)
        ET.SubElement(asset, "mesh", name="cup_mesh", vertex=" ".join(f"{x:.5f}" for x in v.ravel()))
        ET.SubElement(asset, "material", name="cupmat", rgba="0.9 0.85 0.75 1")
        b = ET.SubElement(wb, "body", name="cup", pos="0 0.3 0", mocap="true")
        ET.SubElement(b, "geom", name="cup", type="mesh", mesh="cup_mesh", material="cupmat", contype="0",
                      conaffinity="0", group="2")
    return ex


class CupRig:
    def __init__(self, sc, P, shape):
        import dataclasses

        sc2 = dataclasses.replace(sc.ARM)
        cams = [A.CamSpec(c, mount="world", pos=tuple(p["groups"]["all"]["pos"]), quat=tuple(p["groups"]["all"]["quat"]),
                          fovy=p["fovy"]) for c, p in P["cams"].items()]
        xml = A.build_mjcf(sc2, cams, extras=cup_extras(shape), offwidth=2048, offheight=1600)
        self.arm = A.Arm(xml, sc.ARM.calib, sc.ARM.model)
        self.arm.offsets_deg = np.asarray(P["offsets_deg"], float)
        self.arm.scales = np.asarray(P.get("scales", [1.0] * 5), float)
        self.arm.grip_scale = float(P.get("grip_scale", 1.0))
        self.lens = {c: A.Lens(**p["lens"]) for c, p in P["cams"].items()}
        self.cup_gid = mujoco.mj_name2id(self.arm.m, mujoco.mjtObj.mjOBJ_GEOM, "cup")
        bid = mujoco.mj_name2id(self.arm.m, mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.mocap = self.arm.m.body_mocapid[bid]

    def cup_mask(self, cam, state, xy, w=320, h=240):
        self.arm.set_state(state)
        self.arm.d.mocap_pos[self.mocap] = [xy[0], xy[1], 0.0]
        mujoco.mj_forward(self.arm.m, self.arm.d)
        seg = self.lens[cam].render(self.arm, cam, w, h, seg=True)
        return (seg[..., 0] == self.cup_gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))


def cam_ray(P, cam, u, v, w=640, h=480):
    """world ray (origin, unit dir) through pixel (u, v) (pixel-centre convention of arm.Lens), lens undistorted."""
    p = P["cams"][cam]
    g = p["groups"]["all"]
    L = p["lens"]
    R = A.mat_from_quat(np.asarray(g["quat"]) / np.linalg.norm(g["quat"]))
    f = 0.5 * h / math.tan(math.radians(p["fovy"]) / 2)
    xd = (u + 0.5 - w / 2 - L["cx"] * w) / f
    yd = (v + 0.5 - h / 2 - L["cy"] * h) / f
    xu, yu = xd, yd
    for _ in range(20):
        r2 = xu * xu + yu * yu
        s = 1 + L["k1"] * r2 + L["k2"] * r2 * r2
        xu, yu = xd / s, yd / s
    d = R @ np.array([xu, -yu, -1.0])
    return np.asarray(g["pos"], float), d / np.linalg.norm(d)


def hit_z(o, d, z):
    t = (z - o[2]) / d[2]
    return o + t * d


def closest(o1, d1, o2, d2):
    w = o1 - o2
    a, b, c, dd, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w, d2 @ w
    den = a * c - b * b
    s = (b * e - c * dd) / den
    t = (a * e - b * dd) / den
    p1, p2 = o1 + s * d1, o2 + t * d2
    return 0.5 * (p1 + p2), float(np.linalg.norm(p1 - p2))


def centroid(m):
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--obj-tag", default="_d1")
    ap.add_argument("--ball-r", type=float, default=0.0)
    ap.add_argument("--max-eps", type=int, default=0, help="debug: place only the first N episodes")
    ap.add_argument("--out", default="objects.json")
    a = ap.parse_args()
    sc = F.load_scene("ballcup")
    fj = json.loads(pathlib.Path(a.fit).read_text())
    P = F.reconcile(sc, fj["params"])
    fs = F.real_set(sc)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    eps = sorted(set(ep))
    if a.max_eps:
        eps = eps[:a.max_eps]
    om = {(c, o): F.unpack(np.load(HERE / "masks" / f"ballcup_{c}{a.obj_tag}_{o}.npy")) for c in ("top", "side")
          for o in ("cup", "ball")}
    area = {k: v.reshape(fs.n, -1).sum(1) for k, v in om.items()}
    t0 = time.time()
    # ---- cup: frames with a clear cup blob in both views (cup area within 30 % of the episode median)
    good = {}
    for e in eps:
        idx = np.nonzero(ep == e)[0]
        ok = np.ones(len(idx), bool)
        for c in ("top", "side"):
            ar = area[(c, "cup")][idx]
            med = np.median(ar[ar > 0]) if (ar > 0).any() else 0
            ok &= (ar > 0.7 * med) & (ar < 1.3 * med) & (ar > 300)
        good[e] = idx[ok][:4]   # at most 4 frames per episode (speed)
    small = {c: F.down2(om[(c, "cup")]) for c in ("top", "side")}

    def init_xy(e, h):
        pts = []
        for i in good[e]:
            cu = centroid(om[("top", "cup")][i])
            if cu is None:
                continue
            o, d = cam_ray(P, "top", *cu)
            pts.append(hit_z(o, d, 0.75 * h)[:2])
        return np.median(pts, 0) if pts else np.array([0.0, 0.3])

    def ep_score(rig, e, xy, cams=("top", "side")):
        v = []
        for i in good[e]:
            for c in cams:
                m = rig.cup_mask(c, fs.state[i], xy)
                r = small[c][i]
                u = (m | r).sum()
                v.append((m & r).sum() / u if u else 0.0)
        return float(np.mean(v)) if v else 0.0

    sel_eps = [e for e in eps if len(good[e]) >= 3][::4][:10]
    best = None
    for h in ((0.085, 0.095, 0.105, 0.115) if not a.max_eps else (0.1,)):
        for rt in ((0.038, 0.042, 0.046, 0.050) if not a.max_eps else (0.045,)):
            for taper in (0.70, 0.78):
                shape = (rt, rt * taper, h)
                rig = CupRig(sc, P, shape)
                s = float(np.mean([ep_score(rig, e, init_xy(e, h)) for e in sel_eps]))
                rig.arm.close()
                if best is None or s > best[0]:
                    best = (s, shape)
    print(f"cup shape grid: best {best[0]:.3f} r_top {best[1][0]:.3f} r_bot {best[1][1]:.3f} h {best[1][2]:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    shape = best[1]
    rig = CupRig(sc, P, shape)
    cups = {}
    for e in eps:
        x0 = init_xy(e, shape[2])
        if len(good[e]) == 0:
            cups[e] = dict(xy=[float(x0[0]), float(x0[1])], iou=None, n=0)
            continue
        res = minimize(lambda x: -ep_score(rig, e, x0 + 0.01 * x), np.zeros(2), method="Nelder-Mead",
                       options=dict(xatol=0.05, fatol=1e-3, maxfev=25, initial_simplex=[[0, 0], [1, 0], [0, 1]]))
        xy = x0 + 0.01 * res.x
        cups[e] = dict(xy=[float(xy[0]), float(xy[1])], iou=round(-float(res.fun), 3), n=int(len(good[e])),
                       moved_cm=round(float(np.linalg.norm(0.01 * res.x)) * 100, 2))
    print(f"cups placed ({time.time() - t0:.0f}s): median IoU "
          f"{np.median([v['iou'] for v in cups.values() if v['iou'] is not None]):.3f}", flush=True)
    # ---- ball radius: top-view blob on the table (approach frames: ball far from the gripper)
    R = a.ball_r or 0.033
    # ---- ball per frame
    fit_tcp = mujoco.mj_name2id(rig.arm.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    balls = {}
    for i in range(fs.n):
        rig.arm.set_state(fs.state[i])
        tcp = rig.arm.d.site_xpos[fit_tcp].copy()
        ct = centroid(om[("top", "ball")][i]) if area[("top", "ball")][i] > 80 else None
        cs = centroid(om[("side", "ball")][i]) if area[("side", "ball")][i] > 80 else None
        pos, src = None, "none"
        if ct and cs:
            o1, d1 = cam_ray(P, "top", *ct)
            o2, d2 = cam_ray(P, "side", *cs)
            p, dist = closest(o1, d1, o2, d2)
            if dist < 0.04 and -0.02 < p[2] < 0.5:
                pos, src = p, "triangulated"
        if pos is None and ct:
            o, d = cam_ray(P, "top", *ct)
            p = hit_z(o, d, R)
            if np.linalg.norm(p[:2] - tcp[:2]) < 0.06 and tcp[2] > 2 * R:
                pos, src = tcp, "tcp"
            else:
                pos, src = p, "top_on_table"
        if pos is not None and pos[2] < R:
            pos = np.array([pos[0], pos[1], R])
        balls[i] = dict(pos=None if pos is None else [round(float(x), 4) for x in pos], src=src)
    srcs = [v["src"] for v in balls.values()]
    print("ball sources", {s: srcs.count(s) for s in set(srcs)}, flush=True)
    out = dict(fit=a.fit, obj_tag=a.obj_tag, cup=dict(r_top=shape[0], r_bot=shape[1], h=shape[2], shape_iou=best[0]),
               ball_r=R, episodes={str(e): v for e, v in cups.items()}, frames={str(i): v for i, v in balls.items()})
    (HERE / "out" / a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote out/{a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
