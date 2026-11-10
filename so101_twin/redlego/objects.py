"""REDLEGO: object placement for the RGB render (never used by the metric) from the twin-independent DINO+SAM object
masks of the d1 run (masks/redlego_<cam>_d1_box.npy, _d1_lego.npy), through the FITTED cameras.

    .venv/bin/python so101_twin/redlego/objects.py --fit out/<run>_fit.json   # -> out/objects.json

Box (open cardboard box, static within an episode): outer size lx, ly, wall height h and pose (x, y, yaw); the shape
is a Nelder-Mead fit of the rendered box silhouette to the box masks (both views, the first frame of 8 episodes where
the arm is at rest, i.e. not over the box), then (x, y, yaw) per episode with the shape fixed.
Brick (per frame): triangulated from the two views' brick-mask centroids when both see it (rays within 4 cm), else the
top-view centroid ray at z = brick height / 2 (table) or at the box floor (inside the box footprint); 'held' (at the
tool point, aligned with the gripper) when it is within 6 cm of the tool point while the gripper is partly closed
(10..35 of 100) and above the table; 'box' when it lies inside the box footprint; otherwise 'table' with the yaw of
the top-view min-area box; hidden when no view sees it.
Layout (RGB backdrop): the table-wall edges read off the background medians (out/bg_global_<cam>.npy) at fixed pixels,
ray-cast to z = 0 through the fitted cameras.
"""
import argparse
import dataclasses
import json
import math
import pathlib
import sys

import mujoco
import numpy as np
from scipy import optimize

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
from geom import cam_ray, closest, hit_z  # noqa: E402

W, H = 160, 120
CAMS = ("top", "side")


def load(tag):
    return np.unpackbits(np.load(HERE / "masks" / f"redlego_{tag}.npy"), axis=-1)[..., :640].astype(bool)


def small(m):
    return m.reshape(-1, H, 4, W, 4).mean((2, 4)) >= 0.5


def min_box_yaw(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return 0.0
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1)
    best = (1e18, 0.0)
    for a in np.deg2rad(np.arange(0, 180, 3)):
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        q = pts @ R
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if area < best[0]:
            best = (area, a)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--layout", default="", help="json {cam: [[u, v], ...]} table-edge pixels for the walls")
    a = ap.parse_args()
    sc = F.load_scene("redlego")
    fj = json.loads(pathlib.Path(a.fit).read_text())
    sc.ARM = dataclasses.replace(sc.ARM, model=fj.get("model", sc.ARM.model))
    P = F.reconcile(sc, fj["params"])
    sc.OBJ = None
    rig = F.Rig(sc, P, full=True)
    m, d = rig.arm.m, rig.arm.d
    fs = F.real_set(sc)
    ep = np.asarray(fs.labels["episode"]).astype(int)
    t = np.asarray(fs.labels["t"]).astype(int)
    names = ["box_floor", "box_wx0", "box_wx1", "box_wy0", "box_wy1"]
    gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in names]
    lut = np.zeros(m.ngeom + 1, bool)
    lut[np.asarray(gid) + 1] = True
    bb = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")]
    lb = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lego")]
    d.mocap_pos[lb] = (0, 0, -2)
    real = {c: small(load(f"{c}_d1_box")) for c in CAMS}
    lens = {c: F.A.Lens(**P["cams"][c]["lens"]) for c in CAMS}
    for c in CAMS:
        g = P["cams"][c]["groups"]["all"]
        cid = rig.arm.cam_id[c]
        m.cam_pos[cid] = g["pos"]
        m.cam_quat[cid] = np.asarray(g["quat"]) / np.linalg.norm(g["quat"])
        m.cam_fovy[cid] = P["cams"][c]["fovy"]
    m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_root")] = (0, 0, -5)   # arm out of the way
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
    m.geom_group[tb] = 4   # table not rendered (group 4 hidden)
    T = sc.BOX["t"]

    def set_shape(lx, ly, h):
        lx, ly = lx / 2, ly / 2
        m.geom_size[gid[0]] = [lx, ly, T / 2]
        m.geom_pos[gid[0]] = [0, 0, T / 2]
        for k, (sz, ps) in enumerate((((T / 2, ly, h / 2), (-lx + T / 2, 0, h / 2)),
                                      ((T / 2, ly, h / 2), (lx - T / 2, 0, h / 2)),
                                      ((lx, T / 2, h / 2), (0, -ly + T / 2, h / 2)),
                                      ((lx, T / 2, h / 2), (0, ly - T / 2, h / 2)))):
            m.geom_size[gid[1 + k]] = sz
            m.geom_pos[gid[1 + k]] = ps

    def box_mask(c):
        mujoco.mj_kinematics(m, d)
        seg = lens[c].render(rig.arm, c, W, H, seg=True)
        return lut[seg[..., 0] + 1] & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))

    def rest_frames(e):
        idx = np.nonzero(ep == e)[0]
        return idx[np.argsort(t[idx])][:1]

    def loss(x, frames, shape=None):
        lx, ly, h = shape if shape is not None else x[3:6]
        set_shape(lx, ly, h)
        d.mocap_pos[bb] = (x[0], x[1], 0.0)
        d.mocap_quat[bb] = F.A.quat_from_rotvec([0, 0, x[2]])
        v = []
        for c in CAMS:
            sim = box_mask(c)
            for i in frames:
                r = real[c][i]
                u = (sim | r).sum()
                if r.sum() > 20:
                    v.append((sim & r).sum() / u if u else 0.0)
        return -float(np.mean(v)) if v else 0.0

    # init: top-view box centroid ray at z = h/2, yaw from the top-view min-area box (rough)
    eps = sorted(set(ep.tolist()))
    i0 = rest_frames(eps[0])[0]
    ys, xs = np.nonzero(real["top"][i0])
    p0 = hit_z(*cam_ray(P, "top", xs.mean() * 4 + 1.5, ys.mean() * 4 + 1.5), 0.035)
    best = None
    for yaw0 in (0.0, 0.5, -0.5, 1.2, -1.2):   # a few yaw starts (the box edges are not axis aligned in either view)
        x0 = np.array([p0[0], p0[1], yaw0, 0.30, 0.20, 0.07])
        fr = np.concatenate([rest_frames(e) for e in eps[::5]])
        r = optimize.minimize(loss, x0, args=(fr,), method="Nelder-Mead",
                              options=dict(xatol=1e-3, fatol=1e-4, maxfev=400, initial_simplex=np.array(
                                  [x0, x0 + [0.04, 0, 0, 0, 0, 0], x0 + [0, 0.04, 0, 0, 0, 0], x0 + [0, 0, 0.3, 0, 0, 0],
                                   x0 + [0, 0, 0, 0.05, 0, 0], x0 + [0, 0, 0, 0, 0.05, 0], x0 + [0, 0, 0, 0, 0, 0.03]])))
        print(f"box yaw0 {yaw0:+.1f}: IoU {-r.fun:.3f} x {np.round(r.x, 3).tolist()}", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    shape = [float(v) for v in best.x[3:6]]
    print(f"box shape lx {shape[0]:.3f} ly {shape[1]:.3f} h {shape[2]:.3f} IoU {-best.fun:.3f}", flush=True)
    box, box_iou = {}, {}
    for e in eps:
        fr = rest_frames(e)
        x0 = best.x[:3].copy()
        r = optimize.minimize(loss, x0, args=(fr, shape), method="Nelder-Mead",
                              options=dict(xatol=5e-4, fatol=1e-4, maxfev=120, initial_simplex=np.array(
                                  [x0, x0 + [0.02, 0, 0], x0 + [0, 0.02, 0], x0 + [0, 0, 0.1]])))
        box[str(e)] = [round(float(r.x[0]), 4), round(float(r.x[1]), 4), round(float(r.x[2]), 4)]
        box_iou[str(e)] = round(-float(r.fun), 3)
    print("box IoU per episode median", np.median(list(box_iou.values())), flush=True)
    # ---- brick
    L = sc.LEGO
    lego = {c: load(f"{c}_d1_lego") for c in CAMS}
    g = np.asarray(fs.state[:, 5], float)
    rig2 = F.Rig(sc, P)
    sid = mujoco.mj_name2id(rig2.arm.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    def in_box(p, e):
        bx = box[str(e)]
        c, s_ = math.cos(-bx[2]), math.sin(-bx[2])
        q = np.array([c * (p[0] - bx[0]) - s_ * (p[1] - bx[1]), s_ * (p[0] - bx[0]) + c * (p[1] - bx[1])])
        return abs(q[0]) < shape[0] / 2 - 0.01 and abs(q[1]) < shape[1] / 2 - 0.01

    out = {}
    held_off = []
    for i in range(fs.n):
        e = int(ep[i])
        rig2.arm.set_state(F.frame_state(sc, P, "top", i))
        tcp = rig2.arm.d.site_xpos[sid].copy()
        Rt = rig2.arm.d.site_xmat[sid].reshape(3, 3).copy()
        rays = {}
        for c in CAMS:
            mm = lego[c][i]
            if 40 < mm.sum() < 0.08 * 640 * 480:
                ys, xs = np.nonzero(mm)
                rays[c] = cam_ray(P, c, xs.mean(), ys.mean())
        if not rays:
            out[str(i)] = None
            continue
        p3 = None
        if len(rays) == 2:
            pt, gap = closest(*rays["top"], *rays["side"])
            if gap < 0.04:
                p3 = pt
        if p3 is not None and np.linalg.norm(p3 - tcp) < 0.06 and 10 <= g[i] <= 35 and p3[2] > L["h"] * 0.8:
            out[str(i)] = ["held"]
            held_off.append(Rt.T @ (p3 - tcp))
            continue
        if "top" in rays:
            pt = hit_z(*rays["top"], L["h"] / 2)
            if in_box(pt, e):
                pt = hit_z(*rays["top"], T + L["h"] / 2)
                out[str(i)] = ["box", [round(float(v), 4) for v in pt], float(min_box_yaw(lego["top"][i]))]
            elif p3 is not None and p3[2] > 0.05 and np.linalg.norm(p3 - tcp) < 0.08:
                out[str(i)] = ["held"]
            else:
                out[str(i)] = ["table", [round(float(v), 4) for v in pt], float(min_box_yaw(lego["top"][i]))]
        elif p3 is not None:
            out[str(i)] = ["table", [round(float(v), 4) for v in p3], 0.0]
        else:
            out[str(i)] = None
    ho = np.median(np.array(held_off), 0) if held_off else np.zeros(3)
    print("held offset (gripper frame)", np.round(ho, 4).tolist(), "from", len(held_off), "frames", flush=True)
    # ---- layout (RGB only): back wall from two top-view pixels on the table-wall edge; the room corner from one
    # top-view pixel (a wall perpendicular to the back wall through it); the table's right edge from two top-view pixels
    layout = {}
    if a.layout:
        lay = json.loads(pathlib.Path(a.layout).read_text())
        p1 = hit_z(*cam_ray(P, "top", *lay["back"][0]), 0.0)
        p2 = hit_z(*cam_ray(P, "top", *lay["back"][1]), 0.0)
        dv = (p2 - p1) / np.linalg.norm(p2 - p1)
        nrm = np.array([-dv[1], dv[0], 0.0])
        if nrm @ (np.zeros(3) - p1) > 0:
            nrm = -nrm   # pointing away from the arm (into the wall)
        yaw = math.atan2(dv[1], dv[0])
        mid = 0.5 * (p1 + p2)
        layout["back"] = dict(pos=list(map(float, mid + 0.005 * nrm + np.array([0, 0, 0.8]))),
                              quat=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], size=[3.0, 0.005, 0.8])
        pc = hit_z(*cam_ray(P, "top", *lay["corner"]), 0.0)
        yaw2 = yaw + math.pi / 2
        layout["side"] = dict(pos=list(map(float, pc + 0.005 * dv * np.sign(dv @ (pc - mid)) + np.array([0, 0, 0.8])
                                           + 3.0 * np.array([math.cos(yaw2), math.sin(yaw2), 0.0]) * np.sign(
                                               np.array([math.cos(yaw2), math.sin(yaw2), 0.0]) @ (-nrm)))),
                              quat=[math.cos(yaw2 / 2), 0.0, 0.0, math.sin(yaw2 / 2)], size=[3.0, 0.005, 0.8])
        e1 = hit_z(*cam_ray(P, "top", *lay["table_edge"][0]), 0.0)
        e2 = hit_z(*cam_ray(P, "top", *lay["table_edge"][1]), 0.0)
        layout["table_edge"] = [list(map(float, e1)), list(map(float, e2))]
        print(f"back wall p1 {np.round(p1, 3).tolist()} p2 {np.round(p2, 3).tolist()}; corner {np.round(pc, 3).tolist()};"
              f" table edge {np.round(e1, 3).tolist()} -> {np.round(e2, 3).tolist()}", flush=True)
    res = dict(fit=a.fit, box_shape=dict(lx=round(shape[0], 4), ly=round(shape[1], 4), h=round(shape[2], 4)),
               box=box, box_iou=box_iou, box_default=box[str(eps[0])], lego=out,
               held_offset=[round(float(v), 4) for v in ho], layout=layout)
    (HERE / "out" / "objects.json").write_text(json.dumps(res, indent=1))
    print("brick states:", {k: sum(1 for v in out.values() if (v is None and k == "none") or
                                   (v is not None and v[0] == k)) for k in ("table", "held", "box", "none")})


if __name__ == "__main__":
    main()
