"""Static white box from the real frames (both cameras): fit a box (centre x, y, yaw, half sizes, height; standing on
z = 0) to the box pixels of the median background images, through the fitted cameras + lens.   RGB render only (the
IoU metric is arm-only).

    .venv/bin/python so101_twin/pos5/objects.py --fit out/pos5_r1_fit.json   -> out/objects.json
"""
from __future__ import annotations

import argparse
import json
import math

import mujoco
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

import arm as A
import fit as F
import p5

SEED = {"top": (225, 403), "side": (220, 385)}   # (row, col) inside the box in each camera's median image


def box_masks():
    out = {}
    for c in p5.CAMS:
        st = p5.load(c, "_static")[0]
        lab, _ = ndi.label(st)
        out[c] = ndi.binary_fill_holes(lab == lab[SEED[c]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="out/pos5_r1_fit.json")
    ap.add_argument("--start", default="", help="previous objects json to start from as well")
    a = ap.parse_args()
    P = p5.load_params(p5.HERE / a.fit)
    sc = p5.scene()
    P = F.reconcile(sc, P)
    tgt = box_masks()
    cams = [A.CamSpec(c, pos=tuple(P["cams"][c]["groups"]["all"]["pos"]), quat=tuple(P["cams"][c]["groups"]["all"]["quat"]),
                      fovy=P["cams"][c]["fovy"]) for c in p5.CAMS]

    def extras(root, asset, wb):
        b = ET.SubElement(wb, "body", name="box", pos="0.2 0.2 0.05", mocap="true")
        ET.SubElement(b, "geom", name="box", type="box", size="0.05 0.05 0.05", rgba="1 1 1 1")

    import xml.etree.ElementTree as ET

    xml = A.build_mjcf(sc.ARM, cams, extras=lambda r, s, w: extras(r, s, w))
    arm = A.Arm(xml, "new", "so101")
    m, d = arm.m, arm.d
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
    lens = {c: A.Lens(**P["cams"][c]["lens"]) for c in p5.CAMS}
    arm.d.qpos[:] = 0
    arm.d.qpos[arm.qadr[1]] = -1.7   # fold the arm away from the box (it is not drawn in the box mask anyway)

    def render(x):
        cx, cy, yaw, sx, sy, h, z0 = x
        d.mocap_pos[m.body_mocapid[bid]] = (cx, cy, z0 + abs(h) / 2)
        d.mocap_quat[m.body_mocapid[bid]] = A.quat_from_rotvec([0, 0, yaw])
        m.geom_size[gid] = (abs(sx), abs(sy), abs(h) / 2)
        m.geom_rbound[gid] = float(np.linalg.norm(m.geom_size[gid]))
        mujoco.mj_forward(m, d)
        out = {}
        for c in p5.CAMS:
            seg = lens[c].render(arm, c, 320, 240, seg=True)
            out[c] = (seg[..., 0] == gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        return out

    small = {c: tgt[c].reshape(240, 2, 320, 2).mean((1, 3)) >= 0.5 for c in p5.CAMS}

    def loss(x):
        r = render(x)
        return 1 - np.mean([(r[c] & small[c]).sum() / max((r[c] | small[c]).sum(), 1) for c in p5.CAMS])

    best = None
    rng = np.random.default_rng(0)
    for k in range(400):   # coarse random search
        x = np.array([rng.uniform(-0.3, 0.3), rng.uniform(0.0, 0.5), rng.uniform(-0.8, 0.8), rng.uniform(0.03, 0.12),
                      rng.uniform(0.03, 0.12), rng.uniform(0.03, 0.2), rng.uniform(-0.06, 0.03)])
        L = loss(x)
        if best is None or L < best[0]:
            best = (L, x)
    x = best[1]
    if a.start:   # also start from a previous box (objects.json of another camera fit) and keep the better
        b = json.loads((p5.HERE / a.start).read_text())["box"]
        x0 = np.array([*b["center"], b["yaw"], *b["half"], b["height"], b["z0"]])
        if loss(x0) < loss(x):
            x = x0
    for _ in range(3):
        r = minimize(loss, x, method="Nelder-Mead", options=dict(maxfev=800, xatol=1e-5, fatol=1e-6))
        x = r.x
    rr = render(x)
    per = {c: round(float((rr[c] & small[c]).sum() / max((rr[c] | small[c]).sum(), 1)), 3) for c in p5.CAMS}
    print("box IoU", round(1 - loss(x), 3), per, "params", np.round(x, 4).tolist())
    out = dict(box=dict(center=[float(x[0]), float(x[1])], yaw=float(x[2]), half=[abs(float(x[3])), abs(float(x[4]))],
                        height=abs(float(x[5])), z0=float(x[6])), iou=per, fit=a.fit)
    (p5.OUT / "objects.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()


CUBE_HALF = 0.011   # ~22 mm cube (top view: ~20 px at ~1 mm/px)


def backproject(P: dict, cam: str, u: float, v: float, z: float):
    """real pixel (u, v) of camera `cam` (fitted pose + lens) -> world point on the plane z."""
    c = P["cams"][cam]
    g = c["groups"]["all"]
    L = c["lens"]
    w, h = 640, 480
    f = 0.5 * h / math.tan(math.radians(c["fovy"]) / 2)
    xd, yd = (u - w / 2 - L["cx"] * w) / f, (v - h / 2 - L["cy"] * h) / f
    xu, yu = xd, yd
    for _ in range(20):
        r2 = xu * xu + yu * yu
        s = 1 + L["k1"] * r2 + L["k2"] * r2 * r2 + L.get("k3", 0.0) * r2 ** 3
        xu, yu = xd / s, yd / s
    R = A.mat_from_quat(np.asarray(g["quat"]) / np.linalg.norm(g["quat"]))
    ray = R @ np.array([xu, -yu, -1.0])
    p0 = np.asarray(g["pos"], float)
    t = (z - p0[2]) / ray[2]
    return p0 + t * ray


def cube_track(P: dict, z0: float) -> list:
    """per frame: world (x, y, z) of the cube on the floor from the top camera's cube pixels, or None."""
    cube = p5.load("top", "_cube")
    out = []
    for i in range(len(cube)):
        ys, xs = np.nonzero(cube[i])
        if len(xs) < 40:
            out.append(None)
            continue
        q = backproject(P, "top", float(xs.mean()) + 0.5, float(ys.mean()) + 0.5, z0 + 2 * CUBE_HALF)
        out.append([float(q[0]), float(q[1]), float(z0 + CUBE_HALF)])
    return out


def fill_track(track: list) -> list:
    """frames without a visible cube in the approach phase take the nearest visible approach-phase position of the
    same episode (the gripper hides it); release / unmatched frames stay None (frame_setup: in the box)."""
    fs = F.real_set(p5.scene())
    ep, ph = np.asarray(fs.labels["episode"]), np.asarray(fs.labels["phase"])
    out = list(track)
    for i in range(len(track)):
        if track[i] is None and ph[i] == "approach":
            cand = [j for j in range(len(track)) if ep[j] == ep[i] and ph[j] == "approach" and track[j] is not None]
            if cand:
                out[i] = track[min(cand, key=lambda j: abs(j - i))]
    return out


def add_cube_track(fit_json: str):
    P = F.reconcile(p5.scene(), p5.load_params(p5.HERE / fit_json))
    o = json.loads((p5.OUT / "objects.json").read_text())
    o["cube_half"] = CUBE_HALF
    o["cube_xyz"] = fill_track(cube_track(P, o["box"]["z0"]))
    o["cube_fit"] = fit_json
    (p5.OUT / "objects.json").write_text(json.dumps(o, indent=1))
    n = sum(x is not None for x in o["cube_xyz"])
    print("cube positions from the top view:", n, "frames")
