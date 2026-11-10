"""Fit the "top" and "side" camera poses (and refine the joint offsets) by arm-silhouette IoU against real frames.

    .venv/bin/python so101_twin/pickplace/fit_cameras.py [--frames 18] [--rounds 2] [--no-joints]

Real silhouette: darkest large blob (V = max(RGB) < 70) with small holes filled, see segment().
Sim silhouette: segmentation render of the arm's dark geoms (all arm meshes except the white fixed jaw).
Search: random coarse search over camera pos / lookat / roll / fovy, then scipy Nelder-Mead; optionally
alternate with a Nelder-Mead over the joint OFFSETS_DEG (pan, lift, elbow, wrist_flex, wrist_roll) using
both cameras. Writes cameras.json, joints.json (read by build_scene.py / joints.py), fit_report.json and
overlay sheets out/fit_<cam>.jpg (real frame, sim silhouette edge in green, real-only in red).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_scene  # noqa: E402
import joints  # noqa: E402
from twin import CAMS, Twin, real_frames  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# coarse-search box per camera: (lo, hi) for pos(3), lookat(3), roll(1), fovy(1)
SEARCH = {
    "top": (np.array([-0.15, -1.00, 0.20, -0.10, -0.30, 0.00, -6, 35]),
            np.array([0.15, -0.40, 0.70, 0.10, 0.10, 0.18, 6, 70])),
    "side": (np.array([-0.70, -0.40, 0.08, -0.10, -0.25, 0.00, -6, 40]),
             np.array([-0.25, 0.20, 0.40, 0.12, 0.12, 0.22, 6, 80])),
}


def segment(img: np.ndarray, thr: int = 70, min_hole: int = 2000) -> np.ndarray:
    v = img.max(-1)
    dark = v < thr
    lab, n = ndimage.label(dark)
    if n == 0:
        return dark
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= 0.15 * sizes.max()]
    m = np.isin(lab, keep)
    holes, nh = ndimage.label(~m)
    hs = ndimage.sum(~m, holes, range(1, nh + 1))
    m |= np.isin(holes, [i + 1 for i, s in enumerate(hs) if s < min_hole])
    return m


def shrink(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.BOX)) >= 128


def pick_frames(states: np.ndarray, n: int, rng) -> list[int]:
    """Farthest-point sampling on the normalised joint vector; frame 0 (rest) always first."""
    x = (states[:, :5] - states[:, :5].mean(0)) / (states[:, :5].std(0) + 1e-6)
    chosen = [0]
    d = np.linalg.norm(x - x[0], axis=1)
    while len(chosen) < n:
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, np.linalg.norm(x - x[j], axis=1))
    return chosen


def unpack(p):
    return dict(pos=list(map(float, p[:3])), lookat=list(map(float, p[3:6])), roll=float(p[6]), fovy=float(p[7]))


class Fitter:
    def __init__(self, fs, idx, w, h):
        self.fs, self.idx, self.w, self.h = fs, idx, w, h
        self.tw = Twin(w=w, h=h)
        self.states = np.stack([fs.state[i] for i in idx])
        self.real = {c: np.stack([shrink(segment(fs.obs(i)["images"][c]), w, h) for i in idx]) for c in CAMS}
        self.joint_kw = dict(signs=joints.SIGNS.copy(), offsets_deg=joints.OFFSETS_DEG.copy())

    def iou(self, cam, p=None, frames=None, joint_kw=None):
        if p is not None:
            self.tw.set_camera(cam, p[:3], p[3:6], p[6], p[7])
        sel = range(len(self.idx)) if frames is None else frames
        kw = joint_kw or self.joint_kw
        ious = []
        for k in sel:
            self.tw.set_state(self.states[k], **kw)
            s = self.tw.silhouette(cam)
            r = self.real[cam][k]
            u = (s | r).sum()
            ious.append((s & r).sum() / u if u else 0.0)
        return float(np.mean(ious)), ious

    def coarse(self, cam, n_samples, rng, frames):
        lo, hi = SEARCH[cam]
        best = []
        for _ in range(n_samples):
            p = lo + rng.random(8) * (hi - lo)
            v = self.iou(cam, p, frames)[0]
            best.append((v, p))
        best.sort(key=lambda t: -t[0])
        return best

    def refine(self, cam, p0, maxiter=600):
        scale = np.array([0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 2.0, 4.0])
        f = lambda z: -self.iou(cam, p0 + z * scale)[0]  # noqa: E731
        simplex = np.vstack([np.zeros(8), np.eye(8)])          # explicit: scipy's default step at 0 is 0.00025
        res = optimize.minimize(f, np.zeros(8), method="Nelder-Mead",
                                options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-5, initial_simplex=simplex))
        return p0 + res.x * scale, -res.fun

    def refine_joints(self, cam_p, maxiter=300):
        """Nelder-Mead over the 5 joint offsets (deg) with both cameras fixed; returns the new offsets."""
        off0 = self.joint_kw["offsets_deg"].copy()

        def f(z):
            kw = dict(signs=self.joint_kw["signs"], offsets_deg=off0 + z)
            return -np.mean([self.iou(c, cam_p[c], joint_kw=kw)[0] for c in CAMS])

        res = optimize.minimize(f, np.zeros(5), method="Nelder-Mead",
                                options=dict(maxiter=maxiter, xatol=0.05, fatol=1e-5,
                                             initial_simplex=np.vstack([np.zeros(5), 4 * np.eye(5)])))
        return off0 + res.x, -res.fun


def overlay_sheet(fit: Fitter, cam: str, p, path: pathlib.Path, ncol=6):
    tw = Twin()
    tw.set_camera(cam, p[:3], p[3:6], p[6], p[7])
    tiles = []
    for k, i in enumerate(fit.idx):
        img = fit.fs.obs(i)["images"][cam].copy()
        tw.set_state(fit.states[k], **fit.joint_kw)
        s = tw.silhouette(cam)
        r = segment(img)
        edge = s & ~ndimage.binary_erosion(s, iterations=2)
        img[r & ~s] = (0.5 * img[r & ~s] + [120, 0, 0]).astype(np.uint8)
        img[edge] = [0, 255, 0]
        tiles.append(np.asarray(Image.fromarray(img).resize((320, 240))))
    while len(tiles) % ncol:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + ncol], 1) for i in range(0, len(tiles), ncol)]
    Image.fromarray(np.concatenate(rows, 0)).save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=18)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--no-joints", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="start from cameras.json / joints.json, skip the coarse search")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    fs = real_frames()
    idx = pick_frames(fs.state, a.frames, rng)
    print("frames:", idx)
    t0 = time.time()

    fit = Fitter(fs, idx, 160, 120)
    cam_p = {}
    if a.resume:
        cj = json.loads((HERE / "cameras.json").read_text())
        cam_p = {c: np.array(cj[c]["pos"] + cj[c]["lookat"] + [cj[c]["roll"], cj[c]["fovy"]]) for c in CAMS}
        for cam in CAMS:
            print(f"[{cam}] resumed IoU={fit.iou(cam, cam_p[cam])[0]:.3f}")
    for cam in [] if a.resume else CAMS:
        sub = list(range(0, len(idx), 3))
        best = fit.coarse(cam, a.samples, rng, sub)
        print(f"[{cam}] coarse best (6-frame IoU): {[round(b[0], 3) for b in best[:4]]}  {time.time() - t0:.0f}s")
        cands = []
        for v, p in best[:3]:
            p1, v1 = fit.refine(cam, p, maxiter=1500)
            cands.append((v1, p1))
            print(f"[{cam}]   refined {v:.3f} -> {v1:.3f}")
        cands.sort(key=lambda t: -t[0])
        cam_p[cam] = cands[0][1]
        print(f"[{cam}] round0 IoU={cands[0][0]:.3f} {unpack(cam_p[cam])}  {time.time() - t0:.0f}s")

    if not a.no_joints and not a.resume:
        # wrist-roll convention is ambiguous (sign, +-90): pick the best of 4 before the continuous refinement
        base_off = fit.joint_kw["offsets_deg"].copy()
        best = None
        for sgn in (1.0, -1.0):
            for roff in (0.0, 90.0, -90.0, 180.0):
                signs = fit.joint_kw["signs"].copy(); signs[4] = sgn
                off = base_off.copy(); off[4] = roff
                kw = dict(signs=signs, offsets_deg=off)
                v = np.mean([fit.iou(c, cam_p[c], joint_kw=kw)[0] for c in CAMS])
                print(f"  roll sign {sgn:+.0f} offset {roff:+.0f}: IoU {v:.4f}")
                if best is None or v > best[0]:
                    best = (v, signs, off)
        fit.joint_kw = dict(signs=best[1], offsets_deg=best[2])
    if not a.no_joints:
        for rnd in range(a.rounds):
            off, v = fit.refine_joints(cam_p)
            fit.joint_kw["offsets_deg"] = off
            print(f"[joints] round{rnd + 1}: IoU={v:.3f} offsets_deg={np.round(off, 1).tolist()}  {time.time() - t0:.0f}s")
            for cam in CAMS:
                cam_p[cam], v = fit.refine(cam, cam_p[cam], maxiter=600)
                print(f"[{cam}] round{rnd + 1}: IoU={v:.3f}")

    # final numbers at 320x240 with the final parameters
    fin = Fitter(fs, idx, 320, 240)
    fin.joint_kw = fit.joint_kw
    report = dict(frames=idx, cameras={}, joints=dict(signs=fit.joint_kw["signs"].tolist(),
                                                      offsets_deg=fit.joint_kw["offsets_deg"].tolist()))
    for cam in CAMS:
        mean, per = fin.iou(cam, cam_p[cam])
        report["cameras"][cam] = dict(unpack(cam_p[cam]), iou_mean=mean, iou_min=float(min(per)),
                                      iou_per_frame=[round(float(x), 3) for x in per])
        print(f"[{cam}] FINAL IoU mean={mean:.3f} min={min(per):.3f}  {unpack(cam_p[cam])}")
        overlay_sheet(fin, cam, cam_p[cam], OUT / f"fit_{cam}.jpg")
    (HERE / "cameras.json").write_text(json.dumps({c: unpack(cam_p[c]) for c in CAMS}, indent=1))
    (HERE / "joints.json").write_text(json.dumps(dict(signs=report["joints"]["signs"],
                                                      offsets_deg=report["joints"]["offsets_deg"],
                                                      jaw_pct_open=joints.JAW_PCT_OPEN), indent=1))
    (HERE / "fit_report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote cameras.json joints.json fit_report.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
