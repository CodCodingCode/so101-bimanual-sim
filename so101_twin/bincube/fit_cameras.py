"""Fit the twin's camera poses (and optionally joint offsets) to the real frames by arm-silhouette IoU.

    .venv/bin/python so101_twin/bincube/fit_cameras.py --cam side           # -> out/camera_fit_side.json, out/fit_side.png
    .venv/bin/python so101_twin/bincube/fit_cameras.py --cam top            # wrist camera (params in the Fixed_Jaw frame)
    .venv/bin/python so101_twin/bincube/fit_cameras.py --offsets            # alternate joint-offset / camera refits (both cams)
    .venv/bin/python so101_twin/bincube/fit_cameras.py --apply              # write the fitted CAMS into build_scene.py

Real arm masks (the arm is black):
  side: V = max(RGB) < DARK_THR, minus the static dark background (pixels dark in the per-pixel median over all
        320 frames: black breadboard, pegboard, the arm's base/rail and the controller), small holes filled.
  top (wrist cam): dark pixels in connected components touching the bottom image edge (the two jaws), holes filled.
Sim masks: segmentation render of the arm's visual geoms (side: without the static Base body, which is part of the
subtracted background; top: all arm geoms).
Objective: 1 - mean IoU over N_FRAMES real frames chosen by farthest-point sampling in joint space. Search: random
box search (per-camera box in BOX), then Nelder-Mead from the best few, at 320x240 (random stage at 160x120 for
the side camera); the reported IoU is at full 640x480.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
import twin  # noqa: E402
from twin import HERE, Sim, real_set  # noqa: E402

N_FRAMES = 18
KEYS = ("x", "y", "z", "yaw", "pitch", "roll", "fovy")
OUT = HERE / "out"
DARK_THR = 70
# random-search boxes: (lo, hi) over KEYS. side = world frame (wide, the seed is uncertain); top = Fixed_Jaw frame.
BOX = {
    "side": (np.array([-1.00, 0.30, 0.15, -90, -60, -10, 28]), np.array([-0.05, 1.10, 0.75, 0, -5, 10, 70])),
    "top": (np.array([-0.03, -0.03, -0.13, -115, 15, 165, 45]), np.array([0.03, 0.09, -0.045, -65, 60, 195, 95])),
}
STEP = {"side": np.array([0.04, 0.04, 0.04, 3, 3, 1.5, 4]), "top": np.array([0.005, 0.005, 0.005, 3, 3, 3, 3])}
_BG = {}


# ---- real masks ---------------------------------------------------------------------------------------------
def fill_holes(m: np.ndarray, min_hole: int) -> np.ndarray:
    holes, nh = ndimage.label(~m)
    if nh == 0:
        return m
    hs = ndimage.sum(~m, holes, range(1, nh + 1))
    return m | np.isin(holes, [i + 1 for i, s in enumerate(hs) if s < min_hole])


def dark_freq(fs, cam: str) -> np.ndarray:
    """float HxW: fraction of frames (every 2nd) in which the pixel is dark (V < DARK_THR); cached in out/."""
    if cam in _BG:
        return _BG[cam]
    f = OUT / f"dark_freq_{cam}.npy"
    if f.exists():
        _BG[cam] = np.load(f)
        return _BG[cam]
    fr = np.mean([(fs.obs(i)["images"][cam].max(-1) < DARK_THR) for i in range(0, fs.n, 2)], axis=0)
    np.save(f, fr)
    _BG[cam] = fr
    return fr


def static_dark_bg(fs, cam: str = "side") -> np.ndarray:
    """bool HxW: pixels dark in >= 97% of the frames = truly static dark background (breadboard, pegboard, bin interior,
    open-lid spot, arm base + controller). Both the real and the sim mask are evaluated outside this region."""
    return ndimage.binary_dilation(dark_freq(fs, cam) >= 0.97, iterations=2)


def jaw_region(fs, cam: str = "top") -> np.ndarray:
    """bool HxW: where the jaws ever are in the wrist view (dark in >= 25% of the frames), dilated."""
    return ndimage.binary_dilation(dark_freq(fs, cam) >= 0.25, iterations=6)


def real_arm_mask(rgb: np.ndarray, cam: str, bg: np.ndarray | None = None) -> np.ndarray:
    """bg: static_dark_bg() for "side", jaw_region() for "top"."""
    dark = rgb.max(-1) < DARK_THR
    if cam == "side":
        m = dark & ~bg
        lab, n = ndimage.label(m)
        if n:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            m = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= max(300, 0.05 * sizes.max())])
        return fill_holes(m, 1500)
    # wrist camera: dark components that touch the bottom edge, inside the region the jaws sweep
    dark = dark & bg
    lab, n = ndimage.label(dark)
    if n == 0:
        return dark
    keep = np.unique(lab[-1, :])
    keep = keep[keep > 0]
    m = np.isin(lab, keep)
    return fill_holes(m, 3000)


def downsample(mask: np.ndarray, k: int = 2) -> np.ndarray:
    H, W = mask.shape
    return mask.reshape(H // k, k, W // k, k).mean((1, 3)) > 0.5


def pick_frames(fs, n: int = N_FRAMES, seed: int = 0) -> list[int]:
    """Farthest-point sampling over the five revolute joints (normalised), seeded with the most extended pose."""
    st = fs.state[:, :5].astype(float)
    z = (st - st.mean(0)) / (st.std(0) + 1e-6)
    chosen = [int(np.argmax(st[:, 1]))]          # largest shoulder_lift = arm reaching out
    d = np.linalg.norm(z - z[chosen[0]], axis=1)
    while len(chosen) < n:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(z - z[i], axis=1))
    return chosen


# ---- sim side -----------------------------------------------------------------------------------------------
class Fitter:
    def __init__(self, cam: str, frames: list[int], fs, k: int = 2):
        self.cam = cam
        self.frames = frames
        self.fs = fs
        self.k = k
        self.sim = Sim(w=640 // k, h=480 // k)
        bg = static_dark_bg(fs) if cam == "side" else jaw_region(fs)
        # side: both masks are compared outside the static dark background. top: the real mask is gated to the
        # jaw region but the sim mask is NOT (otherwise a camera inside the arm fills the region with black).
        self.valid = ~bg if cam == "side" else np.ones_like(bg)
        self.valid_k = downsample(self.valid, k) if k > 1 else self.valid
        self.real_full = [real_arm_mask(fs.obs(i)["images"][cam], cam, bg) for i in frames]
        self.real = [downsample(m, k) if k > 1 else m for m in self.real_full]
        self.states = [fs.state[i] for i in frames]
        self.n_eval = 0

    def ious(self, p: np.ndarray) -> np.ndarray:
        lo, hi = BOX[self.cam]
        if np.any(p < lo - 1e-9) or np.any(p > hi + 1e-9):          # keep Nelder-Mead inside the box
            return np.zeros(len(self.frames))
        self.sim.set_camera(self.cam, p[:3], p[3], p[4], p[5], p[6])
        out = []
        for st, rm in zip(self.states, self.real):
            self.sim.set_joints(st)
            sm = self.sim.arm_mask(self.cam, skip_base=self.cam == "side") & self.valid_k
            inter = np.logical_and(sm, rm).sum()
            union = np.logical_or(sm, rm).sum()
            out.append(inter / max(union, 1))
        self.n_eval += 1
        return np.array(out)

    def loss(self, p: np.ndarray) -> float:
        return 1.0 - float(self.ious(p).mean())

    def overlay(self, p: np.ndarray, path: pathlib.Path, n: int = 8) -> None:
        self.sim.set_camera(self.cam, p[:3], p[3], p[4], p[5], p[6])
        tiles = []
        for j in np.linspace(0, len(self.frames) - 1, n).astype(int):
            i = self.frames[j]
            self.sim.set_joints(self.states[j])
            sm = self.sim.arm_mask(self.cam, skip_base=self.cam == "side") & self.valid_k
            rm = self.real[j]
            ov = np.zeros(sm.shape + (3,), np.uint8)
            ov[rm] = (200, 60, 60)
            ov[sm] = (60, 60, 220)
            ov[rm & sm] = (240, 240, 240)
            real = np.asarray(Image.fromarray(self.fs.obs(i)["images"][self.cam]).resize(sm.shape[::-1]))
            tiles.append(np.concatenate([real, self.sim.render(self.cam), ov], 1))
        Image.fromarray(np.concatenate(tiles, 0)).save(path)


def p0_from_cams(cam: str) -> np.ndarray:
    c = build_scene.CAMS[cam]
    return np.array([*c["pos"], c["yaw"], c["pitch"], c["roll"], c["fovy"]], float)


def fit_camera(cam: str, fs, frames, n_random=400, n_refine=4, seed=0, p0=None, box=True) -> tuple[np.ndarray, float]:
    """box=True: uniform samples over BOX[cam] (coarse, 160x120, 6 frames); else around p0 with a small span."""
    rng = np.random.default_rng(seed)
    p0 = p0_from_cams(cam) if p0 is None else np.asarray(p0, float)
    t0 = time.time()
    lo, hi = BOX[cam]
    if box:
        fc = Fitter(cam, frames[:6], fs, k=4 if cam == "side" else 2)
        cands = [(fc.loss(p0), p0)]
        for _ in range(n_random):
            p = rng.uniform(lo, hi)
            cands.append((fc.loss(p), p))
        fc.sim.close()
    else:
        fc = Fitter(cam, frames, fs, k=2)
        span = STEP[cam] * np.array([4, 4, 4, 4, 4, 3, 4])
        cands = [(fc.loss(p0), p0)]
        for _ in range(n_random):
            p = np.clip(p0 + rng.uniform(-1, 1, 7) * span, lo, hi)
            cands.append((fc.loss(p), p))
        fc.sim.close()
    cands.sort(key=lambda t: t[0])
    print(f"[{cam}] random search: best loss {cands[0][0]:.3f} (IoU {1 - cands[0][0]:.3f}) after {time.time() - t0:.0f}s",
          flush=True)
    f = Fitter(cam, frames, fs, k=2)
    best = (f.loss(cands[0][1]), cands[0][1])
    step = STEP[cam]
    for L, p in cands[:n_refine]:
        simplex = np.stack([p] + [p + np.eye(7)[i] * step[i] for i in range(7)])
        res = minimize(f.loss, p, method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, xatol=1e-3, fatol=1e-4, maxiter=700, maxfev=900))
        print(f"[{cam}] NM from {L:.3f} -> {res.fun:.3f} (IoU {1 - res.fun:.3f}), {res.nfev} evals, "
              f"{time.time() - t0:.0f}s  p={np.round(res.x, 3)}", flush=True)
        if res.fun < best[0]:
            best = (res.fun, res.x)
    f.sim.close()
    return best[1], 1 - best[0]


def full_res_iou(cam: str, fs, frames, p) -> tuple[float, np.ndarray]:
    f = Fitter(cam, frames, fs, k=1)
    ious = f.ious(p)
    f.overlay(p, OUT / f"fit_{cam}.png")
    f.sim.close()
    return float(ious.mean()), ious


def save_fit(cam: str, p, iou_small, iou_full, ious, frames) -> None:
    d = dict(cam=cam, frame=build_scene.CAMS[cam].get("body", "world"), params=dict(zip(KEYS, map(float, p))),
             iou_320x240=float(iou_small), iou_640x480=float(iou_full),
             iou_per_frame=[float(x) for x in ious], frames=[int(i) for i in frames],
             signs=twin.SIGNS.tolist(), offsets_deg=twin.OFFSETS_DEG.tolist())
    (OUT / f"camera_fit_{cam}.json").write_text(json.dumps(d, indent=1))


def load_fit(cam: str) -> np.ndarray:
    d = json.loads((OUT / f"camera_fit_{cam}.json").read_text())
    return np.array([d["params"][k] for k in KEYS])


def apply_to_build_scene() -> None:
    """Rewrite the CAMS lines of build_scene.py with the fitted parameters (and twin.py OFFSETS_DEG if fitted)."""
    p = HERE / "build_scene.py"
    s = p.read_text()
    for cam in twin.CAMS:
        v = load_fit(cam)
        body = f', body="{build_scene.CAMS[cam]["body"]}"' if "body" in build_scene.CAMS[cam] else ""
        line = (f'    "{cam}": dict(pos=({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}), yaw={v[3]:.2f}, pitch={v[4]:.2f}, '
                f'roll={v[5]:.2f}, fovy={v[6]:.2f}{body}),')
        s, n = re.subn(rf'^    "{cam}": dict\(pos=.*\),$', line, s, flags=re.M)
        assert n == 1, cam
    p.write_text(s)
    off = OUT / "offsets_fit.json"
    if off.exists():
        o = json.loads(off.read_text())["offsets_deg"]
        t = HERE / "twin.py"
        s = t.read_text()
        line = "OFFSETS_DEG = np.array([" + ", ".join(f"{x:.2f}" for x in o) + "])   # fit_cameras.py --offsets"
        s, n = re.subn(r"^OFFSETS_DEG = np\.array\(\[.*$", line, s, flags=re.M)
        assert n == 1
        t.write_text(s)
    import importlib

    importlib.reload(build_scene).write()
    print("updated build_scene.py CAMS (and twin.py OFFSETS_DEG) and rebuilt the XML")


def fit_offsets(fs, frames, cams_p: dict[str, np.ndarray], joints=(0, 1, 2, 3, 4)) -> np.ndarray:
    """Nelder-Mead over OFFSETS_DEG[joints] with both cameras fixed, objective = mean IoU over both cams."""
    fits = {c: Fitter(c, frames, fs, k=2) for c in twin.CAMS}
    base = twin.OFFSETS_DEG.copy()

    def loss(x):
        off = base.copy()
        off[list(joints)] = x
        twin.OFFSETS_DEG[:] = off
        return 1 - np.mean([fits[c].ious(cams_p[c]).mean() for c in twin.CAMS])

    x0 = base[list(joints)]
    res = minimize(loss, x0, method="Nelder-Mead", options=dict(initial_simplex=np.stack([x0] + [x0 + 4 * np.eye(len(joints))[i] for i in range(len(joints))]),
                                                                xatol=0.05, fatol=1e-4, maxiter=300))
    off = base.copy()
    off[list(joints)] = res.x
    twin.OFFSETS_DEG[:] = off
    for f in fits.values():
        f.sim.close()
    print(f"offsets -> {np.round(off, 2)} (mean IoU {1 - res.fun:.3f})", flush=True)
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", choices=twin.CAMS)
    ap.add_argument("--offsets", action="store_true", help="alternate offset / camera refits starting from the saved fits")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--n-random", type=int, default=400)
    ap.add_argument("--n-refine", type=int, default=4)
    ap.add_argument("--local", action="store_true", help="random search around the current CAMS instead of the BOX")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.apply:
        apply_to_build_scene()
        return
    fs = real_set()
    frames = pick_frames(fs)
    print("frames:", frames, flush=True)
    if a.cam:
        p, iou_s = fit_camera(a.cam, fs, frames, n_random=a.n_random, n_refine=a.n_refine, seed=a.seed, box=not a.local)
        iou_f, ious = full_res_iou(a.cam, fs, frames, p)
        save_fit(a.cam, p, iou_s, iou_f, ious, frames)
        print(f"[{a.cam}] final params {dict(zip(KEYS, np.round(p, 4)))}\n[{a.cam}] IoU 320x240 {iou_s:.3f}, "
              f"640x480 {iou_f:.3f}, per frame {np.round(ious, 2)}", flush=True)
    if a.offsets:
        cams_p = {c: load_fit(c) for c in twin.CAMS}
        for r in range(a.rounds):
            off = fit_offsets(fs, frames, cams_p)
            (OUT / "offsets_fit.json").write_text(json.dumps(dict(offsets_deg=twin.OFFSETS_DEG.tolist(), signs=twin.SIGNS.tolist())))
            for c in twin.CAMS:
                p, iou_s = fit_camera(c, fs, frames, n_random=60, n_refine=2, seed=a.seed + r, p0=cams_p[c], box=False)
                cams_p[c] = p
                iou_f, ious = full_res_iou(c, fs, frames, p)
                save_fit(c, p, iou_s, iou_f, ious, frames)
                print(f"round {r} [{c}] IoU 640x480 {iou_f:.3f} offsets {np.round(off, 2)}", flush=True)


if __name__ == "__main__":
    main()
