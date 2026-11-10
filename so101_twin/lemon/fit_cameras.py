"""Fit the twin's camera poses (and optionally joint offsets) to the real frames by arm-silhouette IoU.

    .venv/bin/python so101_twin/lemon/fit_cameras.py --cam top            # -> out/camera_fit_top.json, out/fit_top.png
    .venv/bin/python so101_twin/lemon/fit_cameras.py --cam side
    .venv/bin/python so101_twin/lemon/fit_cameras.py --offsets            # alternate joint-offset / camera refits (both cams)
    .venv/bin/python so101_twin/lemon/fit_cameras.py --apply              # write the fitted CAMS into build_scene.py

Real arm mask: orange (arm body) or blue (jaws) pixels by HSV threshold. Sim mask: segmentation render of the
arm's orange + blue visual geoms (black motor geoms are holes in both). Objective: 1 - mean IoU over N_FRAMES
real frames chosen by farthest-point sampling in joint space. Search: random box search around the initial guess
in build_scene.CAMS, then Nelder-Mead from the best few, at 320x240; the reported IoU is at full 640x480.
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
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_scene  # noqa: E402
import twin  # noqa: E402
from twin import HERE, Sim, real_set  # noqa: E402

N_FRAMES = 18
KEYS = ("x", "y", "z", "yaw", "pitch", "roll", "fovy")
OUT = HERE / "out"


# ---- real masks ---------------------------------------------------------------------------------------------
def real_arm_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    orange = (h >= 3) & (h <= 22) & (s > 110) & (v > 50)
    blue = (h >= 128) & (h <= 165) & (s > 90) & (v > 12)
    return orange | blue


def downsample(mask: np.ndarray, k: int = 2) -> np.ndarray:
    H, W = mask.shape
    return mask.reshape(H // k, k, W // k, k).mean((1, 3)) > 0.5


def pick_frames(fs, n: int = N_FRAMES, seed: int = 0) -> list[int]:
    """Farthest-point sampling over the five revolute joints (normalised), seeded with the most extended pose."""
    st = fs.state[:, :5].astype(float)
    z = (st - st.mean(0)) / (st.std(0) + 1e-6)
    chosen = [int(np.argmin(st[:, 1]))]          # smallest shoulder_lift = arm reaching out
    d = np.linalg.norm(z - z[chosen[0]], axis=1)
    while len(chosen) < n:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(z - z[i], axis=1))
    return chosen


# ---- sim side -----------------------------------------------------------------------------------------------
class Fitter:
    def __init__(self, cam: str, frames: list[int], fs, small=True):
        self.cam = cam
        self.frames = frames
        self.fs = fs
        self.k = 2 if small else 1
        self.sim = Sim(w=640 // self.k, h=480 // self.k)
        self.real = [downsample(real_arm_mask(fs.obs(i)["images"][cam]), self.k) if small else
                     real_arm_mask(fs.obs(i)["images"][cam]) for i in frames]
        self.states = [fs.state[i] for i in frames]
        self.n_eval = 0

    def ious(self, p: np.ndarray) -> np.ndarray:
        self.sim.set_camera(self.cam, p[:3], p[3], p[4], p[5], p[6])
        out = []
        for st, rm in zip(self.states, self.real):
            self.sim.set_joints(st)
            sm = self.sim.arm_mask(self.cam)
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
            sm = self.sim.arm_mask(self.cam)
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


def fit_camera(cam: str, fs, frames, n_random=400, n_refine=4, seed=0, p0=None) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    f = Fitter(cam, frames, fs, small=True)
    p0 = p0_from_cams(cam) if p0 is None else np.asarray(p0, float)
    span = np.array([0.25, 0.25, 0.25, 25, 20, 6, 18])
    t0 = time.time()
    cands = [(f.loss(p0), p0)]
    for _ in range(n_random):
        p = p0 + rng.uniform(-1, 1, 7) * span
        p[6] = np.clip(p[6], 25, 90)
        cands.append((f.loss(p), p))
    cands.sort(key=lambda t: t[0])
    print(f"[{cam}] random search: best loss {cands[0][0]:.3f} (IoU {1 - cands[0][0]:.3f}) after {time.time() - t0:.0f}s",
          flush=True)
    best = cands[0]
    step = np.array([0.03, 0.03, 0.03, 3, 3, 1.5, 3])
    for L, p in cands[:n_refine]:
        simplex = np.stack([p] + [p + np.eye(7)[i] * step[i] for i in range(7)])
        res = minimize(f.loss, p, method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, xatol=1e-3, fatol=1e-4, maxiter=700, maxfev=900))
        print(f"[{cam}] NM from {L:.3f} -> {res.fun:.3f} (IoU {1 - res.fun:.3f}), {res.nfev} evals, "
              f"{time.time() - t0:.0f}s", flush=True)
        if res.fun < best[0]:
            best = (res.fun, res.x)
    f.sim.close()
    return best[1], 1 - best[0]


def full_res_iou(cam: str, fs, frames, p) -> tuple[float, np.ndarray]:
    f = Fitter(cam, frames, fs, small=False)
    ious = f.ious(p)
    f.overlay(p, OUT / f"fit_{cam}.png")
    f.sim.close()
    return float(ious.mean()), ious


def save_fit(cam: str, p, iou_small, iou_full, ious, frames) -> None:
    d = dict(cam=cam, params=dict(zip(KEYS, map(float, p))), iou_320x240=float(iou_small), iou_640x480=float(iou_full),
             iou_per_frame=[float(x) for x in ious], frames=[int(i) for i in frames],
             signs=twin.SIGNS.tolist(), offsets_deg=twin.OFFSETS_DEG.tolist())
    (OUT / f"camera_fit_{cam}.json").write_text(json.dumps(d, indent=1))


def load_fit(cam: str) -> np.ndarray:
    d = json.loads((OUT / f"camera_fit_{cam}.json").read_text())
    return np.array([d["params"][k] for k in KEYS])


def apply_to_build_scene() -> None:
    """Rewrite the CAMS lines of build_scene.py with the fitted parameters."""
    p = HERE / "build_scene.py"
    s = p.read_text()
    for cam in twin.CAMS:
        v = load_fit(cam)
        line = (f'    "{cam}": dict(pos=({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}), yaw={v[3]:.2f}, pitch={v[4]:.2f}, '
                f'roll={v[5]:.2f}, fovy={v[6]:.2f}),')
        s, n = re.subn(rf'^    "{cam}": dict\(pos=.*\),$', line, s, flags=re.M)
        assert n == 1, cam
    p.write_text(s)
    import importlib

    importlib.reload(build_scene).write()
    print("updated build_scene.py CAMS and rebuilt the XML")


def fit_offsets(fs, frames, cams_p: dict[str, np.ndarray], joints=(1, 2, 3)) -> np.ndarray:
    """Nelder-Mead over OFFSETS_DEG[joints] with both cameras fixed, objective = mean IoU over both cams."""
    fits = {c: Fitter(c, frames, fs, small=True) for c in twin.CAMS}
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
        p, iou_s = fit_camera(a.cam, fs, frames, n_random=a.n_random, seed=a.seed)
        iou_f, ious = full_res_iou(a.cam, fs, frames, p)
        save_fit(a.cam, p, iou_s, iou_f, ious, frames)
        print(f"[{a.cam}] final params {dict(zip(KEYS, np.round(p, 4)))}\n[{a.cam}] IoU 320x240 {iou_s:.3f}, "
              f"640x480 {iou_f:.3f}, per frame {np.round(ious, 2)}", flush=True)
    if a.offsets:
        cams_p = {c: load_fit(c) for c in twin.CAMS}
        for r in range(a.rounds):
            off = fit_offsets(fs, frames, cams_p)
            for c in twin.CAMS:
                p, iou_s = fit_camera(c, fs, frames, n_random=60, n_refine=2, seed=a.seed + r, p0=cams_p[c])
                cams_p[c] = p
                iou_f, ious = full_res_iou(c, fs, frames, p)
                save_fit(c, p, iou_s, iou_f, ious, frames)
                print(f"round {r} [{c}] IoU 640x480 {iou_f:.3f} offsets {np.round(off, 2)}", flush=True)
        (OUT / "offsets_fit.json").write_text(json.dumps(dict(offsets_deg=twin.OFFSETS_DEG.tolist(), signs=twin.SIGNS.tolist())))


if __name__ == "__main__":
    main()
