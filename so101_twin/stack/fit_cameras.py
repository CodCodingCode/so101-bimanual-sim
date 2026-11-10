"""Fit the twin's camera poses (and optionally joint offsets) to the real frames by arm-silhouette IoU.

    .venv/bin/python so101_twin/stack/fit_cameras.py --cam top            # -> out/camera_fit_top.json, out/fit_top.png
    .venv/bin/python so101_twin/stack/fit_cameras.py --cam side
    .venv/bin/python so101_twin/stack/fit_cameras.py --offsets            # alternate joint-offset / camera refits (both cams)
    .venv/bin/python so101_twin/stack/fit_cameras.py --apply              # write the fitted CAMS into build_scene.py

Real arm mask: the arm is WHITE with black motors on a light-grey table / blue cloth, so a plain colour threshold
does not separate it. Per camera a median background (every 4th frame, out/bg_<cam>.npy) is used:
  top : pixels clearly BRIGHTER than the background with low saturation (white links; the arm's shadow is darker
        and so excluded) OR near-black pixels where the background was not dark (motors).
  side: low-saturation bright pixels where the background was NOT white (arm against the blue cloth) OR near-black
        pixels where the background was not dark. White arm over the white table / disc / panel is lost.
Sim mask: segmentation render of all the arm's visual geoms (white links + black motors). Objective: 1 - mean IoU
over N_FRAMES real frames chosen by farthest-point sampling in joint space. Search: random box search around the
initial guess in build_scene.CAMS, then Nelder-Mead from the best few, at 320x240; the reported IoU is at 640x480.
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
OBJ_KEYS = ("green_x", "green_y", "green_half")   # top camera only (10 params); the disc is placed by place_objects.py
# random-search half-widths around build_scene.CAMS: x y z (m), yaw pitch roll fovy (deg)
SPAN = {"top": (0.30, 0.30, 0.20, 40, 25, 30, 20), "side": (0.25, 0.25, 0.10, 30, 15, 10, 20)}
OUT = HERE / "out"


# ---- real masks ---------------------------------------------------------------------------------------------
_BG: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def background(cam: str, fs=None) -> tuple[np.ndarray, np.ndarray]:
    """(rgb uint8, hsv int) median background of the real camera over every 4th frame (cached in out/bg_<cam>.npy)."""
    if cam not in _BG:
        p = OUT / f"bg_{cam}.npy"
        if not p.exists():
            fs = real_set() if fs is None else fs
            imgs = np.stack([fs.obs(i)["images"][cam] for i in range(0, fs.n, 4)])
            np.save(p, np.median(imgs, 0).astype(np.uint8))
        bg = np.load(p)
        _BG[cam] = (bg, np.asarray(Image.fromarray(bg).convert("HSV")).astype(int))
    return _BG[cam]


def real_arm_mask(rgb: np.ndarray, cam: str = "top") -> np.ndarray:
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    s, v = hsv[..., 1], hsv[..., 2]
    bgh = background(cam)[1]
    bgs, bgv = bgh[..., 1], bgh[..., 2]
    if cam == "top":
        white = (v - bgv > 18) & (s < 50) & (v > 120)
        black = (v < 70) & (bgv > 110)
    else:
        white = (v > 185) & (s < 55) & ~((bgv > 185) & (bgs < 55))
        black = (v < 70) & (bgv >= 70)
    return white | black


def real_green_mask(rgb: np.ndarray, cam: str = "side") -> np.ndarray:
    """Green block by hue. The top view is dim (the block's top face is V ~ 40-60) and has a green PCB/LED in the
    base area (u > 450), which is excluded there."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (h >= 65) & (h <= 130) & (s > 45) & (v > 15)
    if cam == "top":
        m[:, 450:] = False
    return m


def real_disc_mask(rgb: np.ndarray, cam: str) -> np.ndarray:
    """White paper disc: much brighter than the grey table in the top view (not usable in the side view)."""
    if cam != "top":
        return np.zeros(rgb.shape[:2], bool)
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    s, v = hsv[..., 1], hsv[..., 2]
    bgv = background(cam)[1][..., 2]
    return (v > 200) & (s < 30) & (bgv > 190)


def ignore_mask(cam: str, fs=None) -> np.ndarray:
    """bool HxW: pixels excluded from both masks. The blue cloth edge (top view) moves between episodes, so pixels
    that are cloth-blue in 15..85 % of the frames are ignored; for the side camera nothing is ignored."""
    if cam != "top":
        return np.zeros((480, 640), bool)
    p = OUT / f"bluefrac_{cam}.npy"
    if not p.exists():
        fs = real_set() if fs is None else fs
        acc, n = np.zeros((480, 640)), 0
        for i in range(0, fs.n, 2):
            hsv = np.asarray(Image.fromarray(fs.obs(i)["images"][cam]).convert("HSV")).astype(int)
            acc += (hsv[..., 0] > 120) & (hsv[..., 0] < 175) & (hsv[..., 1] > 50)
            n += 1
        np.save(p, (acc / n).astype(np.float32))
    frac = np.load(p)
    return (frac > 0.15) & (frac < 0.85)


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
    def __init__(self, cam: str, frames: list[int], fs, small=True, k: int | None = None):
        self.cam = cam
        self.frames = frames
        self.fs = fs
        self.k = (2 if small else 1) if k is None else k
        small = self.k > 1
        self.sim = Sim(w=640 // self.k, h=480 // self.k)
        ign = ignore_mask(cam, fs)
        self.ignore = downsample(ign, self.k) if small else ign
        self.real = [downsample(real_arm_mask(fs.obs(i)["images"][cam], cam), self.k) if small else
                     real_arm_mask(fs.obs(i)["images"][cam], cam) for i in frames]
        self.real = [rm & ~self.ignore for rm in self.real]
        ds = (lambda m: downsample(m, self.k)) if small else (lambda m: m)
        self.real_green = [ds(real_green_mask(fs.obs(i)["images"][cam], cam)) for i in frames]
        self.real_disc = [ds(real_disc_mask(fs.obs(i)["images"][cam], cam)) for i in frames]
        self.terms = ("arm", "green")      # the disc is not separable from the table (see real_disc_mask)
        self.states = [fs.state[i] for i in frames]
        self.n_eval = 0

    def set_params(self, p: np.ndarray) -> None:
        """p = [x y z yaw pitch roll fovy] (+ [green x, y, half] when len(p) == 10)."""
        self.sim.set_camera(self.cam, p[:3], p[3], p[4], p[5], p[6])
        if len(p) >= 10:
            self.sim.set_object("green", p[7:9], (p[9], p[9], p[9]))

    @staticmethod
    def iou(a, b) -> float:
        return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)

    def ious(self, p: np.ndarray, per_term: bool = False) -> np.ndarray:
        """mean IoU per frame over the terms (arm silhouette, green block, disc)."""
        self.set_params(p)
        out = []
        for j, (st, rm) in enumerate(zip(self.states, self.real)):
            self.sim.set_joints(st)
            seg = self.sim.seg(self.cam)
            sm = np.isin(seg, self.sim.arm_geoms) & ~self.ignore
            t = [self.iou(sm, rm)]
            if "green" in self.terms:
                t.append(self.iou(seg == self.sim.obj_geom["green"], self.real_green[j]))
            if "disc" in self.terms:
                t.append(self.iou(seg == self.sim.obj_geom["disc"], self.real_disc[j]))
            out.append(t if per_term else float(np.mean(t)))
        self.n_eval += 1
        return np.array(out)

    def loss(self, p: np.ndarray) -> float:
        return 1.0 - float(self.ious(p).mean())

    def overlay(self, p: np.ndarray, path: pathlib.Path, n: int = 8) -> None:
        self.set_params(p)
        tiles = []
        for j in np.linspace(0, len(self.frames) - 1, n).astype(int):
            i = self.frames[j]
            self.sim.set_joints(self.states[j])
            seg = self.sim.seg(self.cam)
            sm = np.isin(seg, self.sim.arm_geoms) & ~self.ignore
            rm = self.real[j]
            if "green" in self.terms:
                sm = sm | (seg == self.sim.obj_geom["green"]); rm = rm | self.real_green[j]
            if "disc" in self.terms:
                sm = sm | (seg == self.sim.obj_geom["disc"]); rm = rm | self.real_disc[j]
            ov = np.zeros(sm.shape + (3,), np.uint8)
            ov[rm] = (200, 60, 60)
            ov[sm] = (60, 60, 220)
            ov[rm & sm] = (240, 240, 240)
            real = np.asarray(Image.fromarray(self.fs.obs(i)["images"][self.cam]).resize(sm.shape[::-1]))
            tiles.append(np.concatenate([real, self.sim.render(self.cam), ov], 1))
        Image.fromarray(np.concatenate(tiles, 0)).save(path)


# wide box for the global search: x y z (m), yaw pitch roll fovy (deg)
GLOBAL_LO = {"top": (-0.9, -0.5, 0.25, -180, -90, -45, 35), "side": (-0.7, -0.3, 0.0, -180, -30, -30, 30)}
GLOBAL_HI = {"top": (0.9, 1.0, 1.3, 180, -25, 45, 95), "side": (0.7, 0.9, 0.35, 180, 5, 30, 80)}
OBJ_LO, OBJ_HI = (-0.35, 0.02, 0.038), (0.35, 0.60, 0.042)   # block half-size pinned ~4 cm (size/distance is degenerate)
SPAN_OBJ = (0.06, 0.06, 0.012)
STEP_OBJ = (0.01, 0.01, 0.004)


def global_search(cam: str, fs, frames, n: int = 3000, seed: int = 0, n_keep: int = 6) -> list[np.ndarray]:
    """Uniform random search over GLOBAL_LO..HI at 160x120 on the first 6 frames; returns the best n_keep params."""
    rng = np.random.default_rng(seed)
    f = Fitter(cam, frames[:6], fs, k=4)
    lo, hi = np.array(GLOBAL_LO[cam], float), np.array(GLOBAL_HI[cam], float)
    base = p0_from_cams(cam)
    if cam == "top":                       # objects: sample around the build_scene guess, not the whole box
        lo = np.concatenate([lo, np.maximum(base[7:] - 2 * np.array(SPAN_OBJ), OBJ_LO)])
        hi = np.concatenate([hi, np.minimum(base[7:] + 2 * np.array(SPAN_OBJ), OBJ_HI)])
    t0 = time.time()
    cands = []
    for i in range(n):
        p = lo + rng.uniform(0, 1, len(lo)) * (hi - lo)
        cands.append((f.loss(p), p))
        if (i + 1) % 1000 == 0:
            cands.sort(key=lambda t: t[0])
            print(f"[{cam}] global {i + 1}/{n}: best IoU {1 - cands[0][0]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    cands.sort(key=lambda t: t[0])
    f.sim.close()
    # keep distinct candidates (at least 0.1 m / 10 deg apart)
    keep = []
    for L, p in cands:
        if all(np.linalg.norm((p - q)[:3]) > 0.1 or np.abs(p - q)[3:6].max() > 10 for q in keep):
            keep.append(p)
        if len(keep) >= n_keep:
            break
    return keep


def p0_from_cams(cam: str) -> np.ndarray:
    c = build_scene.CAMS[cam]
    p = [*c["pos"], c["yaw"], c["pitch"], c["roll"], c["fovy"]]
    if cam == "top":
        g = build_scene.OBJECTS["green"]
        p += [*g["xy"], g["size"][0]]
    return np.array(p, float)


def fit_camera(cam: str, fs, frames, n_random=400, n_refine=4, seed=0, p0=None) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    f = Fitter(cam, frames, fs, small=True)
    p0 = p0_from_cams(cam) if p0 is None else np.asarray(p0, float)
    span = np.array(SPAN[cam] + (SPAN_OBJ if len(p0) > 7 else ()))
    nparam = len(p0)
    t0 = time.time()
    cands = [(f.loss(p0), p0)]
    for _ in range(n_random):
        p = p0 + rng.uniform(-1, 1, nparam) * span
        p[6] = np.clip(p[6], 25, 90)
        if nparam > 7:
            p[7:] = np.clip(p[7:], OBJ_LO, OBJ_HI)
        cands.append((f.loss(p), p))
    cands.sort(key=lambda t: t[0])
    print(f"[{cam}] random search: best loss {cands[0][0]:.3f} (IoU {1 - cands[0][0]:.3f}) after {time.time() - t0:.0f}s",
          flush=True)
    best = cands[0]
    step = np.array([0.03, 0.03, 0.03, 3, 3, 1.5, 3] + (list(STEP_OBJ) if nparam > 7 else []))
    for L, p in cands[:n_refine]:
        simplex = np.stack([p] + [p + np.eye(nparam)[i] * step[i] for i in range(nparam)])
        res = minimize(f.loss, p, method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, xatol=1e-3, fatol=1e-4, maxiter=1200, maxfev=1500))
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
    d = dict(cam=cam, params=dict(zip(KEYS + OBJ_KEYS, map(float, p))), iou_320x240=float(iou_small), iou_640x480=float(iou_full),
             iou_per_frame=[float(x) for x in ious], frames=[int(i) for i in frames],
             signs=twin.SIGNS[:5].tolist(), offsets_deg=twin.OFFSETS_DEG[:5].tolist(), scales=twin.SCALES[:5].tolist())
    (OUT / f"camera_fit_{cam}.json").write_text(json.dumps(d, indent=1))


def load_fit(cam: str) -> np.ndarray:
    d = json.loads((OUT / f"camera_fit_{cam}.json").read_text())
    keys = KEYS + (OBJ_KEYS if cam == "top" else ())
    return np.array([d["params"][k] for k in keys])


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
    v = load_fit("top")
    s, n = re.subn(r'^    "green": dict\(xy=\([^)]*\), size=\([^)]*\),',
                   f'    "green": dict(xy=({v[7]:.3f}, {v[8]:.3f}), size=({v[9]:.3f}, {v[9]:.3f}, {v[9]:.3f}),', s, flags=re.M)
    assert n == 1, "green"
    p.write_text(s)
    import importlib

    importlib.reload(build_scene).write()
    print("updated build_scene.py CAMS and rebuilt the XML")


def fit_joint(fs, frames, cams_p: dict[str, np.ndarray], maxfev: int = 1500) -> dict[str, np.ndarray]:
    """Nelder-Mead over both cameras + the shared green-block (x, y, half): 17 params, objective = mean IoU of both cams.
    cams_p['top'] carries the green params (10), cams_p['side'] is 7; returns the same layout."""
    fits = {c: Fitter(c, frames, fs, small=True) for c in twin.CAMS}
    x0 = np.concatenate([cams_p["top"][:7], cams_p["side"][:7], cams_p["top"][7:10]])

    def unpack(x):
        g = np.clip(x[14:17], OBJ_LO, OBJ_HI)
        return {"top": np.concatenate([x[:7], g]), "side": np.concatenate([x[7:14], g])}

    def loss(x):
        p = unpack(x)
        return 1 - np.mean([fits[c].ious(p[c]).mean() for c in twin.CAMS])

    step = np.array([0.03, 0.03, 0.03, 3, 3, 1.5, 3] * 2 + list(STEP_OBJ))
    simplex = np.stack([x0] + [x0 + np.eye(17)[i] * step[i] for i in range(17)])
    t0 = time.time()
    L0 = loss(x0)
    res = minimize(loss, x0, method="Nelder-Mead",
                   options=dict(initial_simplex=simplex, xatol=1e-3, fatol=1e-4, maxiter=maxfev, maxfev=maxfev))
    print(f"joint NM: IoU {1 - L0:.3f} -> {1 - res.fun:.3f}, {res.nfev} evals, {time.time() - t0:.0f}s", flush=True)
    for f in fits.values():
        f.sim.close()
    return unpack(res.x)


def fit_offsets(fs, frames, cams_p: dict[str, np.ndarray], joints=(1, 2, 3)) -> np.ndarray:
    """Nelder-Mead over OFFSETS_DEG[joints] and SCALES[joints] with both cameras fixed; objective = mean IoU, both cams."""
    fits = {c: Fitter(c, frames, fs, small=True) for c in twin.CAMS}
    base_off, base_sc = twin.OFFSETS_DEG.copy(), twin.SCALES.copy()
    j = list(joints)

    def apply(x):
        off, sc = base_off.copy(), base_sc.copy()
        off[j] = x[:len(j)]
        sc[j] = np.clip(x[len(j):], 0.7, 1.4)
        twin.OFFSETS_DEG[:] = off
        twin.SCALES[:] = sc

    def loss(x):
        apply(x)
        return 1 - np.mean([fits[c].ious(cams_p[c]).mean() for c in twin.CAMS])

    x0 = np.concatenate([base_off[j], base_sc[j]])
    step = np.concatenate([np.full(len(j), 5.0), np.full(len(j), 0.08)])
    simplex = np.stack([x0] + [x0 + np.eye(len(x0))[i] * step[i] for i in range(len(x0))])
    res = minimize(loss, x0, method="Nelder-Mead",
                   options=dict(initial_simplex=simplex, xatol=0.05, fatol=1e-4, maxiter=500, maxfev=600))
    apply(res.x)
    for f in fits.values():
        f.sim.close()
    print(f"offsets -> {np.round(twin.OFFSETS_DEG[:5], 2)} scales -> {np.round(twin.SCALES[:5], 3)} "
          f"(mean IoU {1 - res.fun:.3f})", flush=True)
    return twin.OFFSETS_DEG.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", choices=twin.CAMS)
    ap.add_argument("--offsets", action="store_true", help="alternate offset / camera refits starting from the saved fits")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--n-random", type=int, default=400)
    ap.add_argument("--global", dest="glob", action="store_true", help="wide random search first (160x120), NM from the best 6")
    ap.add_argument("--joint", action="store_true", help="joint NM over both cameras + green block, then --offsets rounds")
    ap.add_argument("--from-scene", action="store_true", help="--joint: start from build_scene.CAMS / OBJECTS instead of out/camera_fit_*.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.apply:
        apply_to_build_scene()
        return
    fs = real_set()
    frames = pick_frames(fs)
    print("frames:", frames, flush=True)
    if a.cam and a.glob:
        best = (None, 0.0)
        for k, p0 in enumerate(global_search(a.cam, fs, frames, seed=a.seed)):
            p, iou_s = fit_camera(a.cam, fs, frames, n_random=0, n_refine=1, seed=a.seed, p0=p0)
            print(f"[{a.cam}] candidate {k}: IoU 320x240 {iou_s:.3f} params {np.round(p, 3)}", flush=True)
            if iou_s > best[1]:
                best = (p, iou_s)
        p, iou_s = best
        iou_f, ious = full_res_iou(a.cam, fs, frames, p)
        save_fit(a.cam, p, iou_s, iou_f, ious, frames)
        print(f"[{a.cam}] final params {dict(zip(KEYS, np.round(p, 4)))}\n[{a.cam}] IoU 320x240 {iou_s:.3f}, "
              f"640x480 {iou_f:.3f}, per frame {np.round(ious, 2)}", flush=True)
    elif a.cam:
        p, iou_s = fit_camera(a.cam, fs, frames, n_random=a.n_random, seed=a.seed)
        iou_f, ious = full_res_iou(a.cam, fs, frames, p)
        save_fit(a.cam, p, iou_s, iou_f, ious, frames)
        print(f"[{a.cam}] final params {dict(zip(KEYS, np.round(p, 4)))}\n[{a.cam}] IoU 320x240 {iou_s:.3f}, "
              f"640x480 {iou_f:.3f}, per frame {np.round(ious, 2)}", flush=True)
    if a.joint:
        cams_p = {c: (p0_from_cams(c) if a.from_scene else load_fit(c)) for c in twin.CAMS}
        cams_p["side"] = cams_p["side"][:7]
        for r in range(a.rounds):
            cams_p = fit_joint(fs, frames, cams_p)
            for c in twin.CAMS:
                iou_f, ious = full_res_iou(c, fs, frames, cams_p[c])
                save_fit(c, cams_p[c], float(ious.mean()), iou_f, ious, frames)
                print(f"joint round {r} [{c}] IoU 640x480 {iou_f:.3f} params {np.round(cams_p[c], 3)}", flush=True)
            if r < a.rounds - 1:
                fit_offsets(fs, frames, cams_p)
                twin.JOINTS_JSON.write_text(json.dumps(dict(signs=twin.SIGNS[:5].tolist(), offsets_deg=twin.OFFSETS_DEG[:5].tolist(),
                                                            scales=twin.SCALES[:5].tolist()), indent=1))
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
            twin.JOINTS_JSON.write_text(json.dumps(dict(signs=twin.SIGNS[:5].tolist(), offsets_deg=twin.OFFSETS_DEG[:5].tolist(),
                                                    scales=twin.SCALES[:5].tolist()), indent=1))
        print("wrote", twin.JOINTS_JSON)


if __name__ == "__main__":
    main()
