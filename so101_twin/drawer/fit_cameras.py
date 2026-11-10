"""Fit the twin's camera poses (and optionally joint offsets) to the real frames by arm-silhouette IoU.

    .venv/bin/python so101_twin/drawer/fit_cameras.py --cam side           # -> out/camera_fit_side.json, out/fit_side.png
    .venv/bin/python so101_twin/drawer/fit_cameras.py --cam top            # wrist camera (pose in the Fixed_Jaw frame)
    .venv/bin/python so101_twin/drawer/fit_cameras.py --offsets            # alternate joint-offset / camera refits
    .venv/bin/python so101_twin/drawer/fit_cameras.py --apply              # write the fitted CAMS into build_scene.py

"side" (overhead): the real green arm is overexposed to near-white in the lit episodes, so the real mask is
"bright" (V > 140, the board glare stays below ~130) or shaded green / orange by hue, minus the purple / cyan blocks, with fixed zones zeroed in both masks
(drawer unit, leader-arm parts beside the base, clamp strip, laptop area); sim mask = segmentation render of all
printed arm parts (black motors are holes in both). Frames: farthest-point sampling in joint space among the
brightly lit episodes.
"top" (wrist camera on the gripper): real mask = white pixels in the lower part of the frame (the jaws); sim
mask = segmentation of the two jaw meshes, same row cut. Only frames whose upper part has no white (no drawer /
laptop in view) are used, spread over gripper openings. The fitted parameters are in the Fixed_Jaw body frame.
Search: random box search around the initial guess in build_scene.CAMS, then Nelder-Mead from the best few, at
320x240; the reported IoU is at full 640x480.
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
# overhead pixels (x0, x1, y0, y1) zeroed in both masks: orange leader-arm parts beside the base, the clamp/bracket
# strip at the bottom and the laptop/cable area on the far left
SIDE_EXCLUDE = [(222, 305, 292, 400), (30, 95, 285, 350), (0, 640, 395, 480), (0, 125, 0, 480), (180, 310, 0, 95),
                (345, 640, 0, 480)]   # the arm never reaches x > 345 px; the board's glare lives there
TOP_ROW0 = 200          # wrist camera: only rows >= TOP_ROW0 are compared (the jaws live there)
MASK_WHICH = {"side": "nomotor", "top": "jaws"}


def exclude_zone(mask: np.ndarray, k: int = 1) -> np.ndarray:
    for x0, x1, y0, y1 in SIDE_EXCLUDE:
        mask[y0 // k:y1 // k, x0 // k:x1 // k] = False
    return mask


def real_arm_mask(rgb: np.ndarray, cam: str) -> np.ndarray:
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if cam == "side":
        white = v > 140                                                 # overexposed green / white gripper
        green = (h >= 40) & (h <= 100) & (s > 60) & (v > 60)              # shaded green
        orange = (h >= 4) & (h <= 30) & (s > 130) & (v > 90)
        blocks = (((h >= 165) & (h <= 205)) | ((h >= 115) & (h <= 150))) & (s > 80)   # purple / cyan blocks
        return exclude_zone((white | green | orange) & ~blocks)
    white = (s < 55) & (v > 165)
    white[:TOP_ROW0] = False
    return white


def sim_arm_mask(sim: Sim, cam: str, k: int = 1) -> np.ndarray:
    m = sim.arm_mask(cam, MASK_WHICH[cam])
    if cam == "side":
        return exclude_zone(m, k)
    m[:TOP_ROW0 // k] = False
    return m


def downsample(mask: np.ndarray, k: int = 2) -> np.ndarray:
    H, W = mask.shape
    return mask.reshape(H // k, k, W // k, k).mean((1, 3)) > 0.5


def farthest_points(z: np.ndarray, cand: np.ndarray, first: int, n: int) -> list[int]:
    chosen = [int(first)]
    d = np.linalg.norm(z[cand] - z[first], axis=1)
    while len(chosen) < n:
        i = int(cand[np.argmax(d)])
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(z[cand] - z[i], axis=1))
    return chosen


def pick_frames(fs, cam: str = "side", n: int = N_FRAMES, seed: int = 0) -> list[int]:
    """side: farthest-point sampling over the five revolute joints among brightly lit frames, seeded with the
    most extended pose. top: frames without white above TOP_ROW0 (no drawer / laptop in the wrist view), spread
    over gripper opening and wrist angles."""
    st = fs.state.astype(float)
    if cam == "side":
        bright = np.array([fs.obs(i)["images"]["side"][:, 320:].mean() > 45 for i in range(fs.n)])
        z = (st[:, :5] - st[:, :5].mean(0)) / (st[:, :5].std(0) + 1e-6)
        cand = np.nonzero(bright)[0]
        return farthest_points(z, cand, cand[np.argmax(st[cand, 1])], n)   # largest lift = reaching out
    ok = []
    for i in range(fs.n):
        hsv = np.asarray(Image.fromarray(fs.obs(i)["images"]["top"]).convert("HSV")).astype(int)
        white_hi = ((hsv[..., 1] < 55) & (hsv[..., 2] > 165))[:TOP_ROW0].mean()
        white_lo = ((hsv[..., 1] < 55) & (hsv[..., 2] > 165))[TOP_ROW0:].mean()
        if white_hi < 0.01 and 0.1 < white_lo < 0.6:
            ok.append(i)
    cand = np.array(ok)
    z = st[:, [3, 5]] / (st[:, [3, 5]].std(0) + 1e-6)
    return farthest_points(z, cand, cand[np.argmax(st[cand, 5])], min(n, len(cand)))


# ---- sim side -----------------------------------------------------------------------------------------------
class Fitter:
    def __init__(self, cam: str, frames: list[int], fs, small=True):
        self.cam = cam
        self.frames = frames
        self.fs = fs
        self.k = 2 if small else 1
        self.sim = Sim(w=640 // self.k, h=480 // self.k)
        self.real = [downsample(real_arm_mask(fs.obs(i)["images"][cam], cam), self.k) if small else
                     real_arm_mask(fs.obs(i)["images"][cam], cam) for i in frames]
        self.states = [fs.state[i] for i in frames]
        self.n_eval = 0

    def ious(self, p: np.ndarray) -> np.ndarray:
        self.sim.set_camera(self.cam, p[:3], p[3], p[4], p[5], p[6])
        out = []
        for st, rm in zip(self.states, self.real):
            self.sim.set_joints(st)
            sm = sim_arm_mask(self.sim, self.cam, self.k)
            inter = np.logical_and(sm, rm).sum()
            union = np.logical_or(sm, rm).sum()
            out.append(inter / max(union, 1))
        self.n_eval += 1
        return np.array(out)

    def loss(self, p: np.ndarray) -> float:
        return 1.0 - float(self.ious(p).mean()) + anchor_penalty(self.cam, p)

    def overlay(self, p: np.ndarray, path: pathlib.Path, n: int = 8) -> None:
        self.sim.set_camera(self.cam, p[:3], p[3], p[4], p[5], p[6])
        tiles = []
        for j in np.linspace(0, len(self.frames) - 1, n).astype(int):
            i = self.frames[j]
            self.sim.set_joints(self.states[j])
            sm = sim_arm_mask(self.sim, self.cam, self.k)
            rm = self.real[j]
            ov = np.zeros(sm.shape + (3,), np.uint8)
            ov[rm] = (200, 60, 60)
            ov[sm] = (60, 60, 220)
            ov[rm & sm] = (240, 240, 240)
            real = np.asarray(Image.fromarray(self.fs.obs(i)["images"][self.cam]).resize(sm.shape[::-1]))
            tiles.append(np.concatenate([real, self.sim.render(self.cam), ov], 1))
        Image.fromarray(np.concatenate(tiles, 0)).save(path)


# Overhead anchors: world point -> expected pixel, from the real frames (pan axis at ~(190, 420); the breadboard's
# 25 mm hole pitch is ~24 px at table level, grid aligned with the image axes). Adds ANCHOR_W * mean px error / 100
# to the "side" loss so the table-plane geometry is pinned, not only the arm silhouette (which lives 5-25 cm up).
ANCHORS = {"side": [((0.0, 0.0, 0.02), (190.0, 420.0)), ((0.2, 0.0, 0.0), (382.0, 420.0)),
                    ((0.0, 0.2, 0.0), (190.0, 228.0))]}
ANCHOR_W = 0.0   # disabled: the pan axis is hidden under the folded arm in the real view, the guessed anchor pixel was wrong (see README)


def project(p: np.ndarray, X, w: int = 640, h: int = 480) -> np.ndarray:
    """Pixel (u, v) of world point X for camera params p = x y z yaw pitch roll fovy (world-frame camera)."""
    right, up = build_scene.cam_xyaxes(p[3], p[4], p[5])
    right, up = np.asarray(right), np.asarray(up)
    fwd = np.cross(up, right)
    d = np.asarray(X, float) - p[:3]
    xc, yc, zc = d @ right, d @ up, d @ fwd
    fy = (h / 2) / np.tan(np.radians(p[6]) / 2)
    return np.array([w / 2 + fy * xc / max(zc, 1e-6), h / 2 - fy * yc / max(zc, 1e-6)])


def anchor_penalty(cam: str, p: np.ndarray) -> float:
    if cam not in ANCHORS:
        return 0.0
    err = [np.linalg.norm(project(p, X) - np.asarray(uv)) for X, uv in ANCHORS[cam]]
    return ANCHOR_W * float(np.mean(err)) / 100.0


# random-search box half-widths per camera: x y z (m), yaw pitch roll fovy (deg)
SPAN = {"side": np.array([0.12, 0.12, 0.15, 12, 8, 6, 12]),
        "top": np.array([0.04, 0.04, 0.04, 35, 30, 25, 20])}


def p0_from_cams(cam: str) -> np.ndarray:
    c = build_scene.CAMS[cam]
    return np.array([*c["pos"], c["yaw"], c["pitch"], c["roll"], c["fovy"]], float)


def fit_camera(cam: str, fs, frames, n_random=400, n_refine=4, seed=0, p0=None) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    f = Fitter(cam, frames, fs, small=True)
    p0 = p0_from_cams(cam) if p0 is None else np.asarray(p0, float)
    span = SPAN[cam]
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
             anchor_px_err=[float(np.linalg.norm(project(np.asarray(p), X) - np.asarray(uv))) for X, uv in ANCHORS.get(cam, [])],
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


def fit_offsets(fs, frames, cams_p: dict[str, np.ndarray], joints=(0, 1, 2, 3)) -> np.ndarray:
    """Nelder-Mead over OFFSETS_DEG[joints] with the given cameras fixed, objective = mean IoU over those cams."""
    fits = {c: Fitter(c, frames, fs, small=True) for c in cams_p}
    base = twin.OFFSETS_DEG.copy()

    def loss(x):
        off = base.copy()
        off[list(joints)] = x
        twin.OFFSETS_DEG[:] = off
        return 1 - np.mean([fits[c].ious(cams_p[c]).mean() for c in cams_p])

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
    ap.add_argument("--from-saved", action="store_true", help="start --cam from out/camera_fit_<cam>.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.apply:
        apply_to_build_scene()
        return
    fs = real_set()
    if a.cam:
        frames = pick_frames(fs, a.cam)
        print("frames:", frames, flush=True)
        p, iou_s = fit_camera(a.cam, fs, frames, n_random=a.n_random, seed=a.seed,
                              p0=load_fit(a.cam) if a.from_saved else None)
        iou_f, ious = full_res_iou(a.cam, fs, frames, p)
        save_fit(a.cam, p, iou_s, iou_f, ious, frames)
        print(f"[{a.cam}] final params {dict(zip(KEYS, np.round(p, 4)))}\n[{a.cam}] IoU 320x240 {iou_s:.3f}, "
              f"640x480 {iou_f:.3f}, per frame {np.round(ious, 2)}", flush=True)
    if a.offsets:
        # the wrist camera's jaw silhouette does not depend on the arm joints, so offsets are fitted on "side" only
        frames = pick_frames(fs, "side")
        cams_p = {"side": load_fit("side")}
        for r in range(a.rounds):
            off = fit_offsets(fs, frames, cams_p)
            for c in ("side",):
                p, iou_s = fit_camera(c, fs, frames, n_random=60, n_refine=2, seed=a.seed + r, p0=cams_p[c])
                cams_p[c] = p
                iou_f, ious = full_res_iou(c, fs, frames, p)
                save_fit(c, p, iou_s, iou_f, ious, frames)
                print(f"round {r} [{c}] IoU 640x480 {iou_f:.3f} offsets {np.round(off, 2)}", flush=True)
        (OUT / "offsets_fit.json").write_text(json.dumps(dict(offsets_deg=twin.OFFSETS_DEG.tolist(), signs=twin.SIGNS.tolist())))


if __name__ == "__main__":
    main()
