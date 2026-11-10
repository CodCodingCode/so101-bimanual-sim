"""Fit the twin's camera poses (and optionally joint offsets) to the real frames by arm-silhouette IoU.

    .venv/bin/python so101_twin/sort/fit_cameras.py --cam top      # session A static cam -> out/camera_fit_top.json, out/fit_top.png
    .venv/bin/python so101_twin/sort/fit_cameras.py --cam top_b    # session B static cam
    .venv/bin/python so101_twin/sort/fit_cameras.py --cam side     # wrist cam on Fixed_Jaw (jaw silhouette)
    .venv/bin/python so101_twin/sort/fit_cameras.py --offsets      # alternate joint-offset / top-camera refits
    .venv/bin/python so101_twin/sort/fit_cameras.py --apply        # write the fitted CAMS into build_scene.py

Real arm mask (top cams): lilac/white arm body OR light-blue wrist+jaws by HSV threshold (the white arm reads as
lilac under the room light; the white table is lower-saturation). Sim mask: segmentation render of the arm's
non-motor visual geoms (black motors are holes in both). Wrist cam: real mask = dark pixels (the black finger
pads) on frames that mostly show the bright table; sim mask = Fixed_Jaw + Moving_Jaw geoms.
Objective: 1 - mean IoU over N_FRAMES frames chosen by farthest-point sampling in joint space within the
camera's session. Search: random box search around the initial guess in build_scene.CAMS, then Nelder-Mead from
the best few, at 320x240; the reported IoU is at full 640x480.
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
SUFFIX = ""      # --tag appends to the output file names (alternative starting points)
SPAN = {"top": np.array([0.25, 0.25, 0.25, 25, 20, 6, 18]),
        "top_b": np.array([0.25, 0.25, 0.25, 25, 20, 6, 18]),
        "side": np.array([0.015, 0.02, 0.015, 12, 12, 10, 12])}
STEP = {"top": np.array([0.03, 0.03, 0.03, 3, 3, 1.5, 3]),
        "top_b": np.array([0.03, 0.03, 0.03, 3, 3, 1.5, 3]),
        "side": np.array([0.01, 0.01, 0.01, 4, 4, 4, 3])}


# ---- real masks ---------------------------------------------------------------------------------------------
ROI_ROWS = 480     # wrist cam: rows entering the IoU (480 = whole image; the real view is mostly bright table)


def real_arm_mask(rgb: np.ndarray, cam: str = "top") -> np.ndarray:
    """Light-blue wrist + gripper OR the white arm body, which reads lilac (session A, bluish light) / lavender
    (session B, greyer light) against the lower-saturation white table. Approximate: parts of the forearm and some
    table highlights / cube tops leak either way."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (h >= 118) & (h <= 152) & (s > 150) & (v > 60)
    if cam == "top_b":
        body = (h >= 160) & (h <= 205) & (s >= 30) & (s <= 120) & (v >= 85) & (v <= 215)
    else:
        body = (h >= 140) & (h <= 176) & (s >= 58) & (s <= 150) & (v >= 95)
    return blue | body


def real_jaw_mask(rgb: np.ndarray) -> np.ndarray:
    m = rgb.max(-1) < 70
    m[ROI_ROWS:] = False
    return m


def real_mask(cam: str, rgb: np.ndarray) -> np.ndarray:
    return real_jaw_mask(rgb) if cam == "side" else real_arm_mask(rgb, cam)


def downsample(mask: np.ndarray, k: int = 2) -> np.ndarray:
    H, W = mask.shape
    return mask.reshape(H // k, k, W // k, k).mean((1, 3)) > 0.5


def session_frames(fs, cam: str) -> np.ndarray:
    eps = np.asarray(fs.labels["episode"])
    if cam == "side":
        return np.arange(fs.n)
    return np.nonzero([build_scene.top_cam_for_episode(e) == cam for e in eps])[0]


def pick_frames(fs, cam: str, n: int = N_FRAMES) -> list[int]:
    """Farthest-point sampling over the joints (normalised) within the camera's candidate frames."""
    cand = session_frames(fs, cam)
    if cam == "side":
        # frames that mostly show the bright table: mean > 140 and dark fraction < 0.25
        ok = []
        for i in cand:
            im = fs.obs(int(i))["images"]["side"]
            dark = real_jaw_mask(im).mean()
            if im[200:].mean() > 150 and 0.02 < dark < 0.20:
                ok.append(i)
        cand = np.asarray(ok)
        cols = [5, 4, 3]                               # gripper opening, roll, wrist flex matter most
    else:
        cols = [0, 1, 2, 3, 4]
    st = fs.state[cand][:, cols].astype(float)
    z = (st - st.mean(0)) / (st.std(0) + 1e-6)
    if cam == "side":
        chosen = [int(np.argmax(st[:, 0]))]           # most open gripper
    else:
        chosen = [int(np.argmax(fs.state[cand][:, 1]))]   # largest shoulder_lift = arm reaching out
    d = np.linalg.norm(z - z[chosen[0]], axis=1)
    while len(chosen) < min(n, len(cand)):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(z - z[i], axis=1))
    return [int(cand[i]) for i in chosen]


# ---- sim side -----------------------------------------------------------------------------------------------
class Fitter:
    def __init__(self, cam: str, frames: list[int], fs, small=True):
        self.cam = cam
        self.real_cam = "side" if cam == "side" else "top"
        self.frames = frames
        self.fs = fs
        self.k = 2 if small else 1
        self.sim = Sim(w=640 // self.k, h=480 // self.k)
        self.real = [downsample(real_mask(cam, fs.obs(i)["images"][self.real_cam]), self.k) if small else
                     real_mask(cam, fs.obs(i)["images"][self.real_cam]) for i in frames]
        self.states = [fs.state[i] for i in frames]
        self.n_eval = 0

    def sim_mask(self) -> np.ndarray:
        if self.cam == "side":
            m = self.sim.robot_mask("side")
            m[ROI_ROWS // self.k:] = False
            return m
        return self.sim.arm_mask(self.cam)

    def ious(self, p: np.ndarray) -> np.ndarray:
        self.sim.set_camera(self.cam, p[:3], p[3], p[4], p[5], p[6])
        out = []
        for st, rm in zip(self.states, self.real):
            self.sim.set_joints(st)
            sm = self.sim_mask()
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
            sm = self.sim_mask()
            rm = self.real[j]
            ov = np.zeros(sm.shape + (3,), np.uint8)
            ov[rm] = (200, 60, 60)
            ov[sm] = (60, 60, 220)
            ov[rm & sm] = (240, 240, 240)
            real = np.asarray(Image.fromarray(self.fs.obs(i)["images"][self.real_cam]).resize(sm.shape[::-1]))
            tiles.append(np.concatenate([real, self.sim.render(self.cam), ov], 1))
        Image.fromarray(np.concatenate(tiles, 0)).save(path)


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
        p[6] = np.clip(p[6], 25, 110)
        cands.append((f.loss(p), p))
    cands.sort(key=lambda t: t[0])
    print(f"[{cam}] random search: best loss {cands[0][0]:.3f} (IoU {1 - cands[0][0]:.3f}) after {time.time() - t0:.0f}s",
          flush=True)
    best = cands[0]
    step = STEP[cam]
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
    f.overlay(p, OUT / f"fit_{cam}{SUFFIX}.png")
    f.sim.close()
    return float(ious.mean()), ious


def save_fit(cam: str, p, iou_small, iou_full, ious, frames) -> None:
    d = dict(cam=cam, params=dict(zip(KEYS, map(float, p))), iou_320x240=float(iou_small), iou_640x480=float(iou_full),
             iou_per_frame=[float(x) for x in ious], frames=[int(i) for i in frames],
             signs=twin.SIGNS.tolist(), offsets_deg=twin.OFFSETS_DEG.tolist())
    (OUT / f"camera_fit_{cam}{SUFFIX}.json").write_text(json.dumps(d, indent=1))


def load_fit(cam: str) -> np.ndarray:
    d = json.loads((OUT / f"camera_fit_{cam}.json").read_text())
    return np.array([d["params"][k] for k in KEYS])


def apply_to_build_scene() -> None:
    """Rewrite the CAMS lines of build_scene.py with the fitted parameters (cams with a fit file)."""
    p = HERE / "build_scene.py"
    s = p.read_text()
    for cam in twin.SIM_CAMS:
        if not (OUT / f"camera_fit_{cam}.json").exists():
            continue
        v = load_fit(cam)
        line = (f'    "{cam}": dict(pos=({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}), yaw={v[3]:.2f}, pitch={v[4]:.2f}, '
                f'roll={v[5]:.2f}, fovy={v[6]:.2f}),')
        s, n = re.subn(rf'^    "{cam}": dict\(pos=.*\),$', line, s, flags=re.M)
        assert n == 1, cam
    p.write_text(s)
    import importlib

    importlib.reload(build_scene).write()
    print("updated build_scene.py CAMS and rebuilt the XML")


def apply_offsets_to_twin() -> None:
    p = HERE / "twin.py"
    s = p.read_text()
    vals = ", ".join(f"{v:.2f}" for v in twin.OFFSETS_DEG)
    s, n = re.subn(r"^OFFSETS_DEG = np\.array\(\[.*?\]\)", f"OFFSETS_DEG = np.array([{vals}])", s, flags=re.M)
    assert n == 1
    p.write_text(s)


def fit_offsets(fs, frames: dict, cams_p: dict[str, np.ndarray], joints=(1, 2, 3, 4)) -> np.ndarray:
    """Nelder-Mead over OFFSETS_DEG[joints] with the top cameras fixed, objective = mean IoU over both top cams."""
    cams = [c for c in cams_p if c != "side"]
    fits = {c: Fitter(c, frames[c], fs, small=True) for c in cams}
    base = twin.OFFSETS_DEG.copy()

    def loss(x):
        off = base.copy()
        off[list(joints)] = x
        twin.OFFSETS_DEG[:] = off
        return 1 - np.mean([fits[c].ious(cams_p[c]).mean() for c in cams])

    x0 = base[list(joints)]
    res = minimize(loss, x0, method="Nelder-Mead",
                   options=dict(initial_simplex=np.stack([x0] + [x0 + 4 * np.eye(len(joints))[i] for i in range(len(joints))]),
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
    ap.add_argument("--cam", choices=twin.SIM_CAMS)
    ap.add_argument("--offsets", action="store_true", help="alternate offset / camera refits starting from the saved fits")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--n-random", type=int, default=400)
    ap.add_argument("--n-refine", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--p0", type=float, nargs=7, default=None, help="override the initial guess x y z yaw pitch roll fovy")
    a = ap.parse_args()
    global SUFFIX
    SUFFIX = a.tag
    OUT.mkdir(exist_ok=True)
    if a.apply:
        apply_to_build_scene()
        return
    fs = real_set()
    if a.cam:
        frames = pick_frames(fs, a.cam)
        print(f"[{a.cam}] frames:", frames, flush=True)
        p, iou_s = fit_camera(a.cam, fs, frames, n_random=a.n_random, n_refine=a.n_refine, seed=a.seed, p0=a.p0)
        iou_f, ious = full_res_iou(a.cam, fs, frames, p)
        save_fit(a.cam, p, iou_s, iou_f, ious, frames)
        print(f"[{a.cam}] final params {dict(zip(KEYS, np.round(p, 4)))}\n[{a.cam}] IoU 320x240 {iou_s:.3f}, "
              f"640x480 {iou_f:.3f}, per frame {np.round(ious, 2)}", flush=True)
    if a.offsets:
        cams = ("top", "top_b")
        cams_p = {c: load_fit(c) for c in cams}
        frames = {c: pick_frames(fs, c) for c in cams}
        for r in range(a.rounds):
            off = fit_offsets(fs, frames, cams_p)
            for c in cams:
                p, iou_s = fit_camera(c, fs, frames[c], n_random=60, n_refine=2, seed=a.seed + r, p0=cams_p[c])
                cams_p[c] = p
                iou_f, ious = full_res_iou(c, fs, frames[c], p)
                save_fit(c, p, iou_s, iou_f, ious, frames[c])
                print(f"round {r} [{c}] IoU 640x480 {iou_f:.3f} offsets {np.round(off, 2)}", flush=True)
        (OUT / "offsets_fit.json").write_text(json.dumps(dict(offsets_deg=twin.OFFSETS_DEG.tolist(), signs=twin.SIGNS.tolist())))
        apply_offsets_to_twin()
        print("wrote OFFSETS_DEG into twin.py")


if __name__ == "__main__":
    main()
