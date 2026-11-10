"""Fit the twin's camera poses (and optionally joint offsets) to the real frames by white-silhouette IoU.

    .venv/bin/python so101_twin/ringpole/fit_cameras.py --cam top            # -> out/camera_fit_top.json, out/fit_top.png
    .venv/bin/python so101_twin/ringpole/fit_cameras.py --cam side           # shoulder camera (pan-frame parameters)
    .venv/bin/python so101_twin/ringpole/fit_cameras.py --offsets            # alternate joint-offset / camera refits
    .venv/bin/python so101_twin/ringpole/fit_cameras.py --apply              # write the fitted CAMS into build_scene.py

Real mask: "white" pixels by HSV threshold (the arm body, the pole, the pale purple pole base; the white balance
varies a lot between episodes, so the saturation threshold for the overhead camera is loose). Sim mask:
segmentation render of the arm's white visual geoms + pole (+ base for the top camera). Objective: 1 - mean IoU
over N_FRAMES frames chosen by farthest-point sampling in joint space among frames whose real mask is sane.
The side camera (`observation.images.sholder`) is attached to the pan frame, so its parameters are an offset /
orientation relative to the pan axis and the fit uses frames at different pan angles (untinted episodes only,
where the pole is reliably white). Search: random box search around build_scene.CAMS, then Nelder-Mead from the
best few, at 320x240; the reported IoU is at full 640x480. Objects are placed per episode (objects.json from
place_objects.py; default layout if missing).
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
OBJECTS_JSON = HERE / "objects.json"
# search box half-widths around the initial guess (x y z m, yaw pitch roll deg, fovy deg)
SPAN = {"top": np.array([0.25, 0.25, 0.30, 25, 15, 8, 20]), "side": np.array([0.10, 0.08, 0.08, 35, 25, 12, 25])}
STEP = {"top": np.array([0.03, 0.03, 0.03, 3, 3, 1.5, 3]), "side": np.array([0.015, 0.015, 0.015, 3, 3, 1.5, 3])}
UNTINTED_MAX_FRAME = 192       # episodes 0..23 of the set have normal white balance (the rest are orange-tinted)


# ---- real masks ---------------------------------------------------------------------------------------------
def real_white_mask(rgb: np.ndarray, cam: str) -> np.ndarray:
    hsv = np.asarray(Image.fromarray(np.asarray(rgb)).convert("HSV")).astype(int)
    s, v = hsv[..., 1], hsv[..., 2]
    if cam == "top":
        m = (s < 110) & (v > 150)
        m[:25] = False                      # beige wall strip at the far table edge
    else:
        m = (s < 70) & (v > 150)
    return m


def downsample(mask: np.ndarray, k: int = 2) -> np.ndarray:
    H, W = mask.shape
    return mask.reshape(H // k, k, W // k, k).mean((1, 3)) > 0.5


def load_objects() -> dict | None:
    if OBJECTS_JSON.exists():
        return json.loads(OBJECTS_JSON.read_text())
    return None


def episode_layout(objects: dict | None, episode: int) -> dict:
    """{'pole': (x, y), 'ring': (x, y), 'color': str} for a dataset episode index."""
    if objects is not None and str(int(episode)) in objects:
        return objects[str(int(episode))]
    return dict(pole=list(build_scene.POLE["xy"]), ring=list(build_scene.RING["xy"]), color="orange")


def pose_frame(sim: Sim, fs, objects: dict | None, i: int, ring_in_hand: bool = False) -> None:
    """Joints from the real state, pole/ring placed for the frame's episode (ring on the table)."""
    sim.set_joints(fs.state[i])
    lay = episode_layout(objects, fs.labels["episode"][i])
    sim.set_mocap("pole", [lay["pole"][0], lay["pole"][1], 0.0])
    sim.set_mocap("ring", [lay["ring"][0], lay["ring"][1], build_scene.RING["h"] / 2])
    import mujoco

    mujoco.mj_forward(sim.m, sim.d)


def candidate_frames(fs, cam: str) -> np.ndarray:
    """Frames whose real mask is usable for this camera."""
    ok = []
    for i in range(fs.n):
        m = real_white_mask(fs.obs(i)["images"][cam], cam)
        area = m.mean()
        if cam == "top":
            ok.append(area < 0.20)                       # drop frames with big table reflections
        else:
            # untinted episodes, pole-dominated views (the arm close-ups are blurred / overexposed and mask badly)
            ok.append(i < UNTINTED_MAX_FRAME and 0.01 < area < 0.09 and fs.state[i][0] < 45)
    return np.nonzero(ok)[0]


def pick_frames(fs, cam: str, n: int = N_FRAMES) -> list[int]:
    """Farthest-point sampling over the five revolute joints (normalised) among the candidate frames."""
    cand = candidate_frames(fs, cam)
    st = fs.state[cand, :5].astype(float)
    z = (st - st.mean(0)) / (st.std(0) + 1e-6)
    if cam == "side":
        z[:, 0] *= 3.0                                   # pan matters most for the shoulder camera
    chosen = [int(np.argmin(st[:, 1]))]                  # smallest shoulder_lift = arm reaching out
    d = np.linalg.norm(z - z[chosen[0]], axis=1)
    while len(chosen) < n:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(z - z[i], axis=1))
    return [int(cand[i]) for i in chosen]


# ---- sim side -----------------------------------------------------------------------------------------------
class Fitter:
    def __init__(self, cam: str, frames: list[int], fs, small=True):
        self.cam = cam
        self.frames = frames
        self.fs = fs
        self.k = 2 if small else 1
        self.sim = Sim(w=640 // self.k, h=480 // self.k)
        self.objects = load_objects()
        self.real = [downsample(real_white_mask(fs.obs(i)["images"][cam], cam), self.k) if small else
                     real_white_mask(fs.obs(i)["images"][cam], cam) for i in frames]
        self.dil = 6 // self.k
        if cam == "side":
            self.real = [ndimage.binary_dilation(m, iterations=self.dil) for m in self.real]
        self.n_eval = 0
        base_id = self.sim.m.geom("pole_base").id
        self.sim_ids = np.concatenate([self.sim.arm_geoms, self.sim.pole_geoms, [base_id] if cam == "top" else []])

    def sim_mask(self) -> np.ndarray:
        return np.isin(self.sim.seg(self.cam), self.sim_ids)

    def ious(self, p: np.ndarray) -> np.ndarray:
        self.sim.set_camera(self.cam, p[:3], p[3], p[4], p[5], p[6])
        out = []
        for i, rm in zip(self.frames, self.real):
            pose_frame(self.sim, self.fs, self.objects, i)
            sm = self.sim_mask()
            if self.cam == "side":          # thin pole: soften both masks so the objective is not knife-edged
                sm = ndimage.binary_dilation(sm, iterations=self.dil)
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
            pose_frame(self.sim, self.fs, self.objects, i)
            sm = self.sim_mask()
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
    span = SPAN[cam]
    t0 = time.time()
    cands = [(f.loss(p0), p0)]
    for _ in range(n_random):
        p = p0 + rng.uniform(-1, 1, 7) * span
        p[6] = np.clip(p[6], 20, 100)
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
    f.overlay(p, OUT / f"fit_{cam}.png")
    f.sim.close()
    return float(ious.mean()), ious


def save_fit(cam: str, p, iou_small, iou_full, ious, frames) -> None:
    d = dict(cam=cam, frame=("world" if cam == "top" else "pan_frame"), params=dict(zip(KEYS, map(float, p))),
             iou_320x240=float(iou_small), iou_640x480=float(iou_full),
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


def fit_offsets(fs, frames: dict[str, list[int]], cams_p: dict[str, np.ndarray], joints=(1, 2, 3)) -> np.ndarray:
    """Nelder-Mead over OFFSETS_DEG[joints] with the cameras fixed, objective = mean IoU over the fitted cams."""
    fits = {c: Fitter(c, frames[c], fs, small=True) for c in cams_p}
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
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--n-random", type=int, default=400)
    ap.add_argument("--n-refine", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-saved", action="store_true", help="start --cam fit from out/camera_fit_<cam>.json")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.apply:
        apply_to_build_scene()
        return
    fs = real_set()
    if a.cam:
        frames = pick_frames(fs, a.cam)
        print(f"[{a.cam}] frames: {frames}  pans: {np.round(fs.state[frames, 0]).astype(int).tolist()}", flush=True)
        p0 = load_fit(a.cam) if a.from_saved else None
        p, iou_s = fit_camera(a.cam, fs, frames, n_random=a.n_random, n_refine=a.n_refine, seed=a.seed, p0=p0)
        iou_f, ious = full_res_iou(a.cam, fs, frames, p)
        save_fit(a.cam, p, iou_s, iou_f, ious, frames)
        print(f"[{a.cam}] final params {dict(zip(KEYS, np.round(p, 4)))}\n[{a.cam}] IoU 320x240 {iou_s:.3f}, "
              f"640x480 {iou_f:.3f}, per frame {np.round(ious, 2)}", flush=True)
    if a.offsets:
        cams = [c for c in twin.CAMS if (OUT / f"camera_fit_{c}.json").exists()]
        frames = {c: pick_frames(fs, c) for c in cams}
        cams_p = {c: load_fit(c) for c in cams}
        for r in range(a.rounds):
            off = fit_offsets(fs, frames, cams_p)
            for c in cams:
                p, iou_s = fit_camera(c, fs, frames[c], n_random=60, n_refine=2, seed=a.seed + r, p0=cams_p[c])
                cams_p[c] = p
                iou_f, ious = full_res_iou(c, fs, frames[c], p)
                save_fit(c, p, iou_s, iou_f, ious, frames[c])
                print(f"round {r} [{c}] IoU 640x480 {iou_f:.3f} offsets {np.round(off, 2)}", flush=True)
        (OUT / "offsets_fit.json").write_text(json.dumps(dict(offsets_deg=twin.OFFSETS_DEG.tolist(), signs=twin.SIGNS.tolist())))


if __name__ == "__main__":
    main()
