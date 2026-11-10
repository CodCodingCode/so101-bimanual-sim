"""Fit the twin's camera poses (and optionally joint offsets) to the real frames by arm-silhouette IoU.

    .venv/bin/python so101_twin/legocup/fit_cameras.py --roi                # -> out/mat_roi.npy (top-camera region of interest)
    .venv/bin/python so101_twin/legocup/fit_cameras.py --cam top            # -> out/camera_fit_top.json, out/fit_top.png
    .venv/bin/python so101_twin/legocup/fit_cameras.py --cam side           # wrist camera: pose in the Fixed_Jaw frame
    .venv/bin/python so101_twin/legocup/fit_cameras.py --offsets            # joint-offset / top-camera refits
    .venv/bin/python so101_twin/legocup/fit_cameras.py --apply              # write the fitted CAMS into build_scene.py

Real arm mask ("top"): the arm is black, so dark pixels (max RGB < 90) inside the grey-mat ROI whose connected
component touches the top image border (the arm always enters from the top; the black cup is a separate
component unless the arm is over it). Real mask ("side" = wrist camera): dark pixels whose component touches the
top border = the two jaw prongs. Sim mask: segmentation render of the arm's visual geoms (restricted to the ROI
for "top"). Objective: 1 - mean IoU over N_FRAMES real frames chosen by farthest-point sampling in joint space
(for the wrist camera only frames whose real mask is small, i.e. the jaws over the mat, not over the cup).
Search: random box search (around the initial guess for "top"; over a wide box in the jaw frame for "side"), then
Nelder-Mead from the best few, at 320x240; the reported IoU is at full 640x480.
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
from twin import HERE, ROI_NPY, Sim, real_set  # noqa: E402

N_FRAMES = 18
KEYS = ("x", "y", "z", "yaw", "pitch", "roll", "fovy")
KEYS_SIDE = KEYS + ("jaw_gain",)      # the wrist-camera fit also fits the gripper % -> Jaw angle gain
OUT = HERE / "out"
DARK_THR = {"top": 90, "side": 60}
TOP_ROWS = 12


# ---- real masks ---------------------------------------------------------------------------------------------
def build_roi(fs, save: bool = True) -> np.ndarray:
    """Top-camera ROI = the grey mat: per-pixel 60th percentile of max(RGB) over the set > 95 (the arm covers any
    pixel in well under half of the frames), largest component, holes filled, lightly eroded. The strip of rows
    above the mat is kept only in the mat's columns (so the arm's entry from the top counts, not the dark table)."""
    stack = np.stack([fs.obs(i)["images"]["top"].max(-1) for i in range(0, fs.n, 3)])
    p = np.percentile(stack, 60, axis=0)
    roi = p > 95
    lab, n = ndimage.label(roi)
    sizes = ndimage.sum(roi, lab, range(1, n + 1))
    roi = ndimage.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
    roi = ndimage.binary_opening(roi, iterations=4)
    roi = ndimage.binary_erosion(roi, iterations=6)
    cols = roi[TOP_ROWS + 4]
    roi[:TOP_ROWS + 4, :] = cols[None, :]
    if save:
        OUT.mkdir(exist_ok=True)
        np.save(ROI_NPY, roi)
        Image.fromarray((roi * 255).astype(np.uint8)).save(OUT / "mat_roi.png")
    return roi


_ROI = None


def roi() -> np.ndarray:
    global _ROI
    if _ROI is None:
        _ROI = np.load(ROI_NPY)
    return _ROI


_OBJECTS = None
CUP_R_PX = 95


def frame_roi(cam: str, ep: int | None) -> np.ndarray:
    """ROI for one frame: the mat, minus a disc around the black cup (per-episode pixel centroid from
    out/objects.json, if present) so the cup never merges with the arm's dark mask."""
    global _OBJECTS
    if cam != "top":
        return None
    r = roi()
    if ep is None:
        return r
    if _OBJECTS is None:
        pj = OUT / "objects.json"
        _OBJECTS = json.loads(pj.read_text()) if pj.exists() else {}
    o = _OBJECTS.get(str(int(ep)))
    if not o or "cup_px" not in o:
        return r
    u0, v0 = o["cup_px"]
    vv, uu = np.mgrid[:r.shape[0], :r.shape[1]]
    return r & ((uu - u0) ** 2 + (vv - v0) ** 2 > CUP_R_PX ** 2)


def real_arm_mask(rgb: np.ndarray, cam: str, ep: int | None = None) -> np.ndarray:
    dark = rgb.max(-1) < DARK_THR[cam]
    if cam == "top":
        dark &= frame_roi(cam, ep)
    lab, _ = ndimage.label(dark)
    touch = set(np.unique(lab[:TOP_ROWS, :])) - {0}
    m = np.isin(lab, list(touch)) if touch else np.zeros_like(dark)
    return ndimage.binary_fill_holes(m)


def downsample(mask: np.ndarray, k: int = 2) -> np.ndarray:
    if k == 1:
        return mask
    H, W = mask.shape
    return mask.reshape(H // k, k, W // k, k).mean((1, 3)) > 0.5


def pick_frames(fs, cam: str, n: int = N_FRAMES) -> list[int]:
    """Farthest-point sampling over the five revolute joints (normalised), seeded with the most extended pose.
    Top camera: only approach/grasp frames with the arm covering > 2% of the image (see below). Wrist camera: only
    frames whose real mask covers < 12% of the image (jaws over the mat, not over the cup)."""
    st = fs.state[:, :5].astype(float)
    z = (st - st.mean(0)) / (st.std(0) + 1e-6)
    frac = np.array([real_arm_mask(fs.obs(i)["images"][cam], cam, int(fs.labels["episode"][i])).mean() for i in range(fs.n)])
    if cam == "side":
        z = np.concatenate([z, (fs.state[:, 5:6] - 20) / 15.0], 1)   # also spread the gripper opening
        ok = frac < 0.12
    else:
        # the arm must actually be in the picture (the base and upper arm are above the top edge), and it must not
        # be over the black cup (same dark mask): approach / grasp frames only
        ok = frac > 0.02
        if not (OUT / "objects.json").exists():          # no per-episode cup position yet: avoid the cup
            ok &= np.isin(fs.labels["phase"], ["approach", "grasp"])
    cand = np.nonzero(ok)[0]
    first = cand[int(np.argmin(st[cand, 1]))]
    chosen = [int(first)]
    d = np.linalg.norm(z - z[chosen[0]], axis=1)
    d[~ok] = -1
    while len(chosen) < n:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(z - z[i], axis=1))
        d[~ok] = -1
    return chosen


# ---- sim side -----------------------------------------------------------------------------------------------
class Fitter:
    def __init__(self, cam: str, frames: list[int], fs, small=True):
        self.cam = cam
        self.frames = frames
        self.fs = fs
        self.k = 2 if small else 1
        self.sim = Sim(w=640 // self.k, h=480 // self.k)
        eps = [int(fs.labels["episode"][i]) for i in frames]
        self.real = [downsample(real_arm_mask(fs.obs(i)["images"][cam], cam, e), self.k) for i, e in zip(frames, eps)]
        self.rois = [downsample(frame_roi(cam, e), self.k) for e in eps] if cam == "top" else None
        self.states = [fs.state[i] for i in frames]
        self.n_eval = 0

    def sim_mask(self, j: int) -> np.ndarray:
        sm = self.sim.arm_mask(self.cam)
        return sm & self.rois[j] if self.rois is not None else sm

    def ious(self, p: np.ndarray) -> np.ndarray:
        self.sim.set_camera(self.cam, p[:3], p[3], p[4], p[5], p[6])
        if len(p) > 7:
            twin.JAW_GAIN = float(p[7])
        out = []
        for j, (st, rm) in enumerate(zip(self.states, self.real)):
            self.sim.set_joints(st)
            sm = self.sim_mask(j)
            inter = np.logical_and(sm, rm).sum()
            union = np.logical_or(sm, rm).sum()
            out.append(inter / max(union, 1))
        self.n_eval += 1
        return np.array(out)

    def loss(self, p: np.ndarray) -> float:
        return 1.0 - float(self.ious(p).mean())

    def overlay(self, p: np.ndarray, path: pathlib.Path, n: int = 8) -> None:
        self.sim.set_camera(self.cam, p[:3], p[3], p[4], p[5], p[6])
        if len(p) > 7:
            twin.JAW_GAIN = float(p[7])
        tiles = []
        for j in np.linspace(0, len(self.frames) - 1, n).astype(int):
            i = self.frames[j]
            self.sim.set_joints(self.states[j])
            sm = self.sim_mask(j)
            rm = self.real[j]
            ov = np.zeros(sm.shape + (3,), np.uint8)
            ov[rm] = (200, 60, 60)
            ov[sm] = (60, 60, 220)
            ov[rm & sm] = (240, 240, 240)
            real = np.asarray(Image.fromarray(self.fs.obs(i)["images"][self.cam]).resize(sm.shape[::-1]))
            tiles.append(np.concatenate([real, self.sim.render(self.cam), ov], 1))
        Image.fromarray(np.concatenate(tiles, 0)).save(path)


def keys(cam: str) -> tuple[str, ...]:
    return KEYS_SIDE if cam == "side" else KEYS


def p0_from_cams(cam: str) -> np.ndarray:
    c = build_scene.CAMS[cam]
    p = [*c["pos"], c["yaw"], c["pitch"], c["roll"], c["fovy"]]
    return np.array(p + ([twin.JAW_GAIN] if cam == "side" else []), float)


# random-search region: p0 +- SPAN ("side" = wrist camera, pos/angles in the Fixed_Jaw frame; --wide = uniform in BOX)
SPAN = {"top": np.array([0.30, 0.30, 0.30, 30, 25, 8, 20]), "side": np.array([0.03, 0.03, 0.03, 20, 15, 20, 10, 0.4])}
BOX = {"side": (np.array([-0.09, -0.05, -0.09, -180, -70, -180, 45, 0.5]), np.array([0.09, 0.09, 0.09, 180, 70, 180, 85, 2.0]))}
FOVY_LIM = {"top": (25, 100), "side": (45, 85)}


def sample(cam: str, p0: np.ndarray, rng, wide: bool) -> np.ndarray:
    if wide and cam in BOX:
        lo, hi = BOX[cam]
        return lo + rng.uniform(0, 1, len(lo)) * (hi - lo)
    p = p0 + rng.uniform(-1, 1, len(p0)) * SPAN[cam]
    p[6] = np.clip(p[6], *FOVY_LIM[cam])
    if len(p) > 7:
        p[7] = np.clip(p[7], 0.4, 2.0)
    return p


def fit_camera(cam: str, fs, frames, n_random=400, n_refine=4, seed=0, p0=None, wide=None) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    f = Fitter(cam, frames, fs, small=True)
    wide = False if wide is None else wide
    p0 = p0_from_cams(cam) if p0 is None else np.asarray(p0, float)
    if cam == "side" and len(p0) == 7:
        p0 = np.append(p0, twin.JAW_GAIN)
    t0 = time.time()
    cands = [(f.loss(p0), p0)]
    for _ in range(n_random):
        p = sample(cam, p0, rng, wide)
        cands.append((f.loss(p), p))
    cands.sort(key=lambda t: t[0])
    print(f"[{cam}] random search ({'box' if wide else 'span'}, {n_random}): best loss {cands[0][0]:.3f} "
          f"(IoU {1 - cands[0][0]:.3f}) after {time.time() - t0:.0f}s", flush=True)
    best = cands[0]
    step = np.array([0.03, 0.03, 0.03, 3, 3, 1.5, 3]) if cam == "top" else np.array([0.01, 0.01, 0.01, 4, 4, 4, 3, 0.1])
    k = len(p0)
    for L, p in cands[:n_refine]:
        simplex = np.stack([p] + [p + np.eye(k)[i] * step[i] for i in range(k)])
        res = minimize(f.loss, p, method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, xatol=1e-3, fatol=1e-4, maxiter=700, maxfev=900))
        print(f"[{cam}] NM from {L:.3f} -> {res.fun:.3f} (IoU {1 - res.fun:.3f}), {res.nfev} evals, "
              f"{time.time() - t0:.0f}s", flush=True)
        if res.fun < best[0]:
            best = (res.fun, res.x)
    f.sim.close()
    return best[1], 1 - best[0]


def full_res_iou(cam: str, fs, frames, p, tag: str = "") -> tuple[float, np.ndarray]:
    f = Fitter(cam, frames, fs, small=False)
    ious = f.ious(p)
    f.overlay(p, OUT / f"fit_{cam}{tag}.png")
    f.sim.close()
    return float(ious.mean()), ious


def save_fit(cam: str, p, iou_small, iou_full, ious, frames, tag: str = "") -> None:
    d = dict(cam=cam, parent=build_scene.CAM_PARENT[cam], params=dict(zip(keys(cam), map(float, p))),
             iou_320x240=float(iou_small), iou_640x480=float(iou_full),
             iou_per_frame=[float(x) for x in ious], frames=[int(i) for i in frames],
             signs=twin.SIGNS.tolist(), offsets_deg=twin.OFFSETS_DEG.tolist())
    (OUT / f"camera_fit_{cam}{tag}.json").write_text(json.dumps(d, indent=1))


def load_fit(cam: str, tag: str = "") -> np.ndarray:
    d = json.loads((OUT / f"camera_fit_{cam}{tag}.json").read_text())
    return np.array([d["params"][k] for k in keys(cam) if k in d["params"]])


def apply_to_build_scene() -> None:
    """Rewrite the CAMS lines of build_scene.py with the fitted parameters (for the cams that have a fit)."""
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


def fit_offsets(fs, frames, cams_p: dict[str, np.ndarray], joints=(0, 1, 2, 3), cams=("top",)) -> np.ndarray:
    """Nelder-Mead over OFFSETS_DEG[joints] with the camera(s) fixed, objective = mean IoU over `cams`
    (the wrist camera is rigidly attached to the jaw, so only the top camera constrains the arm joints)."""
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
    ap.add_argument("--roi", action="store_true", help="(re)build the top-camera mat ROI")
    ap.add_argument("--cam", choices=twin.CAMS)
    ap.add_argument("--offsets", action="store_true", help="alternate offset / top-camera refits starting from the saved fits")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--n-random", type=int, default=400)
    ap.add_argument("--n-refine", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-fit", action="store_true", help="start --cam from out/camera_fit_<cam>.json (local search)")
    ap.add_argument("--p0", type=str, help="comma-separated start x,y,z,yaw,pitch,roll,fovy[,jaw_gain] (local search)")
    ap.add_argument("--wide", action="store_true", help="side: uniform random search over BOX instead of p0 +- SPAN")
    ap.add_argument("--tag", type=str, default="", help="suffix for out/camera_fit_<cam><tag>.json and fit_<cam><tag>.png")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.apply:
        apply_to_build_scene()
        return
    fs = real_set()
    if a.roi:
        r = build_roi(fs)
        print(f"ROI: {r.mean():.3f} of the image -> {ROI_NPY}")
    if (OUT / "offsets_fit.json").exists():
        twin.OFFSETS_DEG[:] = json.loads((OUT / "offsets_fit.json").read_text())["offsets_deg"]
    if a.cam:
        frames = pick_frames(fs, a.cam)
        print(f"[{a.cam}] frames:", frames, flush=True)
        p0 = load_fit(a.cam) if a.from_fit else (np.array([float(v) for v in a.p0.split(",")]) if a.p0 else None)
        p, iou_s = fit_camera(a.cam, fs, frames, n_random=a.n_random, n_refine=a.n_refine, seed=a.seed, p0=p0,
                              wide=a.wide or None)
        iou_f, ious = full_res_iou(a.cam, fs, frames, p, a.tag)
        save_fit(a.cam, p, iou_s, iou_f, ious, frames, a.tag)
        print(f"[{a.cam}] final params {dict(zip(keys(a.cam), np.round(p, 4)))}\n[{a.cam}] IoU 320x240 {iou_s:.3f}, "
              f"640x480 {iou_f:.3f}, per frame {np.round(ious, 2)}", flush=True)
    if a.offsets:
        frames = {c: pick_frames(fs, c) for c in ("top",)}
        cams_p = {"top": load_fit("top")}
        for r in range(a.rounds):
            off = fit_offsets(fs, frames, cams_p)
            (OUT / "offsets_fit.json").write_text(json.dumps(dict(offsets_deg=twin.OFFSETS_DEG.tolist(), signs=twin.SIGNS.tolist())))
            p, iou_s = fit_camera("top", fs, frames["top"], n_random=60, n_refine=2, seed=a.seed + r, p0=cams_p["top"], wide=False)
            cams_p["top"] = p
            iou_f, ious = full_res_iou("top", fs, frames["top"], p)
            save_fit("top", p, iou_s, iou_f, ious, frames["top"])
            print(f"round {r} [top] IoU 640x480 {iou_f:.3f} offsets {np.round(off, 2)}", flush=True)


if __name__ == "__main__":
    main()
