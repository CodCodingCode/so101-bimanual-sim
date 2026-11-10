"""JAR copy of so101_twin/ballcup/fit.py (no functional change unless marked JAR), which is a ballcup COPY of so101_twin/legocup2/fit.py (a copy of common/fit.py; LEGOCUP2 changes: scene file <here>/scene_<name>.py, outputs in
legocup2/out, per-camera hide_groups / lens pad from the scene, gripper offset grip_off, --cams subset, finer lag grid).

Joint fit of cameras (pose per episode group, fovy, radial distortion, latency) and small joint offsets to SAM masks.

    .venv/bin/python so101_twin/common/fit.py lemon [--mask-tag _v2] [--split even] [--model so101|so100]
        [--out-tag _so101] [--evals 0.7] [--nfit 72] [--stages ABCD] [--from-json | --from-partial | --init-json F]
        [--eval-only] [--export-prompts prompts_twin2]

Official protocol (agreed with the parallel fitkit session): fit on EVEN episodes only (--split even, default), score
on ODD episodes (held-out) and also report all frames. Camera latency: each camera gets dt_frames (image time minus
state time, 30 fps frames), the arm is posed with the full-rate parquet state interpolated at t + dt (fullrate.py);
lemon fits dt ~ -2 for both cameras (the image shows the arm ~2 frames earlier than the recorded state).

Inputs: the scene description so101_twin/common/scenes/<scene>.py (see scenes/README in fit's docstring below and
scenes/lemon.py), the SAM masks so101_twin/masks/<scene>_<cam>.npy (segment.py) and the real states.
Model: the official SO-101 (arm.py) with the scene's calibration convention; the lens is a warp of the pinhole
render (arm.Lens: k1, k2, principal point cx, cy), so MuJoCo stays pinhole.

Stages (all at 320x240, IoU = |sim & real| / |sim | real| per frame, mean over fit frames where either is non-empty):
  A  per camera and episode group: pose (6) + fovy  -- random search (coarse) then CMA-ES
  B  per camera: all group poses + fovy + lens (k1, k2, cx, cy)  -- CMA-ES
  C  latency grid per camera (dt in -3..3 frames), then CMA on joint offsets (5, regularised:
     0.004 * sum((off / OFFSET_SIGMA)^2)) + gripper scale + latencies, then the same plus every group pose (small steps)
  D  per camera: B again (short)
Fit frames: per camera, frames with the real arm visible, chosen by farthest-point sampling in joint space, stratified
by episode group (>= nfit per camera, >= 12 per group). Reported: median / 10th percentile IoU over ALL frames with
the arm visible (real mask >= 0.1 % of the image), full resolution with the lens warp, plus the same numbers for the
old builder twin (its arm silhouette from so101_twin/masks/prompts/) against the same SAM masks.

Writes so101_twin/common/out/<scene><out-tag>_fit.json (params + stats incl. held-out), <scene><out-tag>_iou.npz
(per-frame IoU old/new per camera), <scene><out-tag>_fit_<cam>.jpg (overlays: white = both, red = real only,
blue = twin only), <scene><out-tag>_fit.log and _fit_partial.json (params after every stage).

Scene description (a python module exporting):
  REAL_SET       name of the real FrameSet, e.g. "so101_lemon_real"
  ARM            arm.ArmSpec(...)  (calib 'old' | 'new', model 'so101' | 'so100', wrist_mount, colours, base_pos,
                 base_yaw_deg)
  CAMERAS        {dataset_cam: dict(mount='world'|'gripper'|'shoulder', pos=(..), quat=(w,x,y,z), fovy=deg,
                                    group=None | callable(episode:int) -> str,   # moved camera: one pose per group
                                    init_groups=None | {group: dict(pos=.., quat=..)})}  # optional per-group inits
  OFFSETS_INIT   optional (5,) degrees; ROLL_CANDIDATES optional list of wrist-roll offsets (deg) tried before stage C
  OFFSET_SIGMA   optional (5,) regulariser scale in degrees (default 4)
  FIT_LAG        optional bool (default True): fit per-camera latency if the full-rate parquet is available
  FIT_SCALES     optional bool (default False): also fit per-joint scales (deg per state unit, regularised to 1 with
                 sd 0.05) for datasets recorded in LeRobot's normalised -100..100 mode; SCALES_INIT optional (5,)
  GROUP_RULE     optional {cam: text} describing the episode grouping (copied into the twin set's meta.json)
  extras(root, asset, worldbody)   optional: scene objects / lights / table for the RGB render (not used by the fit)
  frame_setup(rig, fs, i)          optional: move mocap objects for frame i before rendering RGB
  GAIN           optional per camera RGB gain (else estimated from real vs sim mean colour)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time

import cma
import mujoco
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
TWIN = HERE.parent
ROOT = TWIN.parent
OUT = HERE / "out"
MASKS = HERE / "masks"   # BALLCUP: masks live in so101_twin/ballcup/masks
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
import arm as A  # noqa: E402

FW, FH = 320, 240           # fit resolution
VISIBLE = 0.001             # real arm area fraction for "arm visible"
SCALE = dict(pos_world=0.02, pos_mount=0.008, rot=0.03, fovy=2.0, k1=0.08, k2=0.04, c=0.01, off=3.0, grip=0.08,
             lag=1.0, jscale=0.05, gripoff=0.05, mount_pos=0.005, mount_rot=0.05)


# ---------------------------------------------------------------------------------------------------- scene / data
def load_scene(name: str):
    spec = importlib.util.spec_from_file_location(f"scene_{name}", HERE / f"scene_{name}.py")   # LEGOCUP2
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.NAME = name
    return mod


def unpack(a: np.ndarray) -> np.ndarray:
    return np.unpackbits(a, axis=-1)[..., :640].astype(bool)


def load_masks(scene: str, cam: str, tag: str = "", sub: str = "") -> np.ndarray:
    return unpack(np.load(MASKS / sub / f"{scene}_{cam}{tag}.npy"))


def down2(m: np.ndarray) -> np.ndarray:
    n, h, w = m.shape
    return m.reshape(n, h // 2, 2, w // 2, 2).mean((2, 4)) >= 0.5


def init_params(sc) -> dict:
    cams = {}
    for c, cs in sc.CAMERAS.items():
        groups = {}
        for g in group_keys(sc, c):
            gi = (cs.get("init_groups") or {}).get(g, {})
            groups[g] = dict(pos=list(map(float, gi.get("pos", cs["pos"]))), quat=list(map(float, gi.get("quat", cs["quat"]))))
        cams[c] = dict(mount=cs.get("mount", "world"), fovy=float(cs["fovy"]), groups=groups,
                       lens=dict(k1=0.0, k2=0.0, cx=0.0, cy=0.0), dt_frames=0.0)
    off = list(map(float, getattr(sc, "OFFSETS_INIT", [0.0] * 5)))
    return dict(cams=cams, offsets_deg=off, grip_scale=float(getattr(sc, "GRIP_SCALE_INIT", 1.0)),
                cup=dict(getattr(sc, "CUP_INIT", {})),   # BALLCUP
                grip_off=float(getattr(sc, "GRIP_OFF_INIT", 0.0)),   # LEGOCUP2
                scales=list(map(float, getattr(sc, "SCALES_INIT", [1.0] * 5))))


def reconcile(sc, P: dict) -> dict:
    """Make loaded params match the scene's current camera groups (new groups start from 'all' / the first pose)."""
    P = json.loads(json.dumps(P))
    init = init_params(sc)   # LEGOCUP2: cameras missing from P start at the scene init, extra cameras are dropped
    P["cams"] = {c: P["cams"].get(c, init["cams"][c]) for c in sc.CAMERAS}
    P.setdefault("scales", [1.0] * 5)
    P.setdefault("grip_scale", 1.0)
    P.setdefault("grip_off", 0.0)   # LEGOCUP2
    if hasattr(sc, "CUP_INIT"):   # BALLCUP
        P.setdefault("cup", dict(sc.CUP_INIT))
    for c in sc.CAMERAS:
        cp = P["cams"][c]
        cp.setdefault("dt_frames", 0.0)
        want = group_keys(sc, c)
        src = cp["groups"].get("all") or next(iter(cp["groups"].values()))
        cp["groups"] = {g: json.loads(json.dumps(cp["groups"].get(g, src))) for g in want}
    return P


_FS = {}


def real_set(sc):
    if sc.REAL_SET not in _FS:
        from sets import Store

        _FS[sc.REAL_SET] = Store().frames(sc.REAL_SET)
    return _FS[sc.REAL_SET]


_FR = {}


def fullrate(sc):
    if sc.REAL_SET not in _FR:
        from fullrate import FullRate

        _FR[sc.REAL_SET] = FullRate(real_set(sc))
    return _FR[sc.REAL_SET]


def frame_state(sc, P: dict, cam: str, i: int) -> np.ndarray:
    """LeRobot state to pose the arm with for camera `cam` at real frame i: the frame's state shifted by the camera's
    fitted latency dt_frames (image time - state time, in 30 fps frames; needs the full-rate parquet, else dt = 0)."""
    dt = float(P["cams"][cam].get("dt_frames", 0.0))
    if dt == 0.0:
        return real_set(sc).state[i]
    return fullrate(sc).state_at(i, dt)


def frame_groups(sc, cam: str) -> np.ndarray:
    fs = real_set(sc)
    fn = sc.CAMERAS[cam].get("group")
    ep = np.asarray(fs.labels["episode"]).astype(int)
    return np.array(["all" if fn is None else str(fn(int(e))) for e in ep])


def group_keys(sc, cam: str) -> list:
    return sorted(set(frame_groups(sc, cam)))


# ---------------------------------------------------------------------------------------------------- rig
class Rig:
    """The arm (+ optionally the scene) with all dataset cameras, driven by a params dict."""

    def __init__(self, sc, params: dict, full: bool = False):
        self.sc = sc
        cams = []
        for c, p in params["cams"].items():
            g0 = next(iter(p["groups"].values()))
            cams.append(A.CamSpec(c, mount=p["mount"], pos=tuple(g0["pos"]), quat=tuple(g0["quat"]), fovy=p["fovy"]))
        # BALLCUP: occ=True adds only the scene's occluders (cup, ball: sc.occluders) so the arm mask is the VISIBLE arm
        self.occ = (bool(getattr(sc, "MASK_OCCLUDERS", False)) or float(getattr(sc, "CUP_W", 0.0)) > 0) and not full \
            and hasattr(sc, "occluders")
        self.has_obj = (self.occ or full) and hasattr(sc, "update_objects") and getattr(sc, "PIX", None) is not None
        extras = getattr(sc, "extras", None) if full else (sc.occluders if self.occ else None)
        xml = A.build_mjcf(sc.ARM, cams, extras=extras, model_name=f"so101_{sc.NAME}_twin2", offwidth=2048,
                           offheight=1600)   # LEGOCUP2: room for the wide-angle lens pad
        self.xml = xml
        self.arm = A.Arm(xml, sc.ARM.calib, sc.ARM.model)
        self.arm.hide_groups = {c: tuple(sc.CAMERAS[c].get("hide_groups", ())) for c in params["cams"]}   # LEGOCUP2
        # LEGOCUP2: optional rigid offset of the wrist-camera bracket+module (body wrist_camera_mount) in the gripper frame
        self._mb = mujoco.mj_name2id(self.arm.m, mujoco.mjtObj.mjOBJ_BODY, "wrist_camera_mount")
        self._mb0 = (self.arm.m.body_pos[self._mb].copy(), self.arm.m.body_quat[self._mb].copy()) if self._mb >= 0 else None
        self.set_params(params)

    def set_params(self, params: dict):
        self.p = params
        self.arm.offsets_deg = np.asarray(params["offsets_deg"], float)
        self.arm.scales = np.asarray(params.get("scales", [1.0] * 5), float)
        self.arm.grip_scale = float(params.get("grip_scale", 1.0))
        self.arm.grip_off = float(params.get("grip_off", 0.0))   # LEGOCUP2
        self.arm.compliance = np.asarray(params.get("compliance", [0.0] * 5), float)   # JAR gravity sag (deg / N m)
        if self._mb0 is not None:   # LEGOCUP2 bracket offset: [dx, dy, dz (m), rotvec (rad)] in the gripper frame
            mo = np.asarray(params.get("mount_off", [0.0] * 6), float)
            self.arm.m.body_pos[self._mb] = self._mb0[0] + mo[:3]
            self.arm.m.body_quat[self._mb] = A.quat_mul(A.quat_from_rotvec(mo[3:6]), self._mb0[1])
        self.lens = {c: A.Lens(**p["lens"], pad=self.sc.CAMERAS[c].get("pad", 0.25)) for c, p in params["cams"].items()}
        if getattr(self, "has_obj", False):   # BALLCUP: cup shape + object placement for the current cameras
            self.sc.update_objects(self, params)
            if not hasattr(self, "_cup_lut"):
                self._cup_lut = np.zeros(self.arm.m.ngeom + 1, bool)
                self._cup_lut[np.asarray(self.sc.cup_geom_ids(self.arm.m)) + 1] = True

    def pose(self, cam: str, group: str, state):
        p = self.p["cams"][cam]
        g = p["groups"][group]
        cid = self.arm.cam_id[cam]
        self.arm.m.cam_pos[cid] = g["pos"]
        self.arm.m.cam_quat[cid] = np.asarray(g["quat"]) / np.linalg.norm(g["quat"])
        self.arm.m.cam_fovy[cid] = p["fovy"]
        self.arm.set_state(state)

    def mask(self, cam, group, state, w=FW, h=FH, i=None):
        self.pose(cam, group, state)
        if self.has_obj and i is not None:   # BALLCUP: place the occluders for real frame i
            self.sc.place_objects(self, int(i))
        return self.arm.arm_mask(cam, w, h, self.lens[cam])

    def masks2(self, cam, group, state, i, w=FW, h=FH):
        """BALLCUP: (visible arm mask, visible cup mask) from ONE segmentation render with the objects placed."""
        self.pose(cam, group, state)
        self.sc.place_objects(self, int(i))
        seg = self.lens[cam].render(self.arm, cam, w, h, seg=True)
        geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        return self.arm._arm_lut[seg[..., 0] + 1] & geom, self._cup_lut[seg[..., 0] + 1] & geom

    def rgb(self, cam, group, state, w=640, h=480):
        self.pose(cam, group, state)
        return self.arm.render(cam, w, h, self.lens[cam])


def iou_frames(sim: np.ndarray, real: np.ndarray) -> np.ndarray:
    """per-frame IoU; nan where both are empty."""
    inter = (sim & real).reshape(len(sim), -1).sum(1)
    union = (sim | real).reshape(len(sim), -1).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


# ---------------------------------------------------------------------------------------------------- parameter vectors
def apply_x(params: dict, layout: list, x: np.ndarray) -> dict:
    """Return a copy of params perturbed by the scaled vector x according to layout."""
    P = json.loads(json.dumps(params))
    k = 0
    for item in layout:
        kind = item[0]
        if kind == "pose":
            _, c, g = item
            p = P["cams"][c]
            s = SCALE["pos_world"] if p["mount"] == "world" else SCALE["pos_mount"]
            gp = p["groups"][g]
            gp["pos"] = list(np.asarray(gp["pos"]) + s * x[k:k + 3])
            gp["quat"] = list(A.quat_mul(gp["quat"], A.quat_from_rotvec(SCALE["rot"] * x[k + 3:k + 6])))
            k += 6
        elif kind == "fovy":
            P["cams"][item[1]]["fovy"] += SCALE["fovy"] * x[k]
            k += 1
        elif kind == "lens":
            L = P["cams"][item[1]]["lens"]
            if item[1] not in RADIAL_FIXED:   # JAR: --fix-radial CAM keeps k1 = k2 = 0 for that camera
                L["k1"] += SCALE["k1"] * x[k]
                L["k2"] += SCALE["k2"] * x[k + 1]
            L["cx"] += SCALE["c"] * x[k + 2]
            L["cy"] += SCALE["c"] * x[k + 3]
            k += 4
        elif kind == "off":
            P["offsets_deg"] = list(np.asarray(P["offsets_deg"]) + SCALE["off"] * x[k:k + 5])
            k += 5
        elif kind == "grip":
            P["grip_scale"] += SCALE["grip"] * x[k]
            k += 1
        elif kind == "mount":   # LEGOCUP2: wrist-camera bracket offset
            mo = np.asarray(P.get("mount_off", [0.0] * 6), float)
            mo[:3] += SCALE["mount_pos"] * x[k:k + 3]
            mo[3:6] += SCALE["mount_rot"] * x[k + 3:k + 6]
            P["mount_off"] = mo.tolist()
            k += 6
        elif kind == "dolly":   # BALLCUP: move along the optical axis, fovy rescaled to keep the size at DOLLY_REF
            c = item[1]
            p = P["cams"][c]
            for gp in p["groups"].values():
                fwd = -A.mat_from_quat(np.asarray(gp["quat"]) / np.linalg.norm(gp["quat"]))[:, 2]
                pos = np.asarray(gp["pos"], float)
                d_ref = float(np.linalg.norm(np.asarray(DOLLY_REF) - pos))
                dl = 0.03 * x[k]
                gp["pos"] = list(pos - dl * fwd)
            t = math.tan(math.radians(p["fovy"]) / 2) * d_ref / max(d_ref + dl, 0.05)
            p["fovy"] = math.degrees(2 * math.atan(t))
            k += 1
        elif kind == "cup":   # BALLCUP: cup shape r_top, r_bot, h (m) and the top-ray height fraction zc
            cu = P.setdefault("cup", {})
            cu["r_top"] = cu["r_top"] + 0.003 * x[k]
            cu["r_bot"] = cu["r_bot"] + 0.003 * x[k + 1]
            cu["h"] = cu["h"] + 0.006 * x[k + 2]
            cu["zc"] = float(np.clip(cu["zc"] + 0.05 * x[k + 3], 0.0, 1.05))
            k += 4
        elif kind == "comp":   # JAR (as ORANGEBOX): compliance of shoulder_lift, elbow_flex, wrist_flex (deg / N m)
            cp_ = np.asarray(P.get("compliance", [0.0] * 5), float)
            cp_[1:4] += 2.0 * x[k:k + 3]
            P["compliance"] = cp_.tolist()
            k += 3
        elif kind == "gripoff":   # LEGOCUP2
            P["grip_off"] = P.get("grip_off", 0.0) + SCALE["gripoff"] * x[k]
            k += 1
        elif kind == "scale":
            P["scales"] = list(np.asarray(P.get("scales", [1.0] * 5)) + SCALE["jscale"] * x[k:k + 5])
            k += 5
        elif kind == "lag":
            P["cams"][item[1]]["dt_frames"] = P["cams"][item[1]].get("dt_frames", 0.0) + SCALE["lag"] * x[k]
            k += 1
    return P


def dim(layout) -> int:
    return sum({"pose": 6, "fovy": 1, "lens": 4, "off": 5, "grip": 1, "gripoff": 1, "lag": 1, "scale": 5, "mount": 6,
                "cup": 4, "dolly": 1, "comp": 3}[it[0]]
               for it in layout)


class Objective:
    def __init__(self, rig: Rig, sc, fit: dict, reg_base: dict):
        """fit: {cam: (frame_idx, groups[idx], real_small[idx])}"""
        self.rig, self.sc, self.fit = rig, sc, fit
        self.states = real_set(sc).state
        self.off0 = np.asarray(reg_base["offsets_deg"], float)
        self.sigma = np.asarray(getattr(sc, "OFFSET_SIGMA", [4.0] * 5), float)
        self.n = 0
        self.cup_w = float(getattr(sc, "CUP_W", 0.0))   # BALLCUP: cup IoU term (depth cue for the cameras)
        if self.cup_w > 0:
            pix = sc.PIX["frames"]
            self.cup_real, self.cup_ok = {}, {}
            for c, (idx, _, _) in fit.items():
                cm = load_masks(sc.NAME, c, getattr(sc, "CUP_MASK_TAG", "_d1_cup"))
                self.cup_real[c] = down2(cm[idx])
                self.cup_ok[c] = np.array([pix[str(int(i))]["cup_clean"] for i in idx]) & \
                    (self.cup_real[c].reshape(len(idx), -1).sum(1) > 20)

    def score(self, P: dict, cams=None, reg=True) -> float:
        self.rig.set_params(P)
        vals = []
        cvals = []
        for c in (cams or list(self.fit)):
            idx, grp, real = self.fit[c]
            if self.cup_w > 0:   # BALLCUP: one render -> visible arm + visible cup
                mm = [self.rig.masks2(c, g, frame_state(self.sc, P, c, i), i) for i, g in zip(idx, grp)]
                sim = np.stack([m[0] for m in mm])
                cs = np.stack([m[1] for m in mm])
                ok = self.cup_ok[c]
                cv = iou_frames(cs[ok], self.cup_real[c][ok]) if ok.any() else np.array([np.nan])
                cvals.append(float(np.nanmean(cv)) if np.isfinite(cv).any() else 0.0)
            else:
                sim = np.stack([self.rig.mask(c, g, frame_state(self.sc, P, c, i), i=i) for i, g in zip(idx, grp)])
            v = iou_frames(sim, real)
            trim = float(getattr(self.sc, "OBJ_TRIM", 0.0))   # LEGOCUP2: optional trimmed mean (drop worst frames)
            if trim > 0 and np.isfinite(v).sum() > 5:
                vv = np.sort(v[np.isfinite(v)])
                vals.append(float(vv[int(len(vv) * trim):].mean()))
            else:
                vals.append(np.nanmean(v) if np.isfinite(v).any() else 0.0)
        s = float(np.mean(vals))
        if cvals:
            s += self.cup_w * float(np.mean(cvals))
            self.last_cup = float(np.mean(cvals))
        if reg:
            d = (np.asarray(P["offsets_deg"]) - self.off0) / self.sigma
            s -= 0.004 * float(np.sum(d ** 2))
            ds = (np.asarray(P.get("scales", [1.0] * 5)) - np.asarray(getattr(self.sc, "SCALES_PRIOR", [1.0] * 5))) \
                / np.asarray(getattr(self.sc, "SCALE_SIGMA", [0.05] * 5))   # LEGOCUP2: configurable prior
            s -= 0.004 * float(np.sum(ds ** 2))
            cpl = np.asarray(P.get("compliance", [0.0] * 5), float)   # JAR: weak prior on the sag, sd 10 deg / N m
            s -= 0.004 * float(np.sum((cpl / 10.0) ** 2))
            mo = np.asarray(P.get("mount_off", [0.0] * 6), float)   # LEGOCUP2: bracket offset, sd 1.5 cm / 10 deg
            s -= 0.004 * float(np.sum((mo[:3] / 0.015) ** 2) + np.sum((mo[3:] / 0.175) ** 2))
        self.n += 1
        return s


def optimise(obj: Objective, P0: dict, layout: list, cams, evals: int, coarse: int = 0, coarse_sd: float = 1.5,
             sigma0: float = 0.6, seed: int = 0, log=print, tag=""):
    n = dim(layout)
    rng = np.random.default_rng(seed)
    best_x, best = np.zeros(n), obj.score(P0, cams)
    start = best
    for _ in range(coarse):
        x = rng.normal(0, coarse_sd, n)
        s = obj.score(apply_x(P0, layout, x), cams)
        if s > best:
            best, best_x = s, x
    if coarse:
        log(f"{tag} coarse {start:.3f} -> {best:.3f}")
    es = cma.CMAEvolutionStrategy(best_x, sigma0, dict(maxfevals=evals, popsize=max(8, 4 + int(3 * math.log(n))),
                                                         verbose=-9, seed=seed + 1, bounds=[-6, 6]))
    while not es.stop():
        xs = es.ask()
        fs = [-obj.score(apply_x(P0, layout, x), cams) for x in xs]
        es.tell(xs, fs)
        if -min(fs) > best:
            best, best_x = -min(fs), xs[int(np.argmin(fs))]
    log(f"{tag} {start:.3f} -> {best:.3f} ({es.countevals} evals)")
    return apply_x(P0, layout, best_x), best


# ---------------------------------------------------------------------------------------------------- frames
def fps_select(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    if len(X) <= k:
        return np.arange(len(X))
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(len(X)))]
    d = np.linalg.norm(X - X[sel[0]], axis=1)
    while len(sel) < k:
        j = int(np.argmax(d))
        sel.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    return np.array(sorted(sel))


def fit_episode_mask(sc, split: str) -> np.ndarray:
    """bool [n]: frames the fit may use. 'even' (official protocol): even episodes only, odd ones are held out."""
    ep = np.asarray(real_set(sc).labels["episode"]).astype(int)
    return (ep % 2 == 0) if split == "even" else np.ones(len(ep), bool)


def choose_fit_frames(sc, cam: str, real_full: np.ndarray, nfit: int, split: str = "even") -> np.ndarray:
    fs = real_set(sc)
    area = real_full.reshape(len(real_full), -1).mean(1)
    vis = np.nonzero((area >= VISIBLE) & fit_episode_mask(sc, split))[0]
    grp = frame_groups(sc, cam)
    X = fs.state[:, :6] / np.array([30, 30, 30, 30, 30, 30.0])
    keys = sorted(set(grp[vis]))
    out = []
    mode = getattr(sc, "FIT_FRAMES", "fps")   # JAR: 'random' = uniform sample (matches the score's frame mix)
    for g in keys:
        cand = vis[grp[vis] == g]
        k = max(12, int(round(nfit * len(cand) / len(vis))))
        if mode == "random":
            rng = np.random.default_rng(7)
            out.extend(rng.choice(cand, size=min(k, len(cand)), replace=False))
        elif mode == "mix":   # JAR: half farthest-point in joint space (coverage), half uniform random (frame mix)
            sel = list(cand[fps_select(X[cand], k // 2)])
            rest_ = np.setdiff1d(cand, sel)
            rng = np.random.default_rng(7)
            sel += list(rng.choice(rest_, size=min(k - len(sel), len(rest_)), replace=False))
            out.extend(sel)
        else:
            out.extend(cand[fps_select(X[cand], k)])
    return np.array(sorted(set(int(v) for v in out)))


# ---------------------------------------------------------------------------------------------------- evaluation
def evaluate(rig: Rig, sc, real: dict, prompts: dict | None = None) -> dict:
    """Full-resolution IoU over all frames with the arm visible, new twin (and old twin from prompts)."""
    fs = real_set(sc)
    out = {}
    for c in real:
        grp = frame_groups(sc, c)
        area = real[c].reshape(fs.n, -1).mean(1)
        vis = area >= VISIBLE
        new = np.full(fs.n, np.nan)
        old = np.full(fs.n, np.nan)
        for i in range(fs.n):
            if not vis[i]:
                continue
            m = rig.mask(c, grp[i], frame_state(sc, rig.p, c, i), 640, 480, i=i)
            new[i] = iou_frames(m[None], real[c][i][None])[0]
            if prompts is not None:
                old[i] = iou_frames(prompts[c][i][None], real[c][i][None])[0]
        odd = np.asarray(fs.labels["episode"]).astype(int) % 2 == 1
        out[c] = dict(new=stat(new), old=stat(old) if prompts is not None else None, n_visible=int(vis.sum()),
                      new_heldout=stat(np.where(odd, new, np.nan)),
                      old_heldout=stat(np.where(odd, old, np.nan)) if prompts is not None else None,
                      iou_new=new, iou_old=old)
    return out


def stat(v: np.ndarray) -> dict:
    if not np.isfinite(v).any():
        return dict(median=float("nan"), p10=float("nan"), mean=float("nan"), n=0)
    return dict(median=float(np.nanmedian(v)), p10=float(np.nanpercentile(v, 10)), mean=float(np.nanmean(v)),
                n=int(np.isfinite(v).sum()))


def overlay(rig: Rig, sc, real: dict, path_fmt: str, k: int = 8):
    fs = real_set(sc)
    for c in real:
        grp = frame_groups(sc, c)
        vis = np.nonzero(real[c].reshape(fs.n, -1).mean(1) >= VISIBLE)[0]
        idx = vis[np.linspace(0, len(vis) - 1, k).astype(int)]
        tiles = []
        for i in idx:
            im = np.asarray(fs.obs(int(i))["images"][c]).astype(np.float32) * 0.5
            s = rig.mask(c, grp[i], frame_state(sc, rig.p, c, i), 640, 480, i=i)
            r = real[c][i]
            ov = np.zeros((480, 640, 3), np.float32)
            ov[r & s] = 255
            ov[r & ~s] = (255, 40, 40)
            ov[~r & s] = (40, 90, 255)
            a = (r | s)[..., None]
            tiles.append(np.where(a, 0.35 * im + 0.65 * ov, im).astype(np.uint8))
        rows = [np.concatenate(tiles[j:j + 4], 1) for j in range(0, k, 4)]
        Image.fromarray(np.concatenate(rows, 0)).resize((1280, 480 * len(rows) // 2)).save(path_fmt.format(cam=c), quality=88)


# ---------------------------------------------------------------------------------------------------- main fit
DOLLY_REF = (0.0, 0.15, 0.10)   # BALLCUP: a point in the middle of the arm's workspace (world)
RADIAL_FIXED = set()   # JAR: cameras whose radial terms are held at 0 (--fix-radial)


def DOLLY(sc, c):   # BALLCUP: dolly-zoom direction for camera c (when the cup term is on)
    return [("dolly", c)] if float(getattr(sc, "CUP_W", 0.0)) > 0 else []


def CUPL(sc, per_cam=False):   # BALLCUP: cup shape params join every stage when the cup term is on
    if per_cam and not getattr(sc, "CUP_PER_CAM", True):   # --cup-joint-only: shared cup only in the joint stage C
        return []
    return [("cup",)] if float(getattr(sc, "CUP_W", 0.0)) > 0 else []


def checkpoint(sc, P: dict, stage: str):
    """params after each stage -> out/<scene><tag>_fit_partial.json (inspect / resume with --from-partial)."""
    (OUT / f"{sc.NAME}{getattr(sc, 'OUT_TAG', '')}_fit_partial.json").write_text(
        json.dumps(dict(stage=stage, params=P), indent=1))


def fit_scene(sc, mask_tag: str = "", nfit: int = 72, ev: float = 1.0, P: dict | None = None, log=print,
              stages: str = "ABCD", split: str = "even") -> dict:
    fs = real_set(sc)
    real = {c: load_masks(sc.NAME, c, mask_tag) for c in sc.CAMERAS}
    P = reconcile(sc, P) if P else init_params(sc)
    rig = Rig(sc, P)
    fit = {}
    for c in sc.CAMERAS:
        idx = choose_fit_frames(sc, c, real[c], nfit, split)
        fit[c] = (idx, frame_groups(sc, c)[idx], down2(real[c][idx]))
        log(f"{c}: {len(idx)} fit frames, groups {sorted(set(fit[c][1]))}")
    obj = Objective(rig, sc, fit, P)
    E = lambda n: max(40, int(n * ev))   # noqa: E731
    t0 = time.time()
    log(f"start: " + ", ".join(f"{c} {obj.score(P, [c], reg=False):.3f}" for c in sc.CAMERAS))
    # A: per camera / group pose + fovy
    for c in (sc.CAMERAS if "A" in stages else []):
        keys = sorted(set(fit[c][1]))
        for g in keys:
            sub = {c: tuple(a[fit[c][1] == g] for a in fit[c])}
            o2 = Objective(rig, sc, sub, P)
            lay = [("pose", c, g)] + ([("fovy", c)] if len(keys) == 1 else []) + DOLLY(sc, c) + CUPL(sc, True)
            P, _ = optimise(o2, P, lay, [c], E(350), coarse=E(60), log=log, tag=f"[A {c}/{g} {time.time() - t0:.0f}s]")
            checkpoint(sc, P, f"A {c}/{g}")
    # B: per camera, all groups + fovy + lens
    for c in (sc.CAMERAS if "B" in stages else []):
        lay = [("pose", c, g) for g in sorted(set(fit[c][1]))] + [("fovy", c), ("lens", c)] + DOLLY(sc, c) + CUPL(sc, True)
        P, _ = optimise(obj, P, lay, [c], E(450), sigma0=0.5, log=log, tag=f"[B {c} {time.time() - t0:.0f}s]")
        checkpoint(sc, P, f"B {c}")
    # roll candidates (wrist roll zero is arbitrary for some calibrations)
    rc = getattr(sc, "ROLL_CANDIDATES", None)
    if rc and "C" in stages:
        best = None
        for r in rc:
            Q = json.loads(json.dumps(P))
            Q["offsets_deg"][4] = float(r)
            s = obj.score(Q, reg=False)
            log(f"roll offset {r}: {s:.3f}")
            if best is None or s > best[0]:
                best = (s, Q)
        P = best[1]
        obj.off0 = np.asarray(P["offsets_deg"], float)
    # C: camera latency (per camera, if the full-rate states are available), joint offsets, gripper scale; then all
    # of them together with every camera pose (small steps: the poses are already near their optimum)
    if "C" in stages:
        lags = []
        if getattr(sc, "FIT_LAG", True) and fullrate(sc).ok:
            for c in sc.CAMERAS:
                best = None
                for dt in (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0):   # LEGOCUP2 finer grid
                    Q = json.loads(json.dumps(P))
                    Q["cams"][c]["dt_frames"] = dt
                    v = obj.score(Q, [c], reg=False)
                    if best is None or v > best[0]:
                        best = (v, dt)
                P["cams"][c]["dt_frames"] = best[1]
                log(f"[C lag {c}] grid best dt {best[1]:+.0f} frames: {best[0]:.3f}")
                lags.append(("lag", c))
        lay = [(k,) for k in getattr(sc, "JOINT_PARAMS", ("off", "grip"))] + lags \
            + ([("scale",)] if getattr(sc, "FIT_SCALES", False) else []) + CUPL(sc)   # LEGOCUP2: JOINT_PARAMS
        P, _ = optimise(obj, P, lay, None, E(300), sigma0=0.4, log=log, tag=f"[C1 joints {time.time() - t0:.0f}s]")
        lay += [("pose", c, g) for c in sc.CAMERAS for g in sorted(set(fit[c][1]))]
        P, _ = optimise(obj, P, lay, None, E(500), sigma0=0.15, log=log, tag=f"[C2 all {time.time() - t0:.0f}s]")
        log(f"offsets {np.round(P['offsets_deg'], 2).tolist()} grip_scale {P['grip_scale']:.3f} grip_off "
            f"{P.get('grip_off', 0.0):.3f} dt_frames "
            + str({c: round(P['cams'][c].get('dt_frames', 0.0), 2) for c in sc.CAMERAS}))
        checkpoint(sc, P, "C")
    # D: per camera polish
    for c in (sc.CAMERAS if "D" in stages else []):
        lay = [("pose", c, g) for g in sorted(set(fit[c][1]))] + [("fovy", c), ("lens", c)] + DOLLY(sc, c) + CUPL(sc, True)
        P, _ = optimise(obj, P, lay, [c], E(250), sigma0=0.3, log=log, tag=f"[D {c} {time.time() - t0:.0f}s]")
        checkpoint(sc, P, f"D {c}")
    rig.set_params(P)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--mask-tag", default="")
    ap.add_argument("--nfit", type=int, default=72)
    ap.add_argument("--evals", type=float, default=1.0, help="multiplier on the per-stage evaluation budgets")
    ap.add_argument("--from-json", action="store_true", help="start from out/<scene>_fit.json instead of the scene init")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--from-partial", action="store_true", help="start from out/<scene>_fit_partial.json")
    ap.add_argument("--stages", default="ABCD", help="subset of stages to run, e.g. CD")
    ap.add_argument("--export-prompts", default="", help="after fitting, write the new twin's arm silhouettes for all "
                    "frames to so101_twin/masks/<DIR>/<scene>_<cam>.npy (packed) to re-prompt segment.py --prompts")
    ap.add_argument("--split", default="even", choices=("even", "all"),
                    help="fit frames from even episodes only (official: odd episodes are the held-out score) or all")
    ap.add_argument("--model", default="", choices=("", "so101", "so100"), help="override the scene's arm model")
    ap.add_argument("--out-tag", default="", help="suffix for every output file of this run, e.g. _so100")
    ap.add_argument("--init-json", default="", help="start from this fit json (any tag)")
    ap.add_argument("--cams", default="", help="LEGOCUP2: fit only these cameras (comma-separated)")
    ap.add_argument("--occ", action="store_true", help="BALLCUP: cup / ball occlude the twin arm mask (visible arm)")
    ap.add_argument("--cup-w", type=float, default=0.0, help="BALLCUP: weight of the cup IoU term (implies --occ)")
    ap.add_argument("--cup-joint-only", action="store_true", help="BALLCUP: cup shape only in the joint stage C")
    ap.add_argument("--joint-params", default="", help="LEGOCUP2: stage-C joint params, e.g. off,grip,gripoff")
    ap.add_argument("--obj-trim", type=float, default=-1, help="LEGOCUP2: trimmed-mean objective (fraction dropped)")
    ap.add_argument("--fit-frames", default="", choices=("", "fps", "random", "mix"), help="JAR: fit-frame selection")
    ap.add_argument("--fix-radial", default="", help="JAR: cameras with k1 = k2 = 0 held fixed (comma-separated)")
    a = ap.parse_args()
    sc = load_scene(a.scene)
    if a.occ or a.cup_w > 0:   # BALLCUP
        sc.MASK_OCCLUDERS = True
    sc.CUP_W = a.cup_w
    sc.CUP_PER_CAM = not a.cup_joint_only
    if a.cams:   # LEGOCUP2
        sc.CAMERAS = {c: sc.CAMERAS[c] for c in a.cams.split(",")}
    if a.joint_params:
        sc.JOINT_PARAMS = tuple(a.joint_params.split(","))
    if a.obj_trim >= 0:
        sc.OBJ_TRIM = a.obj_trim
    if a.fit_frames:   # JAR
        sc.FIT_FRAMES = a.fit_frames
    if a.fix_radial:   # JAR
        RADIAL_FIXED.update(a.fix_radial.split(","))
    if a.model:
        import dataclasses

        sc.ARM = dataclasses.replace(sc.ARM, model=a.model)
    sc.OUT_TAG = a.out_tag
    stem = f"{a.scene}{a.out_tag}"
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"{stem}_fit.log", "a")

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    jpath = OUT / f"{stem}_fit.json"
    P = json.loads(jpath.read_text())["params"] if (a.from_json or a.eval_only) and jpath.exists() else None
    if a.from_partial:
        P = json.loads((OUT / f"{stem}_fit_partial.json").read_text())["params"]
    if a.init_json:
        P = json.loads(pathlib.Path(a.init_json).read_text())["params"]
    t0 = time.time()
    log(f"run {stem}: model {sc.ARM.model} calib {sc.ARM.calib} masks '{a.mask_tag}' split {a.split} stages "
        f"{a.stages if not a.eval_only else '-'} evals x{a.evals}")
    for c in RADIAL_FIXED:   # JAR
        if P is not None and c in P["cams"]:
            P["cams"][c]["lens"]["k1"] = 0.0
            P["cams"][c]["lens"]["k2"] = 0.0
    if not a.eval_only:
        P = fit_scene(sc, a.mask_tag, a.nfit, a.evals, P, log, stages=a.stages, split=a.split)
    P = reconcile(sc, P)
    rig = Rig(sc, P)
    real = {c: load_masks(sc.NAME, c, a.mask_tag) for c in sc.CAMERAS}
    prompts = None   # BALLCUP: this scene has no old builder twin
    ev = evaluate(rig, sc, real, prompts)
    stats = {c: dict(new=v["new"], old=v["old"], new_heldout=v["new_heldout"], old_heldout=v["old_heldout"],
                     n_visible=v["n_visible"]) for c, v in ev.items()}
    for c, v in stats.items():   # BALLCUP: no old twin columns
        log(f"EVAL {c}: ALL median {v['new']['median']:.3f} p10 {v['new']['p10']:.3f} mean {v['new']['mean']:.3f}"
            f" | n={v['n_visible']}")
        log(f"EVAL {c}: HELD-OUT (odd episodes) median {v['new_heldout']['median']:.3f} p10 "
            f"{v['new_heldout']['p10']:.3f} mean {v['new_heldout']['mean']:.3f} | n={v['new_heldout']['n']}")
    np.savez(OUT / f"{stem}_iou.npz", **{f"{c}_new": v["iou_new"] for c, v in ev.items()},
             **{f"{c}_old": v["iou_old"] for c, v in ev.items()})
    jpath.write_text(json.dumps(dict(scene=a.scene, model=sc.ARM.model, calib=sc.ARM.calib, params=P, stats=stats,
                                     mask_tag=a.mask_tag, split=a.split, occluders=bool(sc.MASK_OCCLUDERS), cup_w=a.cup_w,
                                     fit_seconds=time.time() - t0,
                                     built=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    overlay(rig, sc, real, str(OUT / f"{stem}_fit_{{cam}}.jpg"))
    if a.export_prompts:
        d = MASKS / a.export_prompts
        d.mkdir(parents=True, exist_ok=True)
        fs = real_set(sc)
        for c in sc.CAMERAS:
            grp = frame_groups(sc, c)
            ms = np.stack([np.packbits(rig.mask(c, grp[i], frame_state(sc, P, c, i), 640, 480, i=i), axis=-1)
                           for i in range(fs.n)])
            np.save(d / f"{a.scene}_{c}.npy", ms)
        log(f"exported prompts to {d}")
    log(f"wrote {jpath}")


if __name__ == "__main__":
    main()
