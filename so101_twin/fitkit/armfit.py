"""Arm-only fitter shared by every SO-101 twin: camera pose, lens (fovy, k1, k2, principal point) and joint offsets.

    .venv/bin/python so101_twin/fitkit/armfit.py lemon [--model so100|so101] [--rounds 2]

What is scored: the overlap (IoU) between the real arm's coloured parts and the twin arm's coloured parts, per camera,
on held-out episodes the fit never saw (odd episodes; the fit uses even episodes). Black servo bodies are holes in both
masks. Lens distortion is applied to the real image (resampled onto the twin's pinhole grid), so the twin renders stay
plain pinhole renders; the fitted k1/k2/principal point are what a replay must apply to its renders to match.

Arm models: 'so100' = Menagerie trs_so_arm100 (what the builders used); 'so101' = TheRobotStudio SO-101 old_calib
(same joint ranges and link lengths; base pan direction opposite, base frame lower, gripper mirrored). The so101
joint map (pan sign, roll sign/offset, jaw sign) is chosen from the real frames, not assumed.

Writes fitkit/out/<scene>_<model>.json and <scene>_<model>_<cam>.jpg (real | twin | overlay: red real only,
blue twin only, white both) for held-out frames.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import pathlib
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize

TWIN = pathlib.Path(__file__).resolve().parent.parent
ROOT = TWIN.parent
OUT = TWIN / "fitkit" / "out"
ASSETS = TWIN / "assets"
sys.path.insert(0, str(ROOT / "neuroscope" / "molmoact2"))
sys.path.insert(0, str(TWIN / "fitkit"))

MODELS = {
    "so100": dict(xml=ASSETS / "so_arm100" / "so_arm100.xml", meshdir=ASSETS / "so_arm100", base="Base",
                  mount=dict(jaw="Fixed_Jaw", shoulder="Rotation_Pitch")),
    "so101": dict(xml=ASSETS / "so101_official" / "so101_old_calib.xml", meshdir=ASSETS / "so101_official" / "assets",
                  base="base", mount=dict(jaw="gripper", shoulder="shoulder")),
}
CAM_KEYS = ("x", "y", "z", "yaw", "pitch", "roll", "fovy", "k1", "k2", "cx", "cy")


# ---------------------------------------------------------------------------------------------------- geometry
def cam_xyaxes(yaw: float, pitch: float, roll: float):
    """Same convention as every scene's build_scene.cam_xyaxes (degrees; pitch < 0 looks down)."""
    y, p, r = map(math.radians, (yaw, pitch, roll))
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return math.cos(r) * right + math.sin(r) * up, -math.sin(r) * right + math.cos(r) * up


def cam_quat(yaw, pitch, roll) -> np.ndarray:
    right, up = cam_xyaxes(yaw, pitch, roll)
    R = np.stack([right, up, np.cross(right, up)], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


# ---------------------------------------------------------------------------------------------------- model
def _arm_root(kind: str, wrist_cam: bool = False) -> ET.Element:
    """Arm MJCF tree. wrist_cam=True (so101 only) adds the official wrist-camera bracket + 32x32 UVC module from
    so101_new_calib_camera.xml under the gripper body (same gripper frame as old_calib up to a 3 deg roll)."""
    spec = MODELS[kind]
    root = ET.parse(spec["xml"]).getroot()
    root.find("compiler").set("meshdir", str(spec["meshdir"]))
    for kf in root.findall("keyframe"):
        root.remove(kf)
    if wrist_cam and kind == "so101":
        asset = root.find("asset")
        ET.SubElement(asset, "mesh", file="wrist_camera_mount_so101_v1.stl", scale="0.001 0.001 0.001")
        ET.SubElement(asset, "mesh", file="wrist_camera_so101_v1.stl")
        ET.SubElement(asset, "material", name="wrist_camera_mount_so101_v1_material", rgba="1 0.82 0.12 1")
        ET.SubElement(asset, "material", name="wrist_camera_so101_v1_material", rgba="0.1 0.1 0.1 1")
        grip = next(b for b in root.iter("body") if b.get("name") == "gripper")
        mb = ET.SubElement(grip, "body", name="wrist_camera_mount", pos="-0.0150 0.0240 -0.0315",
                           quat="0.4997981226682881 -0.5001999599912927 0.4998017973423795 -0.5001999599912927")
        ET.SubElement(mb, "geom", type="mesh", mesh="wrist_camera_mount_so101_v1",
                      material="wrist_camera_mount_so101_v1_material", **{"class": "visual"})
        cb = ET.SubElement(mb, "body", name="wrist_camera", pos="-0.04941 -0.03813 0.0175",
                           quat="0.37978693649389456 -0.5964584309361936 -0.37978833153177 -0.596456240026065")
        ET.SubElement(cb, "geom", type="mesh", mesh="wrist_camera_so101_v1", material="wrist_camera_so101_v1_material",
                      **{"class": "visual"})
    return root


def _base_alignment(kind: str) -> tuple[np.ndarray, float]:
    """Translation that puts this model's pan axis where the SO-100's is, with the base bottoms at the same height
    (both models reach along -y at zero pose in their own frames). Returns (xy shift, z shift) in the base frame."""
    if kind == "so100":
        return np.zeros(2), 0.0

    def probe(k):
        m = mujoco.MjModel.from_xml_string(ET.tostring(_arm_root(k), encoding="unicode"))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, MODELS[k]["base"])
        zmin = np.inf
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != base or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            zmin = min(zmin, w[:, 2].min())
        return d.xanchor[0][:2].copy(), zmin

    a_xy, a_z = probe("so100")
    b_xy, b_z = probe(kind)
    return a_xy - b_xy, a_z - b_z


class ArmScene:
    """Arm + cameras only. Base placed like the scene's SO-100 base (BASE_POS / BASE_QUAT of its build_scene)."""

    def __init__(self, kind: str, base_pos, base_quat, cams, w: int, h: int, mounts: dict | None = None,
                 parts: dict | None = None, wrist_cam: bool = False, wrist_board: dict | None = None):
        """parts: which twin geoms count as 'arm' in the mask. Default: the coloured printed parts (servos excluded).
        dict(servos=True) adds the black servos (for black arms); dict(exclude={'so100': [...], 'so101': [...]})
        drops meshes by name (e.g. a white jaw that the real mask cannot see)."""
        self.kind, self.w, self.h = kind, w, h
        self.parts = parts or {}
        self.pad = PAD
        self.rw, self.rh = int(math.ceil(w * PAD)), int(math.ceil(h * PAD))
        self._maps: dict = {}
        mounts = mounts or {}
        root = _arm_root(kind, wrist_cam)
        if wrist_board and kind == "so101":   # a fitted flat camera board on the gripper (real mounts vary)
            grip = next(b for b in root.iter("body") if b.get("name") == "gripper")
            ET.SubElement(grip, "geom", name="wrist_board", type="box", group="2", contype="0", conaffinity="0",
                          size=" ".join(map(str, wrist_board.get("size", (0.0215, 0.0215, 0.006)))), pos="0 0 0")
        vis = root.find("visual") if root.find("visual") is not None else ET.SubElement(root, "visual")
        g = vis.find("global") if vis.find("global") is not None else ET.SubElement(vis, "global")
        g.set("offwidth", str(max(int(math.ceil(w * PAD)), 640)))
        g.set("offheight", str(max(int(math.ceil(h * PAD)), 480)))
        wb = root.find("worldbody")
        base = wb.find("body")
        shift_xy, shift_z = _base_alignment(kind)
        bq = np.array([float(x) for x in base_quat.split()]) if isinstance(base_quat, str) else np.asarray(base_quat)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, bq)
        off = R.reshape(3, 3) @ np.array([shift_xy[0], shift_xy[1], shift_z])
        base.set("pos", " ".join(f"{v:.6f}" for v in np.asarray(base_pos, float) + off))
        base.set("quat", " ".join(f"{v:.8f}" for v in bq))
        for name in cams:
            mount = mounts.get(name, "world")
            body = MODELS[kind]["mount"].get(mount)
            parent = wb if mount == "world" else next(b for b in base.iter("body") if b.get("name") == body)
            ET.SubElement(parent, "camera", name=name, pos="0 0 1", xyaxes="1 0 0 0 1 0", fovy="45")
        self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, self.rh, self.rw)
        self.cid = {c: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in cams}
        self.colored = self._colored_geoms()
        self.qadr = [self.m.jnt_qposadr[j] for j in range(6)]
        self.full_w = 640

    def set_board(self, b) -> None:
        """b = (x, y, z, rx, ry, rz): board pose in the gripper frame (m, deg)."""
        gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "wrist_board")
        if gid < 0:
            return
        self.m.geom_pos[gid] = b[:3]
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.radians(b[3:6]), "xyz")
        self.m.geom_quat[gid] = q

    def _colored_geoms(self) -> np.ndarray:
        ids = []
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "wrist_board")
        if bid >= 0:
            ids.append(bid)
        for g in range(self.m.ngeom):
            if self.m.geom_group[g] != 2 or self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_MESH, self.m.geom_dataid[g]) or ""
            if mesh in self.parts.get("exclude", {}).get(self.kind, ()):
                continue
            if self.parts.get("servos"):
                ids.append(g)
                continue
            if self.kind == "so100":
                rgba = self.m.geom_rgba[g] if self.m.geom_matid[g] < 0 else self.m.mat_rgba[self.m.geom_matid[g]]
                if "Motor" in mesh or rgba[:3].max() < 0.2:
                    continue
            elif mesh.startswith("sts3215"):
                continue
            ids.append(g)
        return np.array(ids)

    def set_q(self, q: np.ndarray) -> None:
        for a, v in zip(self.qadr, q):
            self.d.qpos[a] = v
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)

    def set_cam(self, name: str, p: np.ndarray) -> None:
        cid = self.cid[name]
        self.m.cam_pos[cid] = p[:3]
        self.m.cam_quat[cid] = cam_quat(p[3], p[4], p[5])
        # padded pinhole canvas with the same focal length as the real image
        self.m.cam_fovy[cid] = 2 * math.degrees(math.atan(math.tan(math.radians(p[6]) / 2) * self.rh / self.h))
        mujoco.mj_camlight(self.m, self.d)
        self._p = getattr(self, "_p", {})
        self._p[name] = np.asarray(p, float)

    def _map(self, name: str) -> np.ndarray:
        p = self._p[name]
        key = (name, *np.round(p[6:], 6))
        if key not in self._maps:
            if len(self._maps) > 64:
                self._maps.clear()
            self._maps[key] = lens_map(p, self.w, self.h, self.rw, self.rh, self.full_w)
        return self._maps[key]

    def mask_pinhole(self, name: str) -> np.ndarray:
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, camera=name)
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        return np.isin(seg[..., 0], self.colored)

    def mask(self, name: str) -> np.ndarray:
        """Twin arm coverage (float 0..1) on the REAL image grid: padded pinhole render warped through the lens."""
        return ndi.map_coordinates(self.mask_pinhole(name).astype(float), self._map(name), order=1, mode="constant")

    def rgb(self, name: str) -> np.ndarray:
        self.r.update_scene(self.d, camera=name)
        img = self.r.render()
        mp = self._map(name)
        return np.clip(np.stack([ndi.map_coordinates(img[..., c].astype(float), mp, order=1, mode="nearest")
                                 for c in range(3)], -1), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------- real side
def hsv_mask(rgb: np.ndarray, ranges) -> np.ndarray:
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    out = np.zeros(rgb.shape[:2], bool)
    for (h0, h1), (s0, s1), (v0, v1) in ranges:
        hh = (hsv[..., 0] >= h0) & (hsv[..., 0] <= h1) if h0 <= h1 else (hsv[..., 0] >= h0) | (hsv[..., 0] <= h1)
        out |= hh & (hsv[..., 1] >= s0) & (hsv[..., 1] <= s1) & (hsv[..., 2] >= v0) & (hsv[..., 2] <= v1)
    return out


def clean(mask: np.ndarray, min_px: int, close: int = 1, link: int = 0, keep: str = "edge", fill: int = 0) -> np.ndarray:
    """Drop small blobs (fruit rims, specks). With link > 0, keep only blobs that join (after dilating by `link` px,
    which bridges the black servo gaps between coloured parts) a group touching the image border: the arm always
    enters the frame from its base, loose objects do not."""
    m = ndi.binary_opening(mask, iterations=1)
    lab, n = ndi.label(ndi.binary_closing(m, iterations=close) if close else m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = m & np.isin(lab, 1 + np.flatnonzero(sizes >= min_px))
    if link:
        grp, ng = ndi.label(ndi.binary_dilation(m, iterations=link))
        if keep == "edge":
            sel = np.unique(np.concatenate([grp[0], grp[-1], grp[:, 0], grp[:, -1]]))
            sel = sel[sel > 0]
        else:   # "largest": the biggest linked group (the arm with its base in frame) plus anything entering from
            # the image edge (a gripper reaching in from outside the frame)
            gs = ndi.sum(m, grp, range(1, ng + 1)) if ng else np.array([])
            sel = [int(np.argmax(gs)) + 1] if ng else []
            if ng:
                e = np.unique(np.concatenate([grp[0], grp[-1], grp[:, 0], grp[:, -1]]))
                sel = list(set(sel) | {int(x) for x in e if x > 0 and gs[x - 1] >= min_px})
        m = m & np.isin(grp, sel)
    if fill:   # fill enclosed holes smaller than `fill` px (stickers, highlights on a dark arm)
        holes, nh = ndi.label(~m)
        if nh:
            hs = ndi.sum(~m, holes, range(1, nh + 1))
            m = m | np.isin(holes, 1 + np.flatnonzero(hs < fill))
    return m


def area_down(mask: np.ndarray, k: int) -> np.ndarray:
    H, W = mask.shape
    return mask[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k).mean((1, 3))


PAD = 1.4


def lens_map(p, w: int, h: int, rw: int, rh: int, full_w: int) -> np.ndarray:
    """For every real pixel (h x w grid at this resolution): where it lands on the padded (rh x rw) pinhole canvas with
    the same focal length. Lens: x_d = x_n (1 + k1 r^2 + k2 r^4), principal point offset (cx, cy) in full-res px."""
    f = (h / 2) / math.tan(math.radians(p[6]) / 2)
    s = w / full_w
    vv, uu = np.mgrid[0:h, 0:w].astype(float)
    xd = (uu + 0.5 - w / 2 - p[9] * s) / f
    yd = (vv + 0.5 - h / 2 - p[10] * s) / f
    xn, yn = xd.copy(), yd.copy()
    for _ in range(25):
        r2 = xn * xn + yn * yn
        dd = np.maximum(1 + p[7] * r2 + p[8] * r2 * r2, 0.2)
        xn, yn = xd / dd, yd / dd
    return np.stack([f * yn + rh / 2 - 0.5, f * xn + rw / 2 - 0.5])


def resample_real(cov: np.ndarray, p: np.ndarray, full_w: int) -> np.ndarray:
    """Real coverage (h x w, possibly distorted lens) -> twin pinhole grid of the same size. k1,k2 radial; cx,cy
    principal point offset in full-res pixels."""
    h, w = cov.shape
    f = (h / 2) / math.tan(math.radians(p[6]) / 2)
    s = w / full_w
    v, u = np.mgrid[0:h, 0:w].astype(float)
    xn, yn = (u + 0.5 - w / 2) / f, (v + 0.5 - h / 2) / f
    r2 = xn * xn + yn * yn
    d = 1 + p[7] * r2 + p[8] * r2 * r2
    ur = f * xn * d + w / 2 + p[9] * s - 0.5
    vr = f * yn * d + h / 2 + p[10] * s - 0.5
    return ndi.map_coordinates(cov, [vr, ur], order=1, mode="constant", cval=0.0)


# ---------------------------------------------------------------------------------------------------- scene configs
def lookat_to_ypr(pos, lookat, roll: float = 0.0) -> list:
    """(pos, lookat, roll) camera convention -> [x, y, z, yaw, pitch, roll] in this file's convention."""
    f = np.asarray(lookat, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    return [*map(float, pos), math.degrees(math.atan2(f[1], f[0])), math.degrees(math.asin(f[2])), float(roll)]


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


def scene_config(name: str) -> dict:
    """fitkit/scene_<name>.py defines config() -> dict with keys:
    real (set name), base_pos, base_quat, real_to_sim(state) -> SO-100 Menagerie radians (6,),
    views: {view: dict(cam=<set camera>, init=[x,y,z,yaw,pitch,roll,fovy], mount='world'|'jaw'|'shoulder',
                       episodes=None|list, role='arm'|'wrist')},
    real_mask(rgb, view) -> bool HxW of the real arm's coloured (non-servo) parts."""
    mod = load_module(TWIN / "fitkit" / f"scene_{name}.py", f"fitkit_scene_{name}")
    return mod.config()


def raw_states(ds: pathlib.Path, episodes: set) -> dict:
    """{episode: (T, 6) observation.state} from the LeRobot parquet files (v2.1 per-episode files or v3.0 chunks)."""
    import pyarrow.parquet as pq

    ds = ds if ds.is_absolute() else ROOT / ds
    out: dict = {}
    for f in sorted(ds.glob("data/*/*.parquet")):
        if f.name.startswith("episode_") and int(f.stem.split("_")[1]) not in episodes:
            continue
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"])
        e = t.column("episode_index").to_numpy()
        fi = t.column("frame_index").to_numpy()
        st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(float)
        for k in np.unique(e):
            if int(k) in episodes:
                sel = e == k
                out[int(k)] = st[sel][np.argsort(fi[sel])]
    return out


# ---------------------------------------------------------------------------------------------------- fitter
class Fit:
    """One camera pose + lens per view (a dataset camera, optionally restricted to the episodes of one recording
    session). Joint offsets are shared by all views and fitted on 'arm' views only. Fit frames come from even
    episodes, held-out frames from odd episodes."""

    def __init__(self, cfg: dict, kind: str, k_fit: int = 2, n_fit: int = 24, n_eval: int = 48, seed: int = 0):
        from sets import Store

        self.cfg, self.kind = cfg, kind
        self.fs = Store().frames(cfg["real"])
        self.views = cfg["views"]
        H, W = self.fs.meta["image_shape"][:2]
        self.H, self.W, self.k = H, W, k_fit
        ep = np.asarray(self.fs.labels["episode"])
        rng = np.random.default_rng(seed)
        self.fit_idx, self.ev_idx = {}, {}
        for v, vc in self.views.items():
            ok = np.ones(self.fs.n, bool) if vc.get("episodes") is None else np.isin(ep, list(vc["episodes"]))
            fit_pool = np.flatnonzero(ok & (ep % 2 == 0))
            ev_pool = np.flatnonzero(ok & (ep % 2 == 1))
            self.fit_idx[v] = self._fps(fit_pool, n_fit)
            self.ev_idx[v] = np.sort(rng.choice(ev_pool, size=min(n_eval, len(ev_pool)), replace=False))
        self.ep, self.t = ep, np.asarray(self.fs.labels["t"])
        self.raw = raw_states(pathlib.Path(self.fs.meta["dataset"]) if not str(self.fs.meta["dataset"]).startswith("/")
                              else pathlib.Path(self.fs.meta["dataset"]), set(ep.tolist()))
        self.lag = 0.0
        self.set_lag(0.0)
        self.q_mean = np.mean([self.q_base[i] for i in range(self.fs.n)], axis=0)
        self.joint = dict(sign=np.ones(6), off=np.zeros(6), gain=np.zeros(6))
        if kind == "so101":
            self.joint["sign"][0] = -1.0
        mounts = {v: vc.get("mount", "world") for v, vc in self.views.items()}
        self.board = np.asarray(cfg.get("wrist_board", {}).get("init", [0, 0, 0, 0, 0, 0]), float)
        self.small = ArmScene(kind, cfg["base_pos"], cfg["base_quat"], list(self.views), W // k_fit, H // k_fit, mounts,
                              cfg.get("parts"), cfg.get("wrist_cam", False), cfg.get("wrist_board"))
        self.small.set_board(self.board)
        self.small.full_w = W
        self.full = None
        self.mounts = mounts
        self._real_cache: dict = {}
        self.p = {v: np.array([*vc["init"], 0.0, 0.0, 0.0, 0.0], float) for v, vc in self.views.items()}

    def set_lag(self, lag: float) -> None:
        """Render frame i with the joint state `lag` frames away (negative = earlier), linearly interpolated from the
        full-rate recording: the camera image trails the recorded joint reading by a few tens of ms."""
        self.lag = float(lag)
        self.q_base = {}
        for i in range(self.fs.n):
            S = self.raw.get(int(self.ep[i]))
            if S is None:
                st = self.fs.state[i]
            else:
                x = float(np.clip(self.t[i] + lag, 0, len(S) - 1))
                a = int(np.floor(x))
                b = min(a + 1, len(S) - 1)
                st = S[a] * (1 - (x - a)) + S[b] * (x - a)
            self.q_base[i] = np.asarray(self.cfg["real_to_sim"](st), float)

    def fit_lag(self, grid=np.arange(-4.0, 2.01, 0.25)) -> tuple[float, float]:
        views = self.arm_views()
        best = (np.inf, 0.0)
        for g in grid:
            self.set_lag(g)
            L = float(np.mean([self.loss(v, self.p[v]) for v in views]))
            if L < best[0]:
                best = (L, float(g))
        self.set_lag(best[1])
        return best[1], 1 - best[0]

    def arm_views(self):
        return [v for v, vc in self.views.items() if vc.get("role", "arm") == "arm"]

    def _fps(self, pool: np.ndarray, n: int) -> np.ndarray:
        st = self.fs.state[pool, :5].astype(float)
        z = (st - st.mean(0)) / (st.std(0) + 1e-6)
        chosen = [int(np.argmin(st[:, 1]))]
        d = np.linalg.norm(z - z[chosen[0]], axis=1)
        while len(chosen) < min(n, len(pool)):
            i = int(np.argmax(d))
            chosen.append(i)
            d = np.minimum(d, np.linalg.norm(z - z[i], axis=1))
        return pool[np.array(chosen)]

    def train_colour(self, view: str, erode: int = 4, dilate: int = 12, bins: int = 32) -> float:
        """Colour-only arm classifier for one view: an RGB lookup table P(arm | colour) from the fit frames, labelled
        by the twin at the current fit (arm = twin mask shrunk by `erode` px, background = farther than `dilate` px).
        Returns the fraction of training arm pixels it labels arm (recall on its own labels)."""
        sc = self._full()
        sc.set_cam(view, self.p[view])
        q = 256 // bins
        ha = np.zeros(bins ** 3)
        hb = np.zeros(bins ** 3)
        for i in self.fit_idx[view]:
            sc.set_q(self.q(i))
            sm = sc.mask(view) > 0.5     # already on the real grid
            if sm.sum() < 200:
                continue
            rgb = np.asarray(self.fs.obs(i)["images"][self.views[view]["cam"]])
            idx_pin = (rgb[..., 0] // q) * bins * bins + (rgb[..., 1] // q) * bins + rgb[..., 2] // q
            a = ndi.binary_erosion(sm, iterations=erode)
            b = ~ndi.binary_dilation(sm, iterations=dilate)
            ha += np.bincount(idx_pin[a], minlength=bins ** 3)
            hb += np.bincount(idx_pin[b], minlength=bins ** 3)
        pa = ha / max(ha.sum(), 1)
        pb = hb / max(hb.sum(), 1)
        prior = self.views[view].get("arm_prior", 0.5)
        lut = (pa * prior) / (pa * prior + pb * (1 - prior) + 1e-12)
        self.luts = getattr(self, "luts", {})
        self.luts[view] = (lut, q, bins)
        self._real_cache = {k: v for k, v in self._real_cache.items() if k[1] != view}
        return float((pa * (lut > 0.5)).sum())

    def sam_mask(self, i: int, view: str) -> np.ndarray:
        """Precomputed SAM mask (segment_arm.py --prompts) for frame i; all-false where SAM had no prompt."""
        self._sam = getattr(self, "_sam", {})
        if view not in self._sam:
            z = np.load(TWIN / "fitkit" / "masks" / f"{self.cfg['real']}__{view}.npz")
            bits = np.unpackbits(z["bits"], axis=-1)[..., : int(z["shape"][1])].astype(bool)
            self._sam[view] = {int(fr): bits[k] for k, fr in enumerate(z["frames"])}
        m = self._sam[view].get(int(i))
        if m is None:
            return np.zeros((self.H, self.W), bool)
        sub = self.views[view].get("subtract_hsv")
        if sub:   # loose objects SAM glued to the arm (a held ring): drop pixels of their colours
            rgb = np.asarray(self.fs.obs(int(i))["images"][self.views[view]["cam"]])
            m = m & ~ndi.binary_dilation(hsv_mask(rgb, sub), iterations=2)
        # SAM's low-res logits leave speckle holes on shiny / textured parts: close and fill holes < 1500 px
        m = ndi.binary_closing(m, iterations=2) | m
        holes, nh = ndi.label(~m)
        if nh:
            hs = ndi.sum(~m, holes, range(1, nh + 1))
            m = m | np.isin(holes, 1 + np.flatnonzero(hs < 1500))
        cl = self.views[view].get("sam_clean")   # colour-mask 'clean' settings do not apply to SAM masks
        return clean(m, min_px=cl.get("min_px", 250), link=cl.get("link", 0), keep=cl.get("keep", "edge"),
                     fill=cl.get("fill", 0)) if cl else m

    def real(self, i: int, view: str, k: int) -> np.ndarray:
        key = (i, view, k)
        if key not in self._real_cache:
            rgb = np.asarray(self.fs.obs(i)["images"][self.views[view]["cam"]])
            if self.cfg["real_mask"] == "sam":
                m = self.sam_mask(i, view)
            elif self.cfg["real_mask"] == "auto":
                lut, q, bins = self.luts[view]
                p = lut[(rgb[..., 0] // q) * bins * bins + (rgb[..., 1] // q) * bins + rgb[..., 2] // q]
                cl = self.views[view].get("clean", {})
                m = clean(p > 0.5, min_px=cl.get("min_px", 250), link=cl.get("link", 8), keep=cl.get("keep", "edge"),
                          fill=cl.get("fill", 0))
            else:
                m = self.cfg["real_mask"](rgb, view)
            for y0, y1, x0, x1 in self.views[view].get("exclude", ()):   # static clutter boxes (full-res px)
                m[y0:y1, x0:x1] = False
            self._real_cache[key] = area_down(m, k) if k > 1 else m.astype(float)
        return self._real_cache[key]

    def q(self, i: int, joint=None) -> np.ndarray:
        """sign * (gain about the dataset's mean pose) + offset; gain = 1 + j['gain']."""
        j = joint or self.joint
        qb = self.q_base[i]
        return j["sign"] * ((1 + j["gain"]) * (qb - self.q_mean) + self.q_mean) + j["off"]

    def ignore(self, view: str, k: int) -> np.ndarray | None:
        """Pixels scored in neither mask (real grid, coverage at 1/k): with view option ignore_static_dark=thr, the
        pixels that are darker than thr in >= 85% of the view's frames (static dark background: table edges, rails,
        bags). Both the real and the twin mask are cleared there, so parking the twin on dark clutter earns nothing."""
        thr = self.views[view].get("ignore_static_dark")
        stat = self.views[view].get("ignore_static_mask")   # fraction: pixels the raw real mask marks in >= it of frames
        boxes = self.views[view].get("ignore_boxes", ())    # (y0, y1, x0, x1) full-res, scored in neither mask
        if not thr and not stat and not boxes:
            return None
        self._ign = getattr(self, "_ign", {})
        if (view, k) not in self._ign:
            if (view, 1) not in self._ign and (stat or thr):
                eps = self.views[view].get("episodes")
                idx = [i for i in range(self.fs.n) if eps is None or int(self.ep[i]) in set(eps)]
                if stat:
                    raw = self.views[view]["raw_mask"]
                    frac = np.mean([raw(np.asarray(self.fs.obs(i)["images"][self.views[view]["cam"]])) for i in idx], 0)
                    self._ign[(view, 1)] = ndi.binary_dilation(frac >= stat, iterations=3).astype(float)
                else:
                    frac = np.mean([np.asarray(self.fs.obs(i)["images"][self.views[view]["cam"]]).max(-1) < thr for i in idx], 0)
                    self._ign[(view, 1)] = ndi.binary_dilation(frac >= 0.85, iterations=3).astype(float)
            if (view, 1) not in self._ign:
                self._ign[(view, 1)] = np.zeros((self.H, self.W))
            for y0, y1, x0, x1 in boxes:
                self._ign[(view, 1)][y0:y1, x0:x1] = 1.0
            g = self._ign[(view, 1)]
            self._ign[(view, k)] = area_down(g > 0.5, k) if k > 1 else g
        return self._ign[(view, k)]

    def ious(self, view: str, p: np.ndarray, idx, scene: ArmScene, k: int, soft: bool, joint=None) -> np.ndarray:
        """Per-frame IoU; NaN where neither mask has any arm pixel (the arm is out of view in both)."""
        scene.set_cam(view, p)
        ig = self.ignore(view, k)
        keep = None if ig is None else 1.0 - ig
        out = []
        for i in idx:
            scene.set_q(self.q(i, joint))
            sm = scene.mask(view)                 # twin, warped onto the real grid
            rm = self.real(i, view, k)            # real mask, untouched
            if keep is not None:
                sm, rm = sm * keep, rm * keep
            if soft:
                sm, rm = ndi.gaussian_filter(sm, 1.0), ndi.gaussian_filter(rm, 1.0)
                u = np.maximum(sm, rm).sum()
                out.append(np.minimum(sm, rm).sum() / u if u > 1 else np.nan)
            else:
                rb, sb = rm > 0.5, sm > 0.5
                u = (rb | sb).sum()
                out.append((rb & sb).sum() / u if u > 50 else np.nan)
        return np.array(out)

    def loss(self, view, p, joint=None) -> float:
        x = self.ious(view, p, self.fit_idx[view], self.small, self.k, True, joint)
        return 1.0 - float(np.nanmean(x)) if np.isfinite(x).any() else 1.0

    def fit_cam(self, view: str, free, n_random: int = 0, span=None, seed: int = 0, maxfev: int = 1500) -> float:
        p0 = self.p[view].copy()
        free = np.array(list(free))
        rng = np.random.default_rng(seed)
        best = (self.loss(view, p0), p0)
        if n_random:
            span = np.asarray(span)[: len(free)]
            for _ in range(n_random):
                p = p0.copy()
                p[free] += rng.uniform(-1, 1, len(free)) * span
                p[6] = np.clip(p[6], 20, 110)
                L = self.loss(view, p)
                if L < best[0]:
                    best = (L, p)
        world = self.mounts[view] == "world"
        step = np.array([0.02 if world else 0.005] * 3 + [2, 2, 1.5, 2, 0.05, 0.05, 6, 6])[free]

        def f(x):
            p = best[1].copy()
            p[free] = x
            if not 15 < p[6] < 120 or abs(p[7]) > 0.5 or abs(p[8]) > 0.3 or abs(p[9]) > 40 or abs(p[10]) > 40:
                return 1.0
            return self.loss(view, p)

        x0 = best[1][free]
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(initial_simplex=np.vstack([x0] + [x0 + np.eye(len(free))[j] * step[j] for j in range(len(free))]),
                                    xatol=1e-4, fatol=1e-5, maxfev=maxfev))
        p = best[1].copy()
        if res.fun < best[0]:
            p[free] = res.x
        self.p[view] = p
        return 1 - min(res.fun, best[0])

    def fit_offsets(self, joints=(0, 1, 2, 3, 4), gains: bool = False, maxfev: int = 600) -> float:
        """Joint offsets (and with gains=True per-joint gains) on the arm views, cameras fixed."""
        jl = list(joints)
        n = len(jl)
        views = self.arm_views()
        base_off, base_gain = self.joint["off"].copy(), self.joint["gain"].copy()

        def mk(x):
            j = dict(sign=self.joint["sign"], off=base_off.copy(), gain=base_gain.copy())
            j["off"][jl] += x[:n]
            if gains:
                j["gain"][jl] += x[n:]
            return j

        def f(x):
            lim_all = np.broadcast_to(np.asarray(self.cfg.get("gain_limit", 0.05), float), (6,)).copy()
            lim_all[5] = self.cfg.get("jaw_gain_limit", 0.6)   # gripper % -> jaw angle scale is rarely 1:1
            lim = lim_all[jl]
            if gains and np.any(np.abs(base_gain[jl] + x[n:]) > lim):
                return 1.0
            return float(np.mean([self.loss(v, self.p[v], mk(x)) for v in views]))

        x0 = np.zeros(2 * n if gains else n)
        step = np.r_[np.full(n, 0.05), np.full(n, 0.03)][: len(x0)]
        L0 = f(x0)
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(initial_simplex=np.vstack([x0] + [np.eye(len(x0))[k] * step[k] for k in range(len(x0))]),
                                    xatol=1e-4, fatol=1e-5, maxfev=maxfev))
        if res.fun < L0:
            self.joint = mk(res.x)
        return 1 - min(res.fun, L0)

    def fit_board(self, n_random: int = 200, maxfev: int = 500, seed: int = 0) -> float:
        """Pose of the fitted wrist board in the gripper frame (cameras and joints fixed): random search within
        +-6 cm / any orientation, then Nelder-Mead. Its size is fixed, so it cannot grow to cover errors."""
        if not self.cfg.get("wrist_board"):
            return float("nan")
        rng = np.random.default_rng(seed)
        views = self.arm_views()

        def f(b):
            self.small.set_board(b)
            return float(np.mean([self.loss(v, self.p[v]) for v in views]))

        best = (f(self.board), self.board.copy())
        for _ in range(n_random):
            b = np.r_[rng.uniform(-0.06, 0.06, 3), rng.uniform(-180, 180, 3)]
            L = f(b)
            if L < best[0]:
                best = (L, b)
        res = minimize(f, best[1], method="Nelder-Mead",
                       options=dict(initial_simplex=np.vstack([best[1]] + [best[1] + np.eye(6)[k] * [0.01, 0.01, 0.01, 15, 15, 15][k] for k in range(6)]),
                                    maxfev=maxfev, xatol=1e-4, fatol=1e-5))
        self.board = res.x if res.fun < best[0] else best[1]
        self.small.set_board(self.board)
        if self.full is not None:
            self.full.set_board(self.board)
        return 1 - min(res.fun, best[0])

    def choose_wrist_map(self) -> dict:
        """so101 only: pick roll sign / roll offset (0 or pi) / jaw sign by fit loss at the current cameras."""
        best = None
        for sr in (1, -1):
            for orr in (0.0, math.pi):
                for sg in (1, -1):
                    j = dict(sign=self.joint["sign"].copy(), off=self.joint["off"].copy(), gain=self.joint["gain"].copy())
                    j["sign"][4], j["off"][4], j["sign"][5] = sr, orr, sg
                    j["off"][5] = 0.0 if sg > 0 else 1.57
                    L = float(np.mean([self.loss(v, self.p[v], j) for v in self.views]))
                    if best is None or L < best[0]:
                        best = (L, j, (sr, orr, sg))
        self.joint = best[1]
        return dict(roll_sign=best[2][0], roll_off=best[2][1], jaw_sign=best[2][2], fit_iou=1 - best[0])

    def _full(self) -> ArmScene:
        if self.full is None:
            self.full = ArmScene(self.kind, self.cfg["base_pos"], self.cfg["base_quat"], list(self.views), self.W,
                                 self.H, self.mounts, self.cfg.get("parts"), self.cfg.get("wrist_cam", False),
                                 self.cfg.get("wrist_board"))
            self.full.full_w = self.W
        self.full.set_board(self.board)
        return self.full

    def evaluate(self) -> dict:
        out = {}
        for v in self.views:
            ev = self.ious(v, self.p[v], self.ev_idx[v], self._full(), 1, False)
            fi = self.ious(v, self.p[v], self.fit_idx[v], self._full(), 1, False)
            e = ev[np.isfinite(ev)]
            out[v] = dict(role=self.views[v].get("role", "arm"), heldout_n=int(len(e)),
                          heldout_mean=float(e.mean()) if len(e) else float("nan"),
                          heldout_median=float(np.median(e)) if len(e) else float("nan"),
                          heldout_min=float(e.min()) if len(e) else float("nan"),
                          heldout_frac_ge_080=float((e >= 0.8).mean()) if len(e) else float("nan"),
                          fit_mean=float(np.nanmean(fi)) if np.isfinite(fi).any() else float("nan"),
                          heldout_per_frame=[None if not np.isfinite(x) else round(float(x), 3) for x in ev])
        return out

    def sheet(self, view: str, path: pathlib.Path, n: int = 6) -> None:
        sc = self._full()
        sc.set_cam(view, self.p[view])
        rows = []
        idx = self.ev_idx[view]
        for i in idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]:
            sc.set_q(self.q(i))
            sm = sc.mask(view)
            sm = sm > 0.5
            rm = self.real(i, view, 1) > 0.5
            ov = np.zeros(sm.shape + (3,), np.uint8)
            ov[rm] = (210, 60, 60)
            ov[sm] = (60, 80, 230)
            ov[rm & sm] = (240, 240, 240)
            real = np.asarray(self.fs.obs(i)["images"][self.views[view]["cam"]])
            rows.append(np.concatenate([real, sc.rgb(view), ov], 1))
        img = Image.fromarray(np.concatenate(rows, 0))
        img.thumbnail((1500, 2400))
        img.save(path, quality=88)

    def load_state(self, path: pathlib.Path) -> None:
        """Start from a saved fit (same scene; the model must match)."""
        st = json.loads(pathlib.Path(path).read_text())
        assert st["model"] == self.kind, (st["model"], self.kind)
        for v, d in st["views"].items():
            if v in self.p:
                self.p[v] = np.array([d[k] for k in CAM_KEYS], float)
        self.joint = dict(sign=np.array(st["joint_sign"]), off=np.array(st["joint_off_rad"]),
                          gain=np.array(st.get("joint_gain", np.zeros(6))))
        self.set_lag(st.get("lag_frames", 0.0))
        if "board" in st:
            self.board = np.asarray(st["board"], float)
            self.small.set_board(self.board)

    def export_prompts(self, view: str, path: pathlib.Path, n_pos: int = 7, n_neg: int = 6, seed: int = 0) -> int:
        """SAM prompts for every frame of the view from the twin at the current fit, on the real (distorted) pixel
        grid: a box around the twin arm (+12%), positive points deep inside it, negative points well outside.
        Frames where the twin arm is (almost) out of view get no prompt."""
        rng = np.random.default_rng(seed)
        sc = self._full()
        sc.set_cam(view, self.p[view])
        eps = self.views[view].get("episodes")
        idx = [i for i in range(self.fs.n) if eps is None or int(self.ep[i]) in set(eps)]
        h, w = self.H, self.W
        pv = self.p[view]
        f = (h / 2) / math.tan(math.radians(pv[6]) / 2)
        rows = []
        for i in idx:
            sc.set_q(self.q(i))
            sm = sc.mask(view) > 0.5     # real grid
            if sm.sum() < 300:
                rows.append((i, None, None, None))
                continue
            dt_in = ndi.distance_transform_edt(sm)
            far = ~ndi.binary_dilation(sm, iterations=25)

            def to_real(yx):   # (row, col) on the real grid -> (x, y)
                return np.stack([yx[:, 1] + 0.5, yx[:, 0] + 0.5], 1).astype(float)

            # positives spread along the whole arm: farthest-point sampling over pixels >= 4 px inside the twin
            cand = np.argwhere(dt_in >= min(4.0, 0.6 * dt_in.max()))
            pos = [cand[int(np.argmax(dt_in[cand[:, 0], cand[:, 1]]))]]
            dmin = np.linalg.norm(cand - pos[0], axis=1)
            while len(pos) < min(n_pos, len(cand)):
                pos.append(cand[int(np.argmax(dmin))])
                dmin = np.minimum(dmin, np.linalg.norm(cand - pos[-1], axis=1))
            pos = np.array(pos)
            ys, xs = np.nonzero(sm)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            my, mx = 0.12 * (y1 - y0), 0.12 * (x1 - x0)
            ring = np.argwhere(far)
            near = ring[(ring[:, 0] > y0 - 3 * my) & (ring[:, 0] < y1 + 3 * my) & (ring[:, 1] > x0 - 3 * mx) & (ring[:, 1] < x1 + 3 * mx)]
            neg = near[rng.choice(len(near), size=min(n_neg, len(near)), replace=False)] if len(near) else np.zeros((0, 2), int)
            corners = to_real(np.array([[y0 - my, x0 - mx], [y1 + my, x1 + mx]]))
            box = np.clip([corners[0, 0], corners[0, 1], corners[1, 0], corners[1, 1]], 0, [w - 1, h - 1, w - 1, h - 1])
            rows.append((i, box, to_real(pos), to_real(neg)))
        np.save(path, np.array(rows, dtype=object), allow_pickle=True)
        return sum(r[1] is not None for r in rows)

    def state(self) -> dict:
        return dict(model=self.kind, lag_frames=self.lag, board=self.board.tolist(), views={v: dict(zip(CAM_KEYS, map(float, self.p[v])), cam=self.views[v]["cam"],
                                                    mount=self.mounts[v], episodes=self.views[v].get("episodes"))
                                            for v in self.views},
                    joint_sign=self.joint["sign"].tolist(), joint_off_rad=self.joint["off"].tolist(),
                    joint_gain=self.joint["gain"].tolist(), q_mean=self.q_mean.tolist(),
                    fit_frames={v: x.tolist() for v, x in self.fit_idx.items()},
                    heldout_frames={v: x.tolist() for v, x in self.ev_idx.items()})


def run(scene: str, kind: str, rounds: int, n_random: int, quick: bool = False, tag: str = "", init: str = "",
        overrides: dict | None = None, k_fit: int = 2, n_fit: int = 24) -> dict:
    cfg = scene_config(scene)
    cfg.update(overrides or {})
    t0 = time.time()
    F = Fit(cfg, kind, k_fit=k_fit, n_fit=n_fit)
    if init:
        F.load_state(pathlib.Path(init))
    name = f"{scene}_{kind}{tag}"
    log = lambda *a: print(f"[{name} {time.time() - t0:5.0f}s]", *a, flush=True)
    auto = cfg["real_mask"] == "auto"
    if auto:
        for v in F.views:
            log(v, "colour model (init cams) recall", round(F.train_colour(v), 3))
    for v in F.views:
        world = F.mounts[v] == "world"
        span = (0.12, 0.12, 0.12, 15, 12, 6, 12) if world else (0.02, 0.02, 0.02, 20, 20, 20, 12)
        log(v, "start soft IoU", round(1 - F.loss(v, F.p[v]), 3))
        if not init or v in (overrides or {}).get("refit_views", ()):
            log(v, "extrinsics+fovy", round(F.fit_cam(v, range(7), n_random=n_random, span=span), 3))
    if kind == "so101" and (not init or (overrides or {}).get("rewrist")):
        log("wrist map", F.choose_wrist_map())
    log("camera lag (frames, state shift)", F.fit_lag())
    if cfg.get("wrist_board"):
        log("wrist board", round(F.fit_board(), 3), np.round(F.board, 3).tolist())
    for r in range(rounds):
        if auto:
            for v in F.views:
                log(v, f"round {r} colour model recall", round(F.train_colour(v), 3))
        log(f"round {r} offsets+gains", round(F.fit_offsets(joints=(0, 1, 2, 3, 4, 5), gains=r > 0,
                                                            maxfev=600 if r == 0 else 1400), 3),
            "off deg", np.round(np.degrees(F.joint["off"]), 2).tolist(), "gain", np.round(F.joint["gain"], 3).tolist())
        for v in F.views:
            log(v, f"round {r} full lens", round(F.fit_cam(v, range(11), maxfev=400 if quick else 2500), 3))
    res = F.evaluate()
    st = F.state()
    st.update(eval=res, scene=scene)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(st, indent=1))
    for v in F.views:
        F.sheet(v, OUT / f"{name}_{v}.jpg")
        r = res[v]
        log(v, "[%s] HELD-OUT IoU mean %.3f median %.3f min %.3f frac>=0.8 %.2f n=%d (fit-set %.3f)" % (
            r["role"], r["heldout_mean"], r["heldout_median"], r["heldout_min"], r["heldout_frac_ge_080"],
            r["heldout_n"], r["fit_mean"]))
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--model", default="so101", choices=MODELS)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-random", type=int, default=150)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--init", default="", help="saved fit json to start from")
    ap.add_argument("--sam", action="store_true", help="use precomputed SAM masks (whole arm incl. servos)")
    ap.add_argument("--rewrist", action="store_true", help="re-choose the so101 roll/jaw map after --init")
    ap.add_argument("--refit-views", default="", help="comma list of views to re-search (random + NM) despite --init")
    ap.add_argument("--k-fit", type=int, default=2, help="fit-resolution divisor (1 = full resolution)")
    ap.add_argument("--n-fit", type=int, default=24, help="fit frames per view")
    a = ap.parse_args()
    ov = dict(real_mask="sam", parts=dict(servos=True)) if a.sam else {}
    if a.rewrist:
        ov["rewrist"] = True
    if a.refit_views:
        ov["refit_views"] = a.refit_views.split(",")
    run(a.scene, a.model, a.rounds, a.n_random, a.quick, a.tag, a.init, ov, a.k_fit, a.n_fit)
