"""Joint fit of the real overhead camera + arm spacing + one arm's joint offsets, in the 232 mm mount layout.

Uses the Mac sim copy (sim/pi_sim_so101, TheRobotStudio overhead-mount layout) instead of the laptop's 360 mm
pi_sim, and a real camera model instead of the rig's 2-D scale/shift warp:
  camera: overhead mount body pose (3 pos + 3 rot) + focal length + barrel distortion k1, k2 + principal point
  arms:   half-spacing between the two shoulder-pan axes (CAD: 0.116 m)  -> checks the 232 mm figure
  joints: degree corrections for --side (added to the logged units, which already include rig.json offsets)
Score: mean IoU between the real white-plastic mask (tuned for the dim bench: V > 140, S < 50, 3x3 open, off-table
polygons from rig.json removed) and the sim's printed-part mask (segmentation render of the yellow arm geoms).

    sim/pi_sim_so101/.venv/bin/python rig_tools/fit_overhead_232.py rig_tools/data/right_robot --side right
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIM = Path(__file__).resolve().parents[1] / "pi_sim_so101"
sys.path.insert(0, str(SIM))
import mujoco                                   # noqa: E402
from pi_sim.env import RobotEnv                 # noqa: E402

W, H = 640, 480            # real image
K = 4                      # fit resolution = real / K
RW, RH, RFOVY = 400, 300, 110.0   # pinhole render: wide enough to cover the distorted field
SIDES = ("left", "right")
ARM_PARTS = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1", "camera_mount")


def real_mask(rgb):
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(float)
    m = ((hsv[..., 2] > 140) & (hsv[..., 1] < 50)).astype(np.uint8) * 255
    m = Image.fromarray(m).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    return np.asarray(m) > 0


def offtable(polys):
    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)
    for p in polys:
        dr.polygon([tuple(map(float, q)) for q in p], fill=255)
    return np.asarray(img) > 0


def down(mask):
    return mask.reshape(H // K, K, W // K, K).mean(axis=(1, 3)) > 0.5


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rotvec_quat(v):
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.array([1.0, 0, 0, 0])
    return np.r_[np.cos(a / 2), np.sin(a / 2) * v / a]


class Model:
    def __init__(self):
        self.env = RobotEnv(task="pick_place", control_mode="joint", object_color="red", object_shape="cube", control_hz=50)
        self.env.reset(seed=0, options={"randomize": False})
        m, d = self.m, self.d = self.env.model, self.env.data
        d.joint("cube_joint").qpos[:3] = [2, 2, 0]
        d.mocap_pos[0] = [2, 2, 0]
        self.cam = m.camera("overhead").id
        m.cam_fovy[self.cam] = RFOVY
        self.mount = m.body("overhead_camera_mount").id
        self.pos0, self.quat0 = m.body_pos[self.mount].copy(), m.body_quat[self.mount].copy()
        self.ped = {s: m.body(f"{s}_pedestal").id for s in SIDES}
        self.ped_y = m.body_pos[self.ped["left"]][1]
        self.ped_q0 = {s: m.body_quat[self.ped[s]].copy() for s in SIDES}
        names = {f"{s}_{p}" for s in SIDES for p in ARM_PARTS}
        self.arm_geom = np.zeros(m.ngeom + 1, bool)
        for g in range(m.ngeom):
            rgba = m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]]
            light = rgba[0] > 0.7 and rgba[1] > 0.6
            body = m.body(m.geom_bodyid[g]).name
            if light and m.geom_group[g] <= 2 and (body in names or m.geom(g).name in ("mount_left_arm_base", "mount_right_arm_base")):
                self.arm_geom[g] = True
        self.r = mujoco.Renderer(m, height=RH, width=RW)
        self.r.enable_segmentation_rendering()
        self.fp = (RH / 2) / np.tan(np.deg2rad(RFOVY) / 2)
        v, u = np.mgrid[0:H // K, 0:W // K]
        self.uu, self.vv = (u * K + K / 2).astype(float), (v * K + K / 2).astype(float)

    def set_pose(self, units12):
        m, d = self.m, self.d
        for i, s in enumerate(SIDES):
            arm = self.env.robot[s]
            d.qpos[arm.qpos_ids] = np.deg2rad(units12[i * 6:i * 6 + 5])
            lo, hi = m.actuator_ctrlrange[arm.gripper_actuator]
            d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = lo + float(units12[i * 6 + 5]) / 100 * (hi - lo)

    def masks(self, states, cam, half_x):
        """cam = dict(dpos(3) m, drot(3) rad, f, k1, k2, cx, cy). Returns sim arm masks at fit resolution."""
        m, d = self.m, self.d
        m.body_pos[self.mount] = self.pos0 + cam["dpos"]
        m.body_quat[self.mount] = quat_mul(self.quat0, rotvec_quat(cam["drot"]))
        for s, sx in (("left", -half_x), ("right", half_x)):
            dx, dy, dyaw = cam.get("base", {}).get(s, (0.0, 0.0, 0.0))
            m.body_pos[self.ped[s]][0], m.body_pos[self.ped[s]][1] = sx + dx, self.ped_y + dy
            m.body_quat[self.ped[s]] = quat_mul(self.ped_q0[s], rotvec_quat(np.array([0.0, 0.0, dyaw])))
        # real (distorted) pixel -> normalized -> undistort (fixed point) -> pinhole render pixel
        xd, yd = (self.uu - cam["cx"]) / cam["f"], (self.vv - cam["cy"]) / cam["f"]
        xu, yu = xd.copy(), yd.copy()
        for _ in range(4):
            r2 = xu * xu + yu * yu
            s = 1 + cam["k1"] * r2 + cam["k2"] * r2 * r2
            xu, yu = xd / s, yd / s
        pu = np.round(xu * self.fp + RW / 2).astype(int)
        pv = np.round(yu * self.fp + RH / 2).astype(int)
        ok = (pu >= 0) & (pu < RW) & (pv >= 0) & (pv < RH)
        out = []
        for st in states:
            self.set_pose(st)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)          # camera poses: mj_kinematics alone leaves cam_xpos stale
            self.r.update_scene(d, camera=self.cam)
            seg = self.r.render()
            g = seg[..., 0].astype(int)
            arm = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & self.arm_geom[np.clip(g, 0, m.ngeom)]
            mk = np.zeros(pu.shape, bool)
            mk[ok] = arm[pv[ok], pu[ok]]
            out.append(mk)
        return out


def unpack(x, si_active=None):
    """x: 12 camera/spacing params, then 5 left + 5 right joint corrections (2 deg per unit)."""
    cam = dict(dpos=x[0:3] * 0.01, drot=np.deg2rad(x[3:6]), f=417.0 * np.exp(x[6] * 0.1), k1=x[7] * 0.1 - 0.15,
               k2=x[8] * 0.05, cx=320 + x[9] * 10, cy=240 + x[10] * 10)
    half_x = 0.116 + x[11] * 0.01
    corr = np.zeros(12)
    corr[0:5], corr[6:11] = x[12:17] * 2.0, x[17:22] * 2.0     # degrees, in 12-value state layout
    if len(x) >= 28:                                            # per-arm base: dx, dy (cm), yaw (deg)
        cam["base"] = {s: (x[22 + 3 * i] * 0.01, x[23 + 3 * i] * 0.01, np.deg2rad(x[24 + 3 * i] * 2.0))
                       for i, s in enumerate(SIDES)}
    return cam, half_x, corr


def pack_from_json(res, model):
    """Warm start: camera / spacing / right-arm correction from a previous fit json (left correction reset to 0)."""
    x = np.zeros(28)
    c = res["camera"]
    x[0:3] = (np.array(c["mount_pos"]) - model.pos0) / 0.01
    q0inv = model.quat0 * np.array([1, -1, -1, -1])
    dq = quat_mul(q0inv, np.array(c["mount_quat"]))
    dq = dq / np.linalg.norm(dq) * np.sign(dq[0])
    ang = 2 * np.arccos(np.clip(dq[0], -1, 1))
    x[3:6] = np.rad2deg(dq[1:] / max(np.sin(ang / 2), 1e-9) * ang) if ang > 1e-9 else 0.0
    x[6] = 10 * np.log(c["f_px"] / 417.0)
    x[7], x[8] = (c["k1"] + 0.15) / 0.1, c["k2"] / 0.05
    x[9], x[10] = (c["cx"] - 320) / 10, (c["cy"] - 240) / 10
    x[11] = (res["arm_spacing_mm"] / 2000 - 0.116) / 0.01
    x[17:22] = np.array(res["joint_corr_deg"]["right"]) / 2.0
    return x


class Ctx:
    """Everything a scorer needs; one per worker process (each with its own MuJoCo renderer)."""

    def __init__(self, runs, rig_path, fix_spacing):
        rig = json.loads(Path(rig_path).read_text())
        off = offtable(rig.get("overhead_mask", []))
        self.rig, self.fix_spacing, self.ev, self.frames = rig, fix_spacing, [], []
        for run in runs:
            for l in open(Path(run) / "events.jsonl", encoding="utf-8"):
                e = json.loads(l)
                self.ev.append(e)
                f2 = Path(run) / "frames" / f"c{e['chunk']:02d}_raw_overhead.jpg"
                self.frames.append(f2 if f2.exists() else Path(run) / "frames" / f"c{e['chunk']:03d}_raw_overhead.jpg")
        self.states = [np.array(e["state_in"], float) for e in self.ev]
        self.valid = down(~off)
        self.reals = [down(real_mask(np.asarray(Image.open(f).convert("RGB"))) & ~off) for f in self.frames]
        self.M = Model()

    def params(self, x):
        cam, half_x, corr = unpack(x)
        return cam, (self.fix_spacing if self.fix_spacing is not None else half_x), corr

    def score(self, x, idx=None):
        """Mean IoU over frames `idx` (default all). Frames are interleaved per run, so even/odd = fit/held-out."""
        cam, half_x, corr = self.params(x)
        idx = range(len(self.states)) if idx is None else idx
        ious = []
        sims = self.M.masks([self.states[i] + corr for i in idx], cam, half_x)
        for i, sim in zip(idx, sims):
            real, sim = self.reals[i], sim & self.valid
            u = (real | sim).sum()
            ious.append((real & sim).sum() / u if u else 0.0)
        return float(np.mean(ious)), ious


_CTX = None
_FIT_IDX = None


def _init(runs, rig_path, fix_spacing, fit_idx):
    global _CTX, _FIT_IDX
    _CTX, _FIT_IDX = Ctx(runs, rig_path, fix_spacing), fit_idx


def _score(x):
    return _CTX.score(np.asarray(x), _FIT_IDX)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--side", default="both", help="label only; both arms' corrections are always fitted")
    ap.add_argument("--rig", type=Path, default=Path(__file__).with_name("data") / "rig.json")
    ap.add_argument("--evals", type=int, default=2500)
    ap.add_argument("--fix-spacing", type=float, default=None, help="hold the half-spacing at this value (m)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--per-arm-base", action="store_true", help="fit each arm's base dx, dy, yaw too")
    ap.add_argument("--init", type=Path, default=None, help="warm start from a previous fit json")
    ap.add_argument("--sigma", type=float, default=0.6)
    a = ap.parse_args()
    import multiprocessing as mp
    import cma

    C = Ctx(a.runs, a.rig, a.fix_spacing)
    rig, ev, frames, states, reals, valid, M = C.rig, C.ev, C.frames, C.states, C.reals, C.valid, C.M
    a.run = a.out or a.runs[0]
    a.run.mkdir(parents=True, exist_ok=True)
    score = C.score
    fit_idx = list(range(0, len(states), 2))                    # even frames fit, odd frames held out
    test_idx = list(range(1, len(states), 2))
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init,
                                       initargs=([str(r) for r in a.runs], str(a.rig), a.fix_spacing, fit_idx))

    x0 = np.zeros(28 if a.per_arm_base else 22)
    if a.init:
        x0[:22] = pack_from_json(json.loads(a.init.read_text()), M)[:22]
    s0, _ = score(x0, fit_idx)
    print(f"{len(states)} poses ({len(fit_idx)} fit / {len(test_idx)} held out); start: fit-set IoU {s0:.3f}", flush=True)
    t0 = time.time()
    es = cma.CMAEvolutionStrategy(x0, a.sigma, {"maxfevals": a.evals, "verbose": -9, "seed": 1, "popsize": 20})
    best = (s0, x0)
    while not es.stop():
        xs = es.ask()
        ss = pool.map(_score, [np.asarray(x) for x in xs])
        fs = [-s for s in ss]
        for x, s in zip(xs, ss):
            if s > best[0]:
                best = (s, np.asarray(x).copy())
        es.tell(xs, fs)
        if es.countiter % 10 == 0:
            cam, hx, corr = unpack(best[1])
            print(f"  it {es.countiter:4d} evals {es.countevals:5d} ({time.time() - t0:5.0f}s): IoU {best[0]:.3f} | "
                  f"spacing {2 * (a.fix_spacing or hx) * 1000:5.1f} mm | L " + " ".join(f"{c:+5.1f}" for c in corr[0:5]) +
                  " | R " + " ".join(f"{c:+5.1f}" for c in corr[6:11]), flush=True)
    s, x = best
    cam, hx, corr = unpack(x)
    if a.fix_spacing is not None:
        hx = a.fix_spacing
    _, ious = score(x)
    held_out, _ = score(x, test_idx)
    print(f"held-out IoU {held_out:.3f} (fit set {s:.3f})", flush=True)
    old = {sd: np.array(rig["joint_offsets_deg"][sd], float) for sd in SIDES}
    new = {sd: old[sd].copy() for sd in SIDES}
    new["left"][:5] += corr[0:5]
    new["right"][:5] += corr[6:11]
    n_by_run = np.cumsum([0] + [sum(1 for _ in open(r / "events.jsonl")) for r in a.runs])
    res = {"runs": [str(r) for r in a.runs], "fit_iou": s, "held_out_iou": held_out, "x": np.asarray(x).tolist(),
           "base_dx_dy_yaw": {sd: [round(float(v), 4) for v in cam.get("base", {}).get(sd, (0, 0, 0))] for sd in SIDES},
           "mean_iou": float(np.mean(ious)), "median_iou": float(np.median(ious)),
           "p10_iou": float(np.percentile(ious, 10)),
           "mean_iou_per_run": [float(np.mean(ious[n_by_run[i]:n_by_run[i + 1]])) for i in range(len(a.runs))],
           "arm_spacing_mm": 2 * hx * 1000,
           "joint_corr_deg": {"left": corr[0:5].round(2).tolist(), "right": corr[6:11].round(2).tolist()},
           "joint_offsets_old": {sd: old[sd].tolist() for sd in SIDES},
           "joint_offsets_new": {sd: new[sd].round(1).tolist() for sd in SIDES},
           "camera": {"mount_pos": (M.pos0 + cam["dpos"]).round(4).tolist(),
                      "mount_quat": quat_mul(M.quat0, rotvec_quat(cam["drot"])).round(5).tolist(),
                      "f_px": round(cam["f"], 1), "k1": round(cam["k1"], 4), "k2": round(cam["k2"], 4),
                      "cx": round(cam["cx"], 1), "cy": round(cam["cy"], 1)},
           "per_frame_iou": [round(v, 3) for v in ious], "run": str(a.run)}
    out = a.run / f"fit232_{a.side}{'_fixsp' if a.fix_spacing else ''}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame_iou"}, indent=1))

    # overlay sheet: 8 frames, white both / red real only / blue sim only
    st = [s_ + corr for s_ in states]
    sims = M.masks(st, cam, hx)
    idx = np.linspace(0, len(st) - 1, 8).astype(int)
    tiles = []
    for i in idx:
        rgb = np.asarray(Image.open(frames[i]).convert("RGB").resize((W // K, H // K)))
        ov = (rgb // 3).copy()
        r_, s_ = reals[i], sims[i] & valid
        ov[r_ & s_] = 255
        ov[r_ & ~s_] = (230, 50, 50)
        ov[s_ & ~r_] = (50, 110, 240)
        tiles.append(np.concatenate([rgb, ov], 0))
    sheet = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)
    Image.fromarray(sheet).save(a.run / f"fit232_{a.side}{'_fixsp' if a.fix_spacing else ''}.jpg", quality=88)
    print("overlay ->", a.run / f"fit232_{a.side}{'_fixsp' if a.fix_spacing else ''}.jpg")


if __name__ == "__main__":
    main()
