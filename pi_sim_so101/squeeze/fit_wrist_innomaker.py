#!/usr/bin/env python3
"""Fit the two WRIST cameras of the sim to the real ones (user, 2026-11-13: all three cameras are InnoMaker
U20CAM-1080P-S1, a 130-degree wide-angle 32x32 mm UVC module; the rig file still described the wrists as Arducam IMX708
75-degree cameras, and the sim wrist view was far narrower than the real one).

Real data: sim-e5's logged real runs (squeeze/runs_clamp_v3/vla_real/*.npz: measured joints per tick + 320x240 frames of
all three cameras). Targets per wrist camera:
  - the JAWS: the gripper's own jaws never move in its wrist image, so pixels that are light and grey (V > 125, S < 45) in
    >= 60 % of that camera's real frames are jaw
  - the red CUBE in early frames (cube still on the table), placed in the sim where the CALIBRATED overhead camera sees it
    (vision.detect_cube_pose on the same tick's overhead frame); far away, so it pins down the field of view
Sim side: segmentation renders through the same distorted-lens model as pi_sim's render_camera. Fit (CMA-ES): the wrist
camera's mount offset (3 position, 3 small rotations) and the lens (fx = fy, cx, cy, k1; k2 from the overhead fit),
starting from the overhead's InnoMaker lens. Score: mean IoU of jaw masks + mean IoU of cube masks.

    python squeeze/fit_wrist_innomaker.py --side left --evals 400
Writes rig_tools/data/so101_camera_rig.innomaker_fit.json (overhead unchanged) + a before/after overlay image.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "rig_tools")]
RIG_IN = HERE.parents[1] / "rig_tools" / "data" / "so101_camera_rig.calibrated_full.json"
RIG_OUT = HERE.parents[1] / "rig_tools" / "data" / "so101_camera_rig.innomaker_fit.json"
# the user's camera model (2026-11-13): 130-deg-diagonal lens (103-106 deg H on the full 16:9 sensor), cropped to 4:3 for
# 640x480 -> 77-80 deg H, 58-60 deg V, square pixels, fisheye-style barrel. Equidistant r = f * theta with H = 78.5 deg
# -> f = 467.1 px; Brown-Conrady k1, k2 (k3 = 0, cube_locator reads k1, k2) least-squares fitted to that mapping over
# the whole 4:3 field (monotonic; see SPEC_ERR_PX).
SPEC_F, SPEC_K1, SPEC_K2, SPEC_K3 = 467.1, -0.3207, 0.1385, -0.0323     # max error 0.39 px vs equidistant
# alternative hypothesis: 640x480 = a 1280x960 window of the 1080p sensor scaled by 0.5 (common UVC mode) -> f = 524,
# 70.0 x 52.5 deg (87.5 diagonal); the real-data fits (overhead 67 x 50, left wrist 73 x 56) sit closer to this one
LENSES = {"spec": (SPEC_F, SPEC_K1, SPEC_K2, SPEC_K3), "crop960": (524.0, -0.3271, 0.1587, -0.0466)}
REAL = sorted(glob.glob(str(HERE / "runs_clamp_v3" / "vla_real" / "*.npz")))
SW, SH = 160, 120          # mask resolution for the IoU
ROI_PX = 6                 # real jaw pixels must lie within this of the old calibration's rendered jaws


def load_real(side):
    from PIL import Image
    import cube_locator as CL
    frames = []
    for f in REAL:
        z = np.load(f, allow_pickle=True)
        fi = z["frame_idx"]
        for k in range(len(z[f"jpg_{side}_wrist"])):
            ticks = np.where(fi == k)[0]
            if not len(ticks):
                continue
            t = int(ticks[0])
            w = np.asarray(Image.open(io.BytesIO(z[f"jpg_{side}_wrist"][k])).convert("RGB"))
            # logged frames are 320x240; the calibrated overhead model is for the 640x480 capture
            o = np.asarray(Image.open(io.BytesIO(z["jpg_overhead"][k])).convert("RGB").resize((640, 480), Image.Resampling.BILINEAR))
            frames.append(dict(file=Path(f).name, tick=t, q=z["state"][t].astype(float), wrist=w, overhead=o))
    # jaw mask: the jaws are the only thing that never moves in a wrist camera's view -> STATIC (low variation over
    # all frames), light grey in the median image, bottom half (a plain brightness rule missed the shaded jaws)
    small = np.stack([np.asarray(Image.fromarray(fr["wrist"]).resize((SW, SH))) for fr in frames]).astype(int)
    v = small.max(3)
    med = np.median(small, 0)
    mv, ms = med.max(2), med.max(2) - med.min(2)
    rows = np.arange(SH)[:, None] >= SH // 2
    jaw = (v.std(0) < 30) & (mv > 95) & (ms < 40) & rows
    for fr in frames:
        a = np.asarray(Image.fromarray(fr["wrist"]).resize((SW, SH))).astype(int)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        fr["red"] = (r > 55) & (r - g > 0.3 * r) & (r - b > 0.2 * r)
    return frames, jaw


class WristRenderer:
    def __init__(self, side):
        import mujoco
        import env as E
        E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}

        class S(E._Sim):
            def __init__(self, *a, **k):
                k["camera_rig"] = str(RIG_IN)
                super().__init__(*a, **k)
        E._Sim, old = S, E._Sim
        self.env = E.SqueezeEnv()
        E._Sim = old
        self.env.reset(seed=0)
        self.m, self.d, self.sim = self.env.model, self.env.data, self.env.sim
        self.side, self.name = side, f"{side}_wrist"
        self.cid = self.m.camera(self.name).id
        self.pos0, self.quat0 = self.m.cam_pos[self.cid].copy(), self.m.cam_quat[self.cid].copy()
        self.spec0 = self.sim.camera_specs[self.name]
        self.mj = mujoco
        self.ren = None
        bodies = {f"{side}_gripper", f"{side}_moving_jaw_so101_v1"}
        self.jaw_geoms = np.array([g for g in range(self.m.ngeom) if self.m.body(self.m.geom_bodyid[g]).name in bodies])
        self.cube_geom = self.m.geom("cube_geom").id
        self.cq = self.m.jnt_qposadr[self.m.joint("cube_joint").id]

    def apply(self, p):
        """p: dx dy dz (m), rx ry rz (rad, local small rotations), fx, cx, cy, k1."""
        mj = self.mj
        self.m.cam_pos[self.cid] = self.pos0 + p[0:3]
        dq = np.zeros(4)
        mj.mju_axisAngle2Quat(dq, p[3:6] / (np.linalg.norm(p[3:6]) + 1e-12), float(np.linalg.norm(p[3:6])))
        q = np.zeros(4)
        mj.mju_mulQuat(q, self.quat0, dq)
        self.m.cam_quat[self.cid] = q / np.linalg.norm(q)
        dist = list(self.spec0.distortion)
        dist[0] = p[9]
        # render directly at the mask resolution (intrinsics scaled to SW x SH): same projection, ~16x cheaper
        sc = SW / self.spec0.width
        self.spec = dataclasses.replace(self.spec0, width=SW, height=SH,
                                        intrinsics=(p[6] * sc, p[6] * sc, p[7] * sc, p[8] * sc), distortion=tuple(dist))

    def render_seg(self, q10, cube_xy_yaw=None):
        """(jaw mask, cube mask) at SWxSH for measured joints q10 (deg, 5 per arm) and an optional cube pose."""
        import cv2
        mj, m, d = self.mj, self.m, self.d
        for i, s in enumerate(("left", "right")):
            d.qpos[self.sim.robot[s].qpos_ids] = np.deg2rad(q10[i * 5:i * 5 + 5])
            j = m.joint(f"{s}_gripper")                     # jaws shut, as in every real clamp run
            d.qpos[j.qposadr[0]] = m.jnt_range[j.id][0] if m.jnt_limited[j.id] else 0.0
        if cube_xy_yaw is not None:
            x, y, yaw = cube_xy_yaw
            d.qpos[self.cq:self.cq + 7] = [x, y, 0.016, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        else:
            d.qpos[self.cq:self.cq + 3] = [0.0, 0.0, -1.0]
        mj.mj_forward(m, d)
        width, height, K, mapping = self.spec.render_plan()
        if self.ren is None or self.ren.width != width or self.ren.height != height:
            m.vis.global_.offwidth = max(width, m.vis.global_.offwidth)
            m.vis.global_.offheight = max(height, m.vis.global_.offheight)
            self.ren = mj.Renderer(m, height=height, width=width)
            self.ren.enable_segmentation_rendering()
        self.ren.update_scene(d, camera=self.name)
        for view in self.ren.scene.camera:
            near = view.frustum_near
            view.frustum_width = near * width / (2 * K[0, 0])
            view.frustum_center = near * (width / 2 - K[0, 2]) / K[0, 0]
            view.frustum_top = near * K[1, 2] / K[1, 1]
            view.frustum_bottom = -near * (height - K[1, 2]) / K[1, 1]
        seg = self.ren.render()[..., 0].astype(np.int32)                      # geom id per pixel (-1 none)
        if mapping is not None:
            seg = cv2.remap(seg.astype(np.float32), mapping[..., 0], mapping[..., 1], cv2.INTER_NEAREST).astype(np.int32)
        if seg.shape != (SH, SW):
            seg = cv2.resize(seg.astype(np.float32), (SW, SH), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return np.isin(seg, self.jaw_geoms), seg == self.cube_geom


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left")
    ap.add_argument("--evals", type=int, default=400)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--lens", default="free", choices=["free", "spec", "crop960"],
                    help="spec = the user's InnoMaker model held fixed: 130 deg lens cropped to 4:3 at 640x480, 78.5 x 58.8 "
                         "deg, square pixels, equidistant-style barrel (SPEC_LENS); only fx +-3 %, cx, cy +-20 px and the mount fit")
    ap.add_argument("--out", default=str(RIG_OUT))
    a = ap.parse_args()
    import cma
    import vision as V
    from PIL import Image

    frames, jaw_real = load_real(a.side)
    R = WristRenderer(a.side)
    cam = V.OverheadCamera.from_env(R.sim)
    # cube frames: early in each run, cube found on the overhead frame and red visible in the wrist frame
    cube_frames = []
    for fr in frames:
        if fr["tick"] > 120 or fr["red"].sum() < 15:
            continue
        pose = V.detect_cube_pose(fr["overhead"], cam)
        if pose is not None:
            fr["cube"] = pose
            cube_frames.append(fr)
    rng = np.random.default_rng(0)
    cube_frames = [cube_frames[i] for i in rng.permutation(len(cube_frames))[:a.frames]]
    # Real jaw mask, restricted to the fingers: "static + light" also caught the white floor beyond the table edge
    # (the free fit then grew sim geometry over it). The OLD calibration already places the sim jaws about right
    # (overlay 2026-11-13), so only static-light pixels within ROI_PX of where it renders the jaws count.
    import scipy.ndimage as ndi
    _s = R.spec0                                        # the old calibration, untouched so far
    R.apply(np.array([0, 0, 0, 0, 0, 0, _s.intrinsics[0], _s.intrinsics[2], _s.intrinsics[3], _s.distortion[0]]))
    roi = ndi.binary_dilation(R.render_seg(frames[0]["q"])[0], iterations=ROI_PX)
    jaw_real = jaw_real & roi
    print(f"jaw mask restricted to the fingers: {int(jaw_real.sum())} px", flush=True)
    jaw_frames = [frames[i] for i in rng.permutation(len(frames))[:8]]
    print(f"{a.side}: {len(frames)} real frames, jaw mask {int(jaw_real.sum())} px, {len(cube_frames)} cube frames", flush=True)

    ov = json.loads(RIG_IN.read_text())["cameras"]["overhead"]
    s0 = R.spec0
    if a.lens in LENSES:
        lf, lk1, lk2, lk3 = LENSES[a.lens]
        base_dist = (0.0, lk2, 0.0, 0.0, lk3)
        p0 = np.array([0, 0, 0, 0, 0, 0, lf, 320.0, 240.0, lk1])
        lo = np.array([-0.03, -0.03, -0.03, -0.45, -0.45, -0.45, lf * 0.97, 300.0, 220.0, lk1 - 1e-6])
        hi = np.array([0.03, 0.03, 0.03, 0.45, 0.45, 0.45, lf * 1.03, 340.0, 260.0, lk1 + 1e-6])
    else:
        base_dist = (0.0, ov["distortion"][1], 0.0, 0.0, 0.0)        # k2 = the overhead fit's
        p0 = np.array([0, 0, 0, 0, 0, 0, ov["intrinsics"]["fx"], 320.0, 240.0, ov["distortion"][0]])
        lo = np.array([-0.03, -0.03, -0.03, -0.45, -0.45, -0.45, 380.0, 260.0, 180.0, -1.3])
        hi = np.array([0.03, 0.03, 0.03, 0.45, 0.45, 0.45, 800.0, 380.0, 300.0, -0.3])

    def unpack(x):
        return lo + (hi - lo) * (np.tanh(np.asarray(x)) + 1) / 2

    def pack(p):
        return np.arctanh(np.clip(2 * (p - lo) / (hi - lo) - 1, -0.97, 0.97))

    # also keep the previous (Arducam-assumption) fit's lens for the "before" score
    old = np.array([0, 0, 0, 0, 0, 0, s0.intrinsics[0], s0.intrinsics[2], s0.intrinsics[3], s0.distortion[0]])
    R.spec0 = dataclasses.replace(s0, distortion=base_dist)

    def score(p):
        R.apply(p)
        jj = np.mean([iou(R.render_seg(fr["q"])[0], jaw_real) for fr in jaw_frames])
        cc = np.mean([iou(R.render_seg(fr["q"], fr["cube"])[1], fr["red"]) for fr in cube_frames]) if cube_frames else 0.0
        return jj, cc

    R.spec0 = s0
    jb, cb = score(old)
    R.spec0 = dataclasses.replace(s0, distortion=base_dist)
    print(f"before (current rig file): jaw IoU {jb:.3f}, cube IoU {cb:.3f}", flush=True)
    j0, c0 = score(p0)
    print(f"start (overhead's InnoMaker lens, old mount): jaw IoU {j0:.3f}, cube IoU {c0:.3f}", flush=True)
    es = cma.CMAEvolutionStrategy(pack(p0), 0.35, {"maxfevals": a.evals, "popsize": 14, "seed": 3, "verbose": -9})
    best = (-(j0 + c0), pack(p0))
    while not es.stop():
        xs = es.ask()
        fs = []
        for x in xs:
            jj, cc = score(unpack(x))
            fs.append(-(jj + cc))
        es.tell(xs, fs)
        for x, f in zip(xs, fs):
            if f < best[0]:
                best = (f, np.asarray(x))
        print(f"  it {es.countiter}: best jaw+cube {-best[0]:.3f}", flush=True)
    p = unpack(best[1])
    jf, cf = score(p)
    print(f"fitted: jaw IoU {jf:.3f}, cube IoU {cf:.3f} | pos offset {np.round(p[:3] * 1000, 1)} mm, rot "
          f"{np.round(np.degrees(p[3:6]), 1)} deg, fx {p[6]:.1f}, cx {p[7]:.1f}, cy {p[8]:.1f}, k1 {p[9]:.3f}", flush=True)

    # write the fitted camera into the new rig file (start from it if it exists, so the other wrist is kept)
    out_path = Path(a.out)
    rig = json.loads((out_path if out_path.exists() else RIG_IN).read_text())
    c = rig["cameras"][f"{a.side}_wrist"]
    R.apply(p)
    c["pose"]["pos"] = [round(float(v), 6) for v in R.m.cam_pos[R.cid]]
    c["pose"]["quat_wxyz"] = [round(float(v), 6) for v in R.m.cam_quat[R.cid]]
    c["intrinsics"] = {"fx": round(float(p[6]), 2), "fy": round(float(p[6]), 2), "cx": round(float(p[7]), 2),
                       "cy": round(float(p[8]), 2)}
    c["distortion"] = [round(float(p[9]), 4), base_dist[1], 0, 0, base_dist[4]]
    c["hardware_model"] = "InnoMaker U20CAM-1080P-S1 (130-degree wide-angle, 32x32 mm UVC module), same as the overhead"
    c["calibration_status"] = (f"fitted 2026-11-13 by squeeze/fit_wrist_innomaker.py on {len(frames)} real frames: jaw IoU "
                               f"{jf:.3f} (before {jb:.3f}), cube IoU {cf:.3f} (before {cb:.3f}); lens started from the "
                               f"overhead's fit, k2 shared with it")
    c["optics_note"] = "InnoMaker U20CAM-1080P-S1, 130-degree lens, captured at 640x480 like the overhead."
    rig["note"] = ("All three cameras are InnoMaker U20CAM-1080P-S1 (user, 2026-11-13). Overhead: calibrated 2026-11-10 "
                   "(unchanged). Wrists: refitted 2026-11-13 from real frames (squeeze/fit_wrist_innomaker.py).")
    c["lens_model"] = a.lens
    out_path.write_text(json.dumps(rig, indent=1))
    print("wrote", out_path, flush=True)

    # before / after overlay on 3 cube frames: real | old sim | fitted sim (jaw = green, cube = magenta outlines)
    rows = []
    for fr in cube_frames[:3]:
        real = np.asarray(Image.fromarray(fr["wrist"]).resize((SW * 2, SH * 2)))
        row = [real]
        for params, spec in ((old, s0), (p, dataclasses.replace(s0, distortion=base_dist))):
            R.spec0 = spec
            R.apply(params)
            jm, cm = R.render_seg(fr["q"], fr["cube"])
            img = real.copy()
            for mask, col in ((jm, (0, 255, 0)), (cm, (255, 0, 255))):
                big = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((SW * 2, SH * 2))) > 0
                img[big] = (0.5 * img[big] + 0.5 * np.array(col)).astype(np.uint8)
            row.append(img)
        rows.append(np.concatenate(row, 1))
    out = HERE / "runs_vla" / f"wrist_fit_{a.side}_{a.lens}.png"
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print("overlay (real | before | after):", out, flush=True)


if __name__ == "__main__":
    main()
