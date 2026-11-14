#!/usr/bin/env python3
"""Cube check for a camera rig (robots/so101/calibration/README.md step 3: "replay real arm poses in the sim; the sim
cube in each wrist view should be about 1.0x the real cube's area and within a few pixels of it").

For logged REAL policy runs (C:\\robot\\real\\runs\\real_policy_*: log.json joints per tick, frames every 5th tick =
overhead | left_wrist | right_wrist at 320x240): frames where the cube is still untouched at the spot the runner found
it at the start (both fingertips > 5 cm away) are replayed in the sim -- arm joints = the measured ones, cube at that
spot with the yaw the overhead sees -- and both wrist cameras are rendered with the candidate rig. The red blob
(cube_locator.detect_red, the rig's own detector) is compared: area ratio sim/real and centroid offset (640x480 px).

    python squeeze/rig_cube_check.py --rig <rig.json> --runs <folder with real_policy_* runs> [--examples out.jpg]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "rig_tools")]
WRISTS = ("left_wrist", "right_wrist")


def detect_red_merged(rgb, min_px=30):
    """cube_locator.detect_red's colour rule, but a dark cable crossing the cube no longer splits it (largest-blob kept
    one half: 7.7-9.2 mm off, inference-testing-04's audit): binary closing + hole fill before the largest blob."""
    import scipy.ndimage as ndi
    a = np.asarray(rgb).astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r > 55) & (r - g > 0.3 * r) & (r - b > 0.2 * r)
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=3))
    lab, n = ndi.label(mask)
    if n == 0:
        return None, 0
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    npx = int(sizes[k - 1])
    if npx < min_px:
        return None, npx
    vs, us = np.nonzero(lab == k)
    return (float(us.mean()), float(vs.mean())), npx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--examples", default=None)
    ap.add_argument("--max-tick", type=int, default=200)
    # arm kinematics that go with a camera refit (inference-testing-04): sim joint = logged reading + offset, and a shift
    # of each arm's pedestal. Comma lists; or --kinematics <json> {"left": {"joint_offsets_deg": [5], "pedestal_shift_cm": [3]}, ...}
    ap.add_argument("--offsets-left", default=None, help="5 joint offsets (deg) added to the left arm's logged joints")
    ap.add_argument("--offsets-right", default=None)
    ap.add_argument("--pedestal-left", default=None, help="x,y,z shift of the left arm's pedestal (cm)")
    ap.add_argument("--pedestal-right", default=None)
    ap.add_argument("--kinematics", default=None, help="json with the above per arm")
    ap.add_argument("--show-wrist-mounts", action="store_true", help="keep the wrist camera mount / pcb / lens visible "
                    "(default: hidden -- a real camera cannot see its own mount; inference-testing-04's fits hide them)")
    ap.add_argument("--per-run", action="store_true", help="also print the medians per real run")
    a = ap.parse_args()
    vec = lambda t, n: None if t is None else np.array([float(v) for v in str(t).split(",")])[:n]
    joff = {"left": vec(a.offsets_left, 5), "right": vec(a.offsets_right, 5)}
    ped = {"left": vec(a.pedestal_left, 3), "right": vec(a.pedestal_right, 3)}
    if a.kinematics:
        kin = json.loads(Path(a.kinematics).read_text())
        for s in ("left", "right"):
            if s in kin:
                if kin[s].get("joint_offsets_deg") is not None:
                    joff[s] = np.asarray(kin[s]["joint_offsets_deg"], float)[:5]
                if kin[s].get("pedestal_shift_cm") is not None:
                    ped[s] = np.asarray(kin[s]["pedestal_shift_cm"], float)[:3]
    import os
    os.environ.setdefault("SQUEEZE_TASK", "clamp")
    import mujoco
    from PIL import Image
    import env as E
    import cube_locator as CL
    import vision as V
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    base_sim = E._Sim

    class _RigSim(base_sim):
        def __init__(self, *a_, **k_):
            k_.setdefault("camera_rig", str(a.rig))
            super().__init__(*a_, **k_)
    E._Sim = _RigSim
    env = E.SqueezeEnv()
    env.reset(seed=0)
    sim, m, d = env.sim, env.model, env.data
    glo, ghi = m.actuator_ctrlrange[sim.robot["left"].gripper_actuator]
    cq = m.jnt_qposadr[m.joint("cube_joint").id]
    bq = m.jnt_qposadr[m.joint("can_joint").id]
    for s in ("left", "right"):
        if ped[s] is not None:
            m.body_pos[m.body(f"{s}_pedestal").id] += ped[s] / 100.0
    print("kinematics: joint offsets", {s: None if joff[s] is None else joff[s].tolist() for s in joff},
          "| pedestal shift cm", {s: None if ped[s] is None else ped[s].tolist() for s in ped})
    if not a.show_wrist_mounts:
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name in ("left_camera_mount", "right_camera_mount") and m.geom_group[g] <= 2:
                m.geom_matid[g] = -1
                m.geom_rgba[g, 3] = 0.0
    cam = V.OverheadCamera.from_env(sim)
    rows, examples, shift_mm = [], [], []
    for run in sorted(Path(a.runs).glob("real_policy_*")):
        meta_p, log_p = run / "meta.json", run / "log.json"
        if not (meta_p.exists() and log_p.exists() and (run / "frames").is_dir()):
            continue
        meta = json.loads(meta_p.read_text())
        if not meta.get("live") or "cube" not in meta:
            continue
        L = json.loads(log_p.read_text())
        cx, cy = meta["cube"][:2]
        bx, by = (meta.get("box") or [cx, cy - 0.11])[:2]
        for f in sorted((run / "frames").glob("t*.jpg")):
            k = int(f.stem[1:5])
            if k >= len(L) or k > a.max_tick:
                continue
            u = L[k]["units"]
            for s in ("left", "right"):
                q5 = np.asarray(u[s][:5], float) + (joff[s] if joff[s] is not None else 0.0)
                d.qpos[sim.robot[s].qpos_ids] = np.deg2rad(q5)
                d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = glo + np.clip(u[s][5], 0, 100) / 100 * (ghi - glo)
            mujoco.mj_kinematics(m, d)
            tips = [sim.robot[s].position[:2] for s in ("left", "right")]
            if min(np.linalg.norm(t - np.array([cx, cy])) for t in tips) < 0.05:
                continue                                   # a fingertip near the cube: it may have moved / be hidden
            strip = np.asarray(Image.open(f).convert("RGB"))
            real = {c: strip[:, 320 * i:320 * (i + 1)] for i, c in enumerate(("overhead", "left_wrist", "right_wrist"))}
            ov = np.asarray(Image.fromarray(real["overhead"]).resize((640, 480), Image.Resampling.BILINEAR))
            pose = V.detect_cube_pose(ov, cam)
            if pose is None:
                continue
            # cube position from the OVERHEAD ONLY (its calibration is the trusted one, ~2 px): the runner's logged start
            # reading was triangulated WITH the old wrist models, which are what is being checked here. Back-projecting
            # the blob at one height is biased (the overhead sees mostly the top face; inference-testing-04's audit,
            # ~1 cm), so: ANALYSIS BY SYNTHESIS -- render the sim overhead with the cube at the estimate (arms at this
            # frame's joints), same red detector on sim and real, shift the estimate by the table-plane difference of
            # the two blob centroids until they agree.
            px, py, yaw = float(pose[0]), float(pose[1]), float(pose[2])
            ruv_o, _ = detect_red_merged(ov, min_px=30)
            for _ in range(6):
                d.qpos[cq:cq + 7] = [px, py, 0.016, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                mujoco.mj_forward(m, d)
                suv_o, _ = detect_red_merged(sim.render_camera("overhead", native=True), min_px=30)
                if ruv_o is None or suv_o is None:
                    break
                pr, ps = cam.to_plane(*ruv_o, 0.016), cam.to_plane(*suv_o, 0.016)
                if pr is None or ps is None:
                    break
                px, py = px + float(pr[0] - ps[0]), py + float(pr[1] - ps[1])
                if np.hypot(pr[0] - ps[0], pr[1] - ps[1]) < 2e-4:
                    break
            shift_mm.append(1000 * float(np.hypot(px - pose[0], py - pose[1])))
            d.qpos[cq:cq + 7] = [px, py, 0.016, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            d.qpos[bq:bq + 7] = [bx, by, E.CAN_HEIGHT / 2, 1, 0, 0, 0]
            mujoco.mj_forward(m, d)
            for c in WRISTS:
                ruv, rn = detect_red_merged(real[c], min_px=8)
                sim_img = sim.render_camera(c, native=True)
                suv, sn = detect_red_merged(sim_img, min_px=30)
                if ruv is None or suv is None:
                    rows.append((c, run.name, k, None, None, ruv is not None, suv is not None))
                    continue
                ratio = sn / (rn * 4.0)                       # real frames are 320x240: x4 area at 640x480
                off = float(np.hypot(suv[0] - 2 * ruv[0], suv[1] - 2 * ruv[1]))
                rows.append((c, run.name, k, ratio, off, True, True))
                if a.examples and len(examples) < 8 and c == "left_wrist" and k % 25 == 0:
                    sm = np.asarray(Image.fromarray(sim_img).resize((320, 240)))
                    examples.append(np.concatenate([real[c], sm], 1))
    print(f"rig: {a.rig}")
    if shift_mm:
        print(f"  cube placement by synthesis moved the single-height estimate by median {np.median(shift_mm):.1f} mm "
              f"(p90 {np.percentile(shift_mm, 90):.1f})")
    for c in WRISTS:
        r = [x for x in rows if x[0] == c]
        both = [x for x in r if x[3] is not None]
        if not both:
            print(f"  {c}: no frames where both see the cube ({len(r)} frames; real seen {sum(x[5] for x in r)}, sim seen "
                  f"{sum(x[6] for x in r)})")
            continue
        if a.per_run:
            for rn in sorted({x[1] for x in both}):
                rr = [x for x in both if x[1] == rn]
                print(f"    {rn}: {len(rr):3d} frames | area median {np.median([x[3] for x in rr]):.2f} | offset median "
                      f"{np.median([x[4] for x in rr]):.0f} px, max {max(x[4] for x in rr):.0f} px")
        ratio = np.array([x[3] for x in both])
        off = np.array([x[4] for x in both])
        print(f"  {c}: {len(both)} frames (of {len(r)}; real-only {sum(x[5] and not x[6] for x in r)}, sim-only "
              f"{sum(x[6] and not x[5] for x in r)}) | cube area sim/real median {np.median(ratio):.2f} "
              f"(p10 {np.percentile(ratio, 10):.2f}, p90 {np.percentile(ratio, 90):.2f}) | centroid offset median "
              f"{np.median(off):.0f} px, p90 {np.percentile(off, 90):.0f} px (640x480)")
    if a.examples and examples:
        Image.fromarray(np.concatenate(examples, 0)).save(a.examples, quality=85)
        print(f"  examples (left wrist: real | sim): {a.examples}")


if __name__ == "__main__":
    main()
