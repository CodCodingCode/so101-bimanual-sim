#!/usr/bin/env python3
"""Does the camera student see the REAL cube where it is? Before any real-arm run, feed real camera frames (with the
cube's true position from the overhead detector and the real joint angles) through the student's auxiliary
"where is the cube" head and report the error in cm. The last camera policy misjudged the real cube by ~3 cm.

    python squeeze/validate_real_cube.py runs_clamp_v3/student/student.pt
Samples: real frames saved by the rig runners (full 640x480 or the 320x240 composites), each with its joint angles.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
os.environ.setdefault("SQUEEZE_TASK", "clamp")
os.environ.setdefault("CLAMP_RECOVERY", "1")
SCR = Path("/private/tmp/claude-501/-Users-owner-Downloads-whatif-sim/91ecc604-fde6-4144-892d-9cd4dc1fb82b/scratchpad")


def samples():
    """(name, {cam: HxWx3 uint8}, units[12], cube_xy_true m)"""
    from PIL import Image
    out = []
    d = SCR / "dry1"
    if (d / "overhead_full.jpg").exists():
        out.append(("dry check, arms at rest", {c: np.asarray(Image.open(d / f"{c}_full.jpg").convert("RGB"))
                                                for c in ("overhead", "left_wrist", "right_wrist")},
                    np.load(d / "units.npy"), (-0.029, 0.040)))
    for folder, tick, cube in (("clamp1", "t0085_descend", (-0.005, 0.039)),):
        f, L = SCR / folder / f"{tick}.jpg", SCR / folder / "log.json"
        if f.exists() and L.exists():
            comp = np.asarray(Image.open(f).convert("RGB"))
            cams = {c: comp[:, i * 320:(i + 1) * 320] for i, c in enumerate(("overhead", "left_wrist", "right_wrist"))}
            r = json.load(open(L))[int(tick[1:5])]
            out.append((f"scripted run, {tick}", cams, np.r_[r["units"]["left"], r["units"]["right"]], cube))
    f = SCR / "live1" / "t0000.jpg"
    if f.exists() and (SCR / "grab2" / "log.json").exists():
        comp = np.asarray(Image.open(f).convert("RGB"))
        cams = {c: comp[:, i * 320:(i + 1) * 320] for i, c in enumerate(("overhead", "left_wrist", "right_wrist"))}
        r = json.load(open(SCR / "grab2" / "log.json"))[0]          # same start pose (SIM_HOME)
        out.append(("camera-policy run, start pose", cams, np.r_[r["units"]["left"], r["units"]["right"]], (-0.004, 0.037)))
    return out


def main():
    import mujoco
    import torch
    from PIL import Image
    import clamp_teacher as C
    import distill as Dl
    net = Dl.build_student()
    net.load_state_dict(torch.load(sys.argv[1], map_location="cpu"))
    net.eval()
    env = Dl.make_env(1)
    env.reset(seed=1)
    sim, m, d = env.base.sim, env.model, env.data
    glo, ghi = m.actuator_ctrlrange[sim.robot["left"].gripper_actuator]
    errs = []
    for name, cams, u, cube in samples():
        for i, s in enumerate(("left", "right")):
            arm = sim.robot[s]
            d.qpos[arm.qpos_ids] = np.deg2rad(u[i * 6:i * 6 + 5])
            d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = glo + np.clip(u[i * 6 + 5], 0, 100) / 100 * (ghi - glo)
        mujoco.mj_forward(m, d)
        for i, s in enumerate(("left", "right")):
            sim._ee_targets[i] = sim.robot[s].position.copy()
        imgs = np.stack([np.asarray(Image.fromarray(cams[c]).resize((Dl.IMG, Dl.IMG), Image.Resampling.BILINEAR))
                         for c in Dl.CAMS])
        parts = []
        for i, s in enumerate(("left", "right")):
            arm = sim.robot[s]
            parts += [np.rad2deg(d.qpos[arm.qpos_ids]) / 90.0, [arm.gripper_opening], arm.position * 5.0,
                      sim._ee_targets[i] * 5.0]
        parts += [np.zeros(6), [0.0]]
        prop = np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])
        with torch.no_grad():
            _, q = net.forward_all(Dl.to_tensor(imgs[None], "cpu"), torch.from_numpy(prop[None]))
        q = q.numpy()[0]
        est = np.array([q[0] * 0.05, C.CUBE_Y + q[1] * 0.05])
        e = np.linalg.norm(est - np.array(cube)) * 100
        errs.append(e)
        print(f"{name:34s} true cube ({cube[0] * 100:+.1f}, {(cube[1] - C.CUBE_Y) * 100:+.1f}) cm  student sees "
              f"({est[0] * 100:+.1f}, {(est[1] - C.CUBE_Y) * 100:+.1f}) cm  -> error {e:.1f} cm", flush=True)
    print(f"real frames: {len(errs)}, mean error {np.mean(errs):.1f} cm, max {np.max(errs):.1f} cm")


if __name__ == "__main__":
    main()
