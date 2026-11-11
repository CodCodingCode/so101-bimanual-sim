#!/usr/bin/env python3
"""Record the CAMERA student doing the clamp task in sim: random cube/box spawn, random colours/light, disturbances.
Big view = a free camera for you; strip at the bottom = the three 64x64 images the policy actually sees.

    python squeeze/record_student.py --episodes 6 --out /tmp/student.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
os.environ.setdefault("SQUEEZE_TASK", "clamp")
os.environ.setdefault("CLAMP_RECOVERY", "1")

import distill as Dl      # noqa: E402

LABEL = {"none": "no disturbance", "nudge": "cube moved before the grab", "slip": "cube knocked out mid-carry",
         "push": "arm shoved"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=str(Dl.OUT / "student.pt"))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=950_000)       # unseen seeds (training used 100k.., eval 900k..)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import imageio
    import mujoco
    import torch
    from PIL import Image, ImageDraw
    net = Dl.build_student()
    net.load_state_dict(torch.load(a.student, map_location="cpu"))
    net.eval()
    env = Dl.make_env(a.seed)
    rnd = Dl.Randomizer(env)
    m, d = env.model, env.data
    W, H = 768, 512
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W), max(m.vis.global_.offheight, H)
    ren = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, -0.04, 0.04]
    cam.distance, cam.azimuth, cam.elevation = 0.62, -70, -32
    writer = imageio.get_writer(a.out, fps=25, codec="libx264", quality=7, macro_block_size=16)
    wins = 0
    for k in range(a.episodes):
        seed = a.seed + k
        rng = np.random.default_rng(seed + 777)
        rnd.apply(rng)
        env.reset(seed=seed)
        prev, after, result = np.zeros(6, np.float32), 0, "running"
        for t in range(env.base.max_steps):
            imgs, prop = Dl.observe(env, prev)
            with torch.no_grad():
                act = net(Dl.to_tensor(imgs[None], "cpu"), torch.from_numpy(prop[None])).numpy()[0]
            _, _, term, trunc, info = env.step(np.clip(act, -1, 1).astype(np.float32))
            prev = act.astype(np.float32)
            ren.update_scene(d, cam)
            big = Image.fromarray(ren.render())
            frame = Image.new("RGB", (W, H + 200), (18, 18, 22))
            frame.paste(big, (0, 0))
            for i, im in enumerate(imgs):
                frame.paste(Image.fromarray(im).resize((192, 192), Image.NEAREST), (8 + i * 200, H + 4))
            if info["is_success"]:
                result = "SUCCESS - cube in box"
                after += 1
            dr = ImageDraw.Draw(frame)
            cube, box = env.base.cube_pos(), env.base.can_pos()
            dr.text((10, 8), f"episode {k + 1}/{a.episodes}  (seed {seed})   CAMERA POLICY: cameras + joints only", fill=(255, 255, 255))
            dr.text((10, 26), f"spawn: cube ({env._attempt_cube[0] * 100:+.1f}, {env._attempt_cube[1] * 100:+.1f}) cm   "
                              f"box ({box[0] * 100:+.1f}, {box[1] * 100:+.1f}) cm", fill=(220, 220, 220))
            dr.text((10, 44), f"disturbance: {LABEL[info['perturbation']]}" + ("  <- happened" if info["perturbed"] else ""),
                    fill=(255, 200, 120))
            dr.text((10, 62), f"t = {t / 25:4.1f} s   grab attempt {info['attempts']}", fill=(220, 220, 220))
            dr.text((10, 80), result, fill=(90, 230, 120) if info["is_success"] else (200, 200, 200))
            dr.text((620, H + 12), "what the policy sees:", fill=(200, 200, 200))
            dr.text((620, H + 30), "overhead | left wrist |", fill=(160, 160, 160))
            dr.text((620, H + 46), "right wrist (64x64)", fill=(160, 160, 160))
            writer.append_data(np.asarray(frame))
            if term or trunc or after > 40:
                break
        ok = bool(info["is_success"])
        wins += ok
        print(f"episode {k + 1}: seed {seed}, {info['perturbation']}, {'SUCCESS' if ok else 'failed'}, "
              f"attempts {info['attempts']}, contacts outside grace {info['bad_free_steps']}", flush=True)
        for _ in range(25):
            writer.append_data(np.asarray(frame))
    writer.close()
    print(f"{wins}/{a.episodes} successful -> {a.out}")


if __name__ == "__main__":
    main()
