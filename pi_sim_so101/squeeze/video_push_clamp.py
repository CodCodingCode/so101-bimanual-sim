#!/usr/bin/env python3
"""Video of the wide push-then-clamp in sim (cube anywhere across x +-12 cm in the reachable band): free camera view,
caption with the start spot, phase and outcome. Camera-based cube sensing (--detect) as on the rig.

    CLAMP_GRIP_DR=1 WIDE_SERVO_BIAS=1 python squeeze/video_push_clamp.py --out runs_vla/push_clamp_demo.mp4
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "runs_vla" / "push_clamp_demo.mp4"))
    ap.add_argument("--starts", default="-9,0;9,0;-6,3;6,-3;12,0")
    ap.add_argument("--detect", action="store_true")
    a = ap.parse_args()
    import av
    import mujoco
    from PIL import Image, ImageDraw
    import clamp_teacher as C
    import clamp_wide as W
    import chunk_student as CS
    import push_clamp_wide as P
    T = CS.Teacher(str(HERE / P.CLAMP_POLICY))
    env = W.make_env_cls()(seed=7, level=0.0, yaw_max=45.0, episode_seconds=P.EPISODE_S)
    ren = mujoco.Renderer(env.model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, -0.03, 0.03]
    cam.distance, cam.azimuth, cam.elevation = 0.62, 90, -42
    box = av.open(a.out, "w")
    st = box.add_stream("libx264", rate=25)
    st.width, st.height, st.pix_fmt = 640, 480, "yuv420p"
    for k, s in enumerate(a.starts.split(";")):
        x, y = (float(v) for v in s.split(","))
        o, _ = env.reset(seed=5000 + k)
        o = P._set_start(env, (x / 100, C.CUBE_Y + y / 100))
        pc = P.PushClamp(env, T, detect=a.detect)
        pc.start(o)
        info = {}
        for t in range(env.base.max_steps):
            _, _, te, tr, info = pc.step()
            if t % 2 == 0:
                ren.update_scene(env.data, cam)
                im = Image.fromarray(ren.render())
                dr = ImageDraw.Draw(im)
                ph = pc.push.mode if pc.phase == "push" else (pc.phase if pc.phase == "rehome" else env.plan.phase)
                dr.rectangle([0, 0, 640, 22], fill=(0, 0, 0))
                dr.text((6, 5), f"start x {x:+.0f} cm, y {y:+.0f} cm | {pc.phase}: {ph} | t {t / 25:4.1f} s"
                        f"{' | IN THE BOX' if info.get('is_success') else ''}", fill=(255, 255, 255))
                for p in st.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")):
                    box.mux(p)
            if te or tr or (info.get("is_success") and pc.phase == "clamp" and env.plan.done):
                break
        print(f"start ({x:+.0f},{y:+.0f}): success {info.get('is_success')}", flush=True)
    for p in st.encode():
        box.mux(p)
    box.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
