#!/usr/bin/env python3
"""Record the cube-anywhere demo in sim: one arm pushes the cube into the clamp zone (closed-loop pusher, push_teacher),
then -- as on the robot, a separate run from the start pose -- both arms tip-clamp it and drop it in the box at its
fixed spot (clamp design design_clamp3c_twistoff). Real joint ranges, fitted servos.

    CLAMP_DESIGN=design_clamp3c_twistoff.json python squeeze/record_push_clamp.py --out ~/Downloads/push_clamp_demo.mp4
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
os.environ.setdefault("CLAMP_DESIGN", "design_clamp3c_twistoff.json")

BOX_XY = (0.0, -0.077)
SPOTS = ((-0.10, 0.0), (0.125, 0.0), (-0.05, 0.025), (0.075, 0.025), (0.025, 0.0))   # cm offsets / 100 from nominal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="behind", help="comma list of behind / front / side (panels side by side)")
    ap.add_argument("--hide-mount", action="store_true", help="hide the overhead camera post + head in the video")
    a = ap.parse_args()
    import imageio
    import mujoco
    from PIL import Image, ImageDraw
    import clamp_teacher as C
    import env as E
    import push_teacher as PT
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    VIEWS = {"behind": ([0.0, -0.01, 0.03], 0.66, -90, -48),     # the original view (behind the arms, past the camera post)
             "front": ([0.0, 0.0, 0.05], 0.50, -90, -18),        # low, from the cube side of the table, facing the arms
             "side": ([0.0, -0.02, 0.04], 0.56, 0, -24)}         # from the right, level with the clamp
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    W, H = (800, 560) if len(views) == 1 else (640, 448)
    writer = imageio.get_writer(a.out, fps=25, codec="libx264", quality=7, macro_block_size=8)
    cams = []
    for v in views:
        c = mujoco.MjvCamera()
        look, dist, az, el = VIEWS[v]
        c.lookat[:] = look
        c.distance, c.azimuth, c.elevation = dist, az, el
        cams.append(c)

    def new_env(cube_xy, yaw_deg):
        env = E.SqueezeEnv(episode_seconds=120.0)
        m, d = env.model, env.data
        env.reset(seed=1)
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W), max(m.vis.global_.offheight, H)
        cq, bq = m.jnt_qposadr[m.joint("cube_joint").id], m.jnt_qposadr[m.joint("can_joint").id]
        y = np.radians(yaw_deg)
        d.qpos[cq:cq + 7] = [cube_xy[0], cube_xy[1], C.CUBE_HALF + 0.001, np.cos(y / 2), 0, 0, np.sin(y / 2)]
        d.qpos[bq:bq + 7] = [BOX_XY[0], BOX_XY[1], E.CAN_HEIGHT / 2, 1, 0, 0, 0]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        for _ in range(20):
            mujoco.mj_step(m, d)
        if a.hide_mount:                   # video only: the overhead camera post + head (collision shapes untouched)
            for g in range(m.ngeom):
                if m.geom(g).name in ("mount_cam_mount_bottom", "mount_cam_mount_middle", "mount_cam_mount_top",
                                      "overhead_camera_housing", "overhead_camera_lens"):
                    m.geom_matid[g] = -1
                    m.geom_rgba[g, 3] = 0.0
        return env, mujoco.Renderer(m, H, W)

    def yaw_of(env):
        R = env.data.xmat[env.model.body("cube").id].reshape(3, 3)
        return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    def frame(env, ren, lines, zone=True):
        panels = []
        for c in cams:
            ren.update_scene(env.data, c)
            panels.append(ren.render())
        img = Image.fromarray(np.concatenate(panels, 1))
        dr = ImageDraw.Draw(img)
        for i, v in enumerate(views):
            dr.text((W * i + W - 70, H - 20), v + " view", fill=(230, 230, 230))
        for i, (t, col) in enumerate(lines):
            dr.text((12, 10 + 18 * i), t, fill=col)
        writer.append_data(np.asarray(img))

    def act(env, tg):
        sim = env.sim
        u = np.zeros((2, 4))
        u[:, :3] = (tg - sim._ee_targets) / sim.delta_scale
        env.step(np.clip(u, -1, 1).ravel())

    wins = 0
    for n, (dx, dy) in enumerate(SPOTS, 1):
        start = (dx, C.CUBE_Y + dy)
        head = f"demo {n}/{len(SPOTS)}: cube starts {dx * 100:+.1f} cm sideways, {(start[1] + 0.22) * 100:.1f} cm from the bases"
        # ---- push run
        env, ren = new_env(start, 0.0)
        push = PT.ClosedLoopPush(env.sim, env.cube_pos()[:2])
        k = 0
        while not push.done and k < 1500:
            act(env, push.step(env.sim._ee_targets.copy(), env.cube_pos()[:2], yaw_of(env)))
            if k % 2 == 0:
                c = env.cube_pos()
                frame(env, ren, [(head, (255, 255, 255)),
                                 (f"STEP 1  PUSH ({push.side} arm, closed loop): {push.mode}", (255, 200, 120)),
                                 (f"cube at {c[0] * 100:+.1f}, {(c[1] - C.CUBE_Y) * 100:+.1f} cm, twist {yaw_of(env):+.0f} deg"
                                  f"  -> zone: +-1 cm sideways, 0.2..1.8 cm deep, twist <= 22 deg", (210, 210, 210))])
            k += 1
        c, yaw = env.cube_pos(), yaw_of(env)
        for _ in range(12):
            frame(env, ren, [(head, (255, 255, 255)), (f"push done: cube at {c[0] * 100:+.1f}, {(c[1] - C.CUBE_Y) * 100:+.1f} cm, "
                                                       f"twist {yaw:+.0f} deg -> clamp", (120, 230, 120))])
        env.close()
        # ---- clamp run (fresh, from the start pose, like the real runner)
        env, ren = new_env((c[0], c[1]), yaw)
        for s in C.SIDES:
            env.sim.pad_quat[s] = C.clamp_quat(s, (c[0], c[1]))
        plan = C.ClampPlan((c[0], c[1], 0.0), BOX_XY)
        k = 0
        while not plan.done:
            act(env, plan.next_targets(env.sim._ee_targets.copy()))
            if k % 2 == 0:
                frame(env, ren, [(head, (255, 255, 255)),
                                 (f"STEP 2  TWO-ARM TIP CLAMP (no jaws) -> box: {plan.phase}", (150, 200, 255))])
            k += 1
        for _ in range(40):
            env.step(np.zeros(8))
        ok = bool(env.on_can())
        wins += ok
        for _ in range(30):
            frame(env, ren, [(head, (255, 255, 255)),
                             ("SUCCESS: cube in the box" if ok else "failed", (90, 230, 120) if ok else (240, 90, 90))])
        print(f"demo {n}: start {dx * 100:+.1f},{dy * 100:+.1f} cm -> pushed to {c[0] * 100:+.1f},{(c[1] - C.CUBE_Y) * 100:+.1f} "
              f"twist {yaw:+.0f} -> {'IN BOX' if ok else 'failed'}", flush=True)
        env.close()
    writer.close()
    print(f"{wins}/{len(SPOTS)} in the box -> {a.out}")


if __name__ == "__main__":
    main()
