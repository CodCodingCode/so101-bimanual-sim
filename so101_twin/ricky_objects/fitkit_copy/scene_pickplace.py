"""pickplace: lerobot/svla_so101_pickplace. Black arm with white jaws on a white table; two fixed cameras
('top' = the real 'up' camera, 'side'). Real arm = dark pixels (max RGB < 70), the largest linked group, small holes filled; twin arm =
every arm mesh including the black servos, minus the two white jaws."""
import json
import pathlib

import numpy as np

from armfit import clean, load_module, lookat_to_ypr

D = pathlib.Path(__file__).resolve().parent.parent / "pickplace"
THR = 70


def config() -> dict:
    jm = load_module(D / "joints.py", "pickplace_joints")
    cams = json.loads((D / "cameras.json").read_text())
    views = {c: dict(cam=c, init=[*lookat_to_ypr(v["pos"], v["lookat"], v["roll"]), v["fovy"]], mount="world")
             for c, v in cams.items()}

    def mask(rgb, view):
        return clean(np.asarray(rgb).max(-1) < THR, min_px=300, link=6, keep="largest", fill=2000)

    return dict(real="so101_pickplace_real", views=views, base_pos=(0.0, 0.0, 0.0), base_quat="1 0 0 0",
                real_to_sim=jm.real_to_sim, real_mask=mask,
                look=dict(body="0.07 0.07 0.07 1", servo="0.04 0.04 0.04 1", jaw="0.92 0.92 0.90 1"),
                parts=dict(servos=True, exclude=dict(so100=["Fixed_Jaw", "Moving_Jaw"],
                                                     so101=["wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"])))
