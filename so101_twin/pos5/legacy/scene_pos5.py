"""pos5 (replacement for sort): hbseong/record-pick-and-place-pos5-so101. White SO-101 on a green screen, blue cube
into a white box; 'top' = overhead camera, 'side' = the dataset's 'right' camera (low side view). No builder scene:
real mask = not-green pixels, with pixels that are not-green in >= 80% of frames (the box, the base, mat seams)
ignored in both masks."""
import numpy as np
from PIL import Image

import _v3joints
from armfit import clean


def notgreen(rgb):
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    green = (hsv[..., 0] >= 70) & (hsv[..., 0] <= 140) & (hsv[..., 1] >= 50)
    return ~green & (hsv[..., 2] >= 40)


CUBE = [((112, 150), (70, 255), (50, 255))]   # the cyan-blue cube SAM glues to the gripper (PIL hue scale)


def config() -> dict:
    views = {
        "top": dict(cam="top", init=[0.15, 0.25, 0.75, -90.0, -89.0, 0.0, 60.0], mount="world", role="arm",
                    ignore_static_mask=0.8, raw_mask=notgreen, clean=dict(link=6, keep="edge", min_px=300),
                    subtract_hsv=CUBE),
        "side": dict(cam="side", init=[0.75, 0.15, 0.12, 180.0, -5.0, 0.0, 60.0], mount="world", role="arm",
                     ignore_static_mask=0.8, raw_mask=notgreen, clean=dict(link=6, keep="largest", min_px=300),
                     ignore_boxes=[(0, 280, 0, 60)],   # pale wall corner, not green, top-left
                     subtract_hsv=CUBE),
    }

    def mask(rgb, v):
        return clean(notgreen(rgb), **views[v]["clean"])

    return dict(real="so101_pos5_real", views=views, base_pos=(0.0, 0.0, 0.0), base_quat="0 0 0 1",
                real_to_sim=_v3joints.real_to_sim, gain_limit=0.35,   # -100..100 normalised joint units
                real_mask=mask,
                parts=dict(servos=True), look=dict(body="0.93 0.93 0.92 1", servo="0.08 0.08 0.08 1"))
