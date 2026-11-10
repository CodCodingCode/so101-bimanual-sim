"""jonh (replacement for drawer): jonhpark/so101_pick_place (v2.1, old LeRobot calibration, degrees). White SO-101 on a
dark grey table, plush toy into a white bowl; 'top' = overhead camera (arm base at the bottom edge), 'side' = the
dataset's 'front' camera (arm at the back of the table). Real mask = bright unsaturated pixels; pixels bright in
>= 80% of frames ignored in both masks. Joint map: lemon's old-calibration signs with zero offsets (fitted)."""
import numpy as np
from PIL import Image

from armfit import clean

SIGNS = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
JAW_RANGE = (-0.174, 1.75)


def real_to_sim(q):
    q = np.asarray(q, float)
    rad = np.deg2rad(SIGNS * q[:5])
    return np.concatenate([rad, [JAW_RANGE[0] + np.clip(q[5], 0, 100) / 100.0 * (JAW_RANGE[1] - JAW_RANGE[0])]])


def bright(rgb):
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    return (hsv[..., 2] >= 150) & (hsv[..., 1] <= 60)


BLUE = [((125, 175), (45, 255), (40, 255))]   # pale-blue plush toy and the blue base clamps (PIL hue scale)


def config() -> dict:
    views = {
        "top": dict(cam="top", init=[0.0, 0.25, 0.75, 90.0, -89.0, 0.0, 60.0], mount="world", role="arm",
                    ignore_static_mask=0.8, raw_mask=bright, clean=dict(link=6, keep="edge", min_px=300), subtract_hsv=BLUE),
        "side": dict(cam="side", init=[0.0, 0.9, 0.35, -90.0, -20.0, 0.0, 60.0], mount="world", role="arm",
                     ignore_static_mask=0.8, raw_mask=bright, clean=dict(link=6, keep="largest", min_px=300),
                     subtract_hsv=BLUE),
    }

    def mask(rgb, v):
        return clean(bright(rgb), **views[v]["clean"])

    return dict(real="so101_jonh_real", views=views, base_pos=(0.0, 0.0, 0.0), base_quat="0 0 0 1",
                real_to_sim=real_to_sim, real_mask=mask, parts=dict(servos=True),
                look=dict(body="0.93 0.93 0.92 1", servo="0.08 0.08 0.08 1"))
