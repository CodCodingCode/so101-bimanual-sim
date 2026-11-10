"""cube19 (replacement for bincube): jackvial/so101_pick_and_place_cube_19_e50. White SO-101 on a dark table, cube
onto an X marker; 'top' = overhead, 'side' = low view from across the table. Real mask = bright, unsaturated pixels;
pixels bright in >= 80% of frames (the X marker, the base) ignored in both masks."""
import numpy as np
from PIL import Image

import _v3joints
from armfit import clean


def bright(rgb):
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    return (hsv[..., 2] >= 110) & (hsv[..., 1] <= 70)


def config() -> dict:
    views = {
        "top": dict(cam="top", init=[0.20, 0.10, 0.80, 0.0, -89.0, 0.0, 60.0], mount="world", role="arm",
                    ignore_static_mask=0.8, raw_mask=bright, clean=dict(link=6, keep="edge", min_px=300)),
        "side": dict(cam="side", init=[-0.40, 0.45, 0.20, -45.0, -15.0, 0.0, 60.0], mount="world", role="arm",
                     ignore_static_mask=0.8, raw_mask=bright, clean=dict(link=6, keep="edge", min_px=300)),
    }

    def mask(rgb, v):
        return clean(bright(rgb), **views[v]["clean"])

    return dict(real="so101_cube19_real", views=views, base_pos=(0.0, 0.0, 0.0), base_quat="0 0 0 1",
                real_to_sim=_v3joints.real_to_sim, gain_limit=[0.35, 0.35, 0.35, 0.35, 1.0, 0.0],   # -100..100 normalised units; roll may span ~a full turn
                real_mask=mask,
                parts=dict(servos=True), look=dict(body="0.93 0.93 0.92 1", servo="0.08 0.08 0.08 1"))
