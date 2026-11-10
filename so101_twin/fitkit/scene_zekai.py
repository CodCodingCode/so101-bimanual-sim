"""zekai (replacement for bincube, third try): Zekai-Chen/so101_pi05_pick_place_3cam (v3.0). Orange SO-101 on a dark
mat; 'top' = overhead (arm base at the top edge), 'side' = low front view. Real mask: saturated orange (the cardboard
box is browner and less saturated), linked to the image edge (top) / largest group (side)."""
import numpy as np

import _v3joints
from armfit import clean, hsv_mask

ORANGE = [((4, 24), (165, 255), (90, 255))]


def config() -> dict:
    views = {
        "top": dict(cam="top", init=[0.0, 0.30, 0.75, -90.0, -85.0, 0.0, 60.0], mount="world", role="arm",
                    clean=dict(link=8, keep="edge", min_px=300)),
        "side": dict(cam="side", init=[0.0, 0.9, 0.2, -90.0, -8.0, 0.0, 55.0], mount="world", role="arm",
                     clean=dict(link=8, keep="largest", min_px=200)),
    }

    def mask(rgb, v):
        return clean(hsv_mask(rgb, ORANGE), **views[v]["clean"])

    return dict(real="so101_zekai_real", views=views, base_pos=(0.0, 0.0, 0.0), base_quat="0 0 0 1",
                real_to_sim=_v3joints.real_to_sim, gain_limit=[0.35, 0.35, 0.35, 0.35, 1.0, 0.0],
                real_mask=mask, look=dict(body="0.98 0.45 0.05 1", servo="0.06 0.06 0.06 1"))
