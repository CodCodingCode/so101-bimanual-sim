"""tape (replacement for bincube, second try after cube19): ReubenLim/so101_tape_in_square (v2.1, old calibration,
degrees). Black SO-101 pushing a tape roll on a white foam sheet; 'top' = birdEye overhead camera (arm base at the
bottom edge), 'side' = thirdPerson low camera (arm at the left, office background). Real mask for the start: dark
pixels (max RGB < 70) linked to the image edge, with pixels dark in >= 85% of frames ignored in both masks; SAM
rounds follow. Joint map: jonh's old-calibration signs with zero offsets (fitted)."""
import numpy as np

from armfit import clean
from scene_jonh import real_to_sim


def dark(rgb):
    return np.asarray(rgb).max(-1) < 70


def config() -> dict:
    views = {
        "top": dict(cam="top", init=[0.0, 0.30, 0.70, -90.0, -80.0, 0.0, 60.0], mount="world", role="arm",
                    ignore_static_dark=70, clean=dict(link=6, keep="edge", min_px=300)),
        "side": dict(cam="side", init=[0.5, 0.3, 0.15, 180.0, -10.0, 0.0, 60.0], mount="world", role="arm",
                     ignore_static_dark=70, clean=dict(link=6, keep="edge", min_px=300)),
    }

    def mask(rgb, v):
        return clean(dark(rgb), **views[v]["clean"])

    return dict(real="so101_tape_real", views=views, base_pos=(0.0, 0.0, 0.0), base_quat="0 0 0 1",
                real_to_sim=real_to_sim, real_mask=mask, parts=dict(servos=True), wrist_cam=True,
                look=dict(body="0.08 0.08 0.08 1", servo="0.05 0.05 0.05 1"))
