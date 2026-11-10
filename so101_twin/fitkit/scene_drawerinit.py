"""drawer, initialisation only: fixed 'side' camera on brightly lit episodes, real mask = the builder's
hand-tuned overhead mask, twin = coloured parts. Gives a starting camera for the SAM rounds of scene_drawer (all episodes, both views)."""
import numpy as np
from PIL import Image

from _common import builder, view
from armfit import clean


def config() -> dict:
    bs, tw = builder("drawer")
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "neuroscope" / "molmoact2"))
    from sets import Store

    fs = Store().frames("so101_drawer_real")
    ep = np.asarray(fs.labels["episode"])
    bright = [int(e) for e in np.unique(ep)
              if np.mean([np.asarray(fs.obs(i)["images"]["side"]).mean() for i in np.flatnonzero(ep == e)[:3]]) > 70]

    from armfit import load_module
    fc = load_module(__import__("pathlib").Path(__file__).resolve().parents[1] / "drawer" / "fit_cameras.py",
                     "drawer_fit_cameras")

    def mask(rgb, v):   # the builder's overhead mask (green / white / orange parts minus the blocks and fixed clutter)
        return fc.real_arm_mask(rgb, "side")

    views = {"side": view(bs, "side", episodes=bright)}
    return dict(real="so101_drawer_real", views=views, base_pos=bs.BASE_POS, base_quat=bs.BASE_QUAT,
                real_to_sim=tw.real_to_sim, real_mask=mask, bright_episodes=bright)
