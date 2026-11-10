"""lemon: Beegbrain/pick_lemon_and_drop_in_bowl. Orange SO-100-style arm, blue jaws; two fixed RealSense views."""
import pathlib

from armfit import clean, hsv_mask, load_module

D = pathlib.Path(__file__).resolve().parent.parent / "lemon"


def config() -> dict:
    bs = load_module(D / "build_scene.py", "lemon_build_scene")
    tw = load_module(D / "twin.py", "lemon_twin")
    # hue/sat/val on PIL's 0-255 scale. Arm orange is hue 5-13 (peach/orange fruit 17-22, apple/strawberry <= 4);
    # jaw blue is hue ~144 with sat > 200 (the black servos read as dark low-sat blue, val < 40).
    orange = ((4, 14), (140, 255), (60, 255))
    blue = ((134, 156), (150, 255), (45, 255))

    def mask(rgb, view):
        return clean(hsv_mask(rgb, [orange, blue]), min_px=250, link=8)

    views = {c: dict(cam=c, init=[*v["pos"], v["yaw"], v["pitch"], v["roll"], v["fovy"]], mount="world")
             for c, v in bs.CAMS.items()}
    return dict(real="so101_lemon_real", views=views, base_pos=bs.BASE_POS, base_quat=bs.BASE_QUAT,
                real_to_sim=tw.real_to_sim, real_mask=mask)
