"""bincube: MCeut/so101_lab_openBin_putCube. Black arm (servos included in the twin mask) on light wood.
'side' = fixed camera; 'top' = wrist camera on the fixed jaw."""
from _common import builder, view

PARTS = dict(servos=True)


def config() -> dict:
    bs, tw = builder("bincube")
    views = {"side": view(bs, "side", clean=dict(link=6, keep="largest", min_px=300, fill=1500),
                         ignore_static_dark=70),
             "top": view(bs, "top", mount="jaw", role="wrist", clean=dict(link=4, keep="edge", min_px=300))}
    return dict(real="so101_bincube_real", views=views, base_pos=bs.BASE_POS, base_quat=bs.BASE_QUAT,
                real_to_sim=tw.real_to_sim, real_mask="auto", parts=PARTS)
