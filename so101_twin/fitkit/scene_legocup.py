"""legocup: Rayenghali lego -> black cup. Black arm reaching in from the top edge of the fixed 'top' camera;
'side' = wrist camera on the fixed jaw."""
from _common import builder, view

PARTS = dict(servos=True)


def config() -> dict:
    bs, tw = builder("legocup")
    views = {"top": view(bs, "top", clean=dict(link=6, keep="edge", min_px=300, fill=1500)),
             "side": view(bs, "side", mount="jaw", role="wrist", clean=dict(link=4, keep="edge", min_px=300))}
    return dict(real="so101_legocup_real", views=views, base_pos=bs.BASE_POS, base_quat=bs.BASE_QUAT,
                real_to_sim=tw.real_to_sim, real_mask="auto", parts=PARTS)
