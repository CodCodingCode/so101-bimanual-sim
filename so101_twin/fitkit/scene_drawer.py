"""drawer: 0mrrabbit0/so101_place_block_in_drawer. Green arm, black servos, orange/yellow parts, black pegboard.
'side' = fixed overhead camera (the arm enters from the image edge); 'top' = wrist camera on the fixed jaw."""
from _common import builder, view


def config() -> dict:
    bs, tw = builder("drawer")
    views = {"side": view(bs, "side", clean=dict(link=8, keep="edge", min_px=300)),
             "top": view(bs, "top", mount="jaw", role="wrist", clean=dict(link=4, keep="edge", min_px=300))}
    return dict(real="so101_drawer_real", views=views, base_pos=bs.BASE_POS, base_quat=bs.BASE_QUAT,
                real_to_sim=tw.real_to_sim, real_mask="auto",
                look=dict(body="0.56 0.74 0.50 1", servo="0.08 0.08 0.08 1", jaw="0.92 0.92 0.90 1"))
