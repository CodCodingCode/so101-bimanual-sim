"""stack: ichbinblau/so101_stack2cubes_dataset. White arm, black servos; 'top' = overhead camera that shifts a little
between episodes (one global pose here), 'side' = low fixed camera that sees the arm only near the blocks."""
from _common import builder, view


def config() -> dict:
    bs, tw = builder("stack")
    views = {"top": view(bs, "top", clean=dict(link=8, keep="edge", min_px=300)),
             "side": view(bs, "side", clean=dict(link=6, keep="edge", min_px=300))}
    return dict(real="so101_stack_real", views=views, base_pos=bs.BASE_POS, base_quat=bs.BASE_QUAT,
                real_to_sim=tw.real_to_sim, real_mask="auto")
