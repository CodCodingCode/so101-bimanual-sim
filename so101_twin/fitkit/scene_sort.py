"""sort: hubnemo/so101_sort_cubes. White arm with light-blue wrist/jaws; the fixed 'top' camera moved between two
recording sessions (episodes 0-59 and 60+), so it is two views; 'side' = wrist camera on the fixed jaw."""
from _common import builder, view


def config() -> dict:
    bs, tw = builder("sort")
    a, b = list(bs.SESSIONS["top"]), list(bs.SESSIONS["top_b"])
    views = {"top": view(bs, "top", episodes=a, clean=dict(link=8, keep="edge", min_px=300)),
             "top_b": view(bs, "top_b", set_cam="top", episodes=b, clean=dict(link=8, keep="edge", min_px=300)),
             "side": view(bs, "side", mount="jaw", role="wrist", clean=dict(link=4, keep="edge", min_px=300))}
    return dict(real="so101_sort_real", views=views, base_pos=bs.BASE_POS, base_quat=bs.BASE_QUAT,
                real_to_sim=tw.real_to_sim, real_mask="auto")
